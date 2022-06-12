import torch
import torch.nn.functional as F
import scipy.signal
import numpy as np


def discounted_cumsum(x, gamma):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


class GAE:
    def __init__(self, lmbda, gamma):
        self.lmbda = lmbda
        self.gamma = gamma

    def gae(self, rewards, values, last_val=0):
        rewards = np.append(rewards, last_val)
        values = np.append(values, last_val)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = discounted_cumsum(deltas, self.gamma * self.lmbda)
        returns = discounted_cumsum(rewards, self.gamma)[:-1]

        # normalize here or in mini batch...?
        # advantages_mean, advantages_std = np.mean(advantages), np.std(advantages)
        # advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)

        return advantages, returns


class PPO:
    def __init__(self, logger, k_epochs, mini_batch_size, entropy_coeff, value_coeff, actor_coeff, grad_clip=0.5, policy_clip=-1, value_clip_param=-1, device=torch.device("cpu")):
        super().__init__()
        self.logger = logger
        self.k_epochs = k_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.actor_coeff = actor_coeff
        self.grad_clip = grad_clip
        self.policy_clip = policy_clip
        self.value_clip_param = value_clip_param
        self.device = device

    def get_actor_loss(self, new_log_probs, log_probs, advantages):
        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        if self.policy_clip != -1:
            surr2 = torch.clip(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages
            return -torch.min(surr1, surr2).mean()
        else:
            return -surr1.mean()

    def get_value_loss(self, values, returns, old_values):
        # could try huber, less sensitive to outliers, but "outliers" can occur as a good training signal (new exploration)
        l1 = F.mse_loss(returns, values)
        if self.value_clip_param != -1:
            l2 = F.mse_loss(returns, old_values + torch.clip(values - old_values, -self.value_clip_param, self.value_clip_param))
            return torch.max(l1, l2)
        else:
            return l1

    def update(self, agent, batch):
        observations, categorical_actions, old_categorical_log_probs, gaussian_actions, old_gaussian_log_probs, old_values, advantages, returns = batch
        categorical_actions = categorical_actions.long()

        for k in range(self.k_epochs):
            tot_loss, ent_loss, act_loss, cri_loss = self.mini_batch_update(agent, observations, categorical_actions, old_categorical_log_probs, gaussian_actions, old_gaussian_log_probs, old_values, advantages, returns)
            self.logger.add_scalar("loss/total_loss", torch.Tensor(tot_loss).mean().cpu().numpy().item(), global_step=self.logger.total_time)
            self.logger.add_scalar("loss/entropy_loss", torch.Tensor(ent_loss).mean().cpu().numpy().item(), global_step=self.logger.total_time)
            self.logger.add_scalar("loss/actor_loss", torch.Tensor(act_loss).mean().cpu().numpy().item(), global_step=self.logger.total_time)
            self.logger.add_scalar("loss/critic_loss", torch.Tensor(cri_loss).mean().cpu().numpy().item(), global_step=self.logger.total_time)
            self.logger.add_scalar("train/std", torch.exp(agent.actor.gaussian_actor.log_std).mean().item(), global_step=self.logger.total_time)
            self.logger.updates += 1

    def mini_batch_update(self, agent, observations, categorical_actions, old_categorical_log_probs, gaussian_actions, old_gaussian_log_probs, old_values, advantages, returns):
        # break single batch update into mini batches to fit on gpu
        total_loss_log = []
        entropy_loss_log = []
        actor_loss_log = []
        critic_loss_log = []

        batch_size = observations.shape[0]
        if batch_size > self.mini_batch_size:
            num_iterations = batch_size // self.mini_batch_size
            mini_batch_size = self.mini_batch_size
        else:
            num_iterations = 1
            mini_batch_size = batch_size

        sample_indices = torch.randperm(mini_batch_size * num_iterations).to(self.device)
        start_idx = 0

        for i in range(num_iterations):
            observation_batch = observations[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device)
            categorical_action_batch = categorical_actions[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device).long()
            gaussian_action_batch = gaussian_actions[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device)
            old_gaussian_log_probs_batch = old_gaussian_log_probs[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device)
            old_categorical_log_probs_batch = old_categorical_log_probs[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device)
            advantages_batch = advantages[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device)
            returns_batch = returns[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device)
            old_values_batch = old_values[sample_indices[start_idx: start_idx + mini_batch_size]].to(self.device)

            # normalize here or when calculating returns...?
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            new_categorical_log_probs, categorical_entropy, new_gaussian_log_probs, gaussian_entropy, values = agent.evaluate_actions(observation_batch, categorical_action_batch, gaussian_action_batch)
            # losses
            categorical_loss = self.get_actor_loss(new_categorical_log_probs, old_categorical_log_probs_batch, advantages_batch)
            gaussian_loss = self.get_actor_loss(new_gaussian_log_probs, old_gaussian_log_probs_batch, advantages_batch)
            actor_loss = (categorical_loss + gaussian_loss)
            entropy_loss = (-torch.mean(categorical_entropy) - torch.mean(gaussian_entropy)) * 0.5
            values = values.view(-1)
            critic_loss = self.get_value_loss(values, returns_batch, old_values_batch)
            total_loss = self.actor_coeff * actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss

            # update
            agent.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_clip)
            agent.optimizer.step()
            if agent.scheduler:
                agent.scheduler.step()
            start_idx += mini_batch_size

            total_loss_log.append(total_loss.item())
            entropy_loss_log.append(entropy_loss.item())
            actor_loss_log.append(actor_loss.item())
            critic_loss_log.append(critic_loss.item())

        return total_loss_log, entropy_loss_log, actor_loss_log, critic_loss_log
