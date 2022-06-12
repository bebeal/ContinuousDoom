import os
from os import path

import torch
import torch.utils.tensorboard as tb
from VizDoomGym.VizDoomEnv import DoomEnv
import utils
from actor_critic import ActorCritic
from config import get_config
from ppo import PPO, GAE
from memory_buffer import MemoryBuffer
import numpy as np
from modules import ConvFeatureExtractor


def eval_agent(config, agent, logger):
    if config.eval_episodes > 0:
        agent.eval()
        device = config.device
        map_discrete = utils.base_buttons_with_concat(5, [])

        env = DoomEnv(config="defend_the_center.cfg", frame_skip=config.frame_skip, down_sample=config.down_sample, frame_stack=config.frame_stack, multiple_buttons=True)
        timeout = 1024

        total_rewards = []
        total_lengths = []

        for i in range(config.eval_episodes):
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done and episode_length < timeout:
                # env.render()
                categorical_action, gaussian_action = agent.predict(observation[None].to(device))
                categorical_action, gaussian_action = categorical_action.cpu()[0].item(), gaussian_action.cpu()[0].item()

                env_action = map_discrete[categorical_action].copy()
                env_action[0] = gaussian_action
                next_observation, reward, done, _ = env.step(np.array(env_action))
                episode_reward += reward
                episode_length += 1
                observation = next_observation

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)

        mean_reward = torch.tensor(total_rewards).float().mean().item()
        mean_length = torch.tensor(total_lengths).float().mean().item()

        logger.add_scalar("eval/mean_reward", mean_reward, logger.total_time)
        logger.add_scalar("eval/mean_length", mean_length, logger.total_time)

        env.close()
        agent.train()


def train(config):
    os.makedirs(config.log_dir, exist_ok=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, "train"), flush_secs=1)
    map_discrete = utils.base_buttons_with_concat(5, [])
    env = DoomEnv(config="defend_the_center.cfg", frame_skip=config.frame_skip, down_sample=config.down_sample, frame_stack=config.frame_stack, multiple_buttons=True)
    num_updates = ((config.end // config.num_frames_per_update) * (config.num_frames_per_update // config.mini_batch_size)) * config.k_epochs
    agent = ActorCritic(env.observation_space.shape[0], h_dim=config.hdim, num_discrete=len(map_discrete), num_continuous=1, lr=config.lr, T_max=num_updates, log_std_init=config.log_std_init, weight_decay=config.weight_decay, feature_extractor=ConvFeatureExtractor)
    agent = agent.to(config.device)
    ppo = PPO(logger=logger, k_epochs=config.k_epochs, mini_batch_size=config.mini_batch_size, entropy_coeff=config.entropy_coeff, value_coeff=config.value_coeff, actor_coeff=config.actor_coeff, grad_clip=config.grad_clip, policy_clip=config.clip_param, value_clip=config.value_clip, device=config.device)
    gae = GAE(config.gamma, config.lmbda)
    memory = MemoryBuffer(config.num_frames_per_update, env.observation_space.shape, device=config.device)

    epoch = 0
    logger.total_time = 0
    logger.updates = 0
    train_time = 0
    mean_rew, mean_len = [], []
    while logger.total_time < config.end:
        observation = env.reset()
        episode_length = 0
        episode_reward = 0
        done = False

        # Plays through an entire episode, recording data to memory buffer
        while not done and episode_length < config.max_episode_length and train_time < config.num_frames_per_update:
            with torch.no_grad():
                categorical_action, categorical_log_prob, gaussian_action, gaussian_log_prob, value = agent(observation[None].to(config.device))
            categorical_action, categorical_log_prob, gaussian_action, gaussian_log_prob, value = categorical_action.clone()[0].cpu().item(), categorical_log_prob.clone()[0].cpu().item(), gaussian_action[0].clone().cpu(), gaussian_log_prob.clone()[0].cpu().item(), value.clone().cpu()[0].item()
            env_action = utils.create_action(gaussian_action, categorical_action, map_discrete)
            next_observation, reward, done, _ = env.step(env_action)

            memory.add(observation.clone().numpy(), categorical_action, categorical_log_prob, gaussian_action, gaussian_log_prob, reward, value)
            logger.total_time += 1
            episode_length += 1
            train_time += 1
            episode_reward += reward

            observation = next_observation
        logger.add_scalar("train/episode_reward", episode_reward, epoch)
        logger.add_scalar("train/episode_length", episode_length, epoch)
        mean_rew.append(episode_reward)
        mean_len.append(episode_length)

        # After each episode, should calculate and store gae returns
        if done:
            last_value = 0
        else:
            with torch.no_grad():
                last_value = agent.predict_values(observation[None].to(config.device))
                last_value = last_value.clone().cpu()[0].item()
        memory.add_episode(gae, last_value)

        # if we've stored enough data, do ppo update
        if train_time >= config.num_frames_per_update:
            agent.train()
            batch = memory.get()
            ppo.update(agent, batch)
            train_time = 0
            memory.reset()
            logger.add_scalar("train/mean_episode_reward", torch.tensor(mean_rew).float().mean().item(), logger.total_time)
            logger.add_scalar("train/mean_episode_len", torch.tensor(mean_len).float().mean().item(), logger.total_time)
            if agent.scheduler:
                logger.add_scalar("train/lr", agent.scheduler.get_last_lr()[0], logger.total_time)
            mean_rew = []
            mean_len = []

        # eval mode
        if epoch % config.every == 0:
            eval_agent(config, agent, logger)
            torch.save(agent.state_dict(), path.join(config.log_dir, "agent.pt"))

        epoch += 1

    torch.save(agent.state_dict(), path.join(config.log_dir, "agent.pt"))


if __name__ == '__main__':
    train(get_config())
