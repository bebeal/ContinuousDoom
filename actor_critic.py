import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from modules import ConvFeatureExtractor, NatureCNN
import numpy as np


def sum_independent_dims(tensor):
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class CategoricalActor(nn.Module):
    def __init__(self, h_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(True),
            nn.Linear(h_dim // 2, act_dim),
        )

        self.distribution = distributions.Categorical

    def get_distribution_parameters(self, features):
        logits = self.net(features)
        return logits

    def get_distribution(self, features):
        logits = self.get_distribution_parameters(features)
        return self.distribution(logits=logits)

    def evaluate_actions(self, features, actions):
        distribution = self.get_distribution(features)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy

    def forward(self, features):
        distribution = self.get_distribution(features)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def predict(self, features):
        with torch.no_grad():
            return torch.argmax(self.get_distribution(features).probs, dim=1)


class GaussianActor(nn.Module):
    def __init__(self, h_dim, act_dim, log_std_init=-0.001):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init, requires_grad=True)
        self.mu = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.Tanh(),
            nn.Linear(h_dim // 2, act_dim),
        )

        self.distribution = distributions.Normal

    def get_distribution_parameters(self, features):
        mean = self.mu(features)
        action_std = torch.ones_like(mean) * self.log_std.exp()
        return mean, action_std

    def get_distribution(self, features):
        mean, std = self.get_distribution_parameters(features)
        return self.distribution(mean, std)

    def get_log_prob(self, features, actions):
        distribution = self.get_distribution(features)
        return sum_independent_dims(distribution.log_prob(actions))

    def evaluate_actions(self, features, actions):
        distribution = self.get_distribution(features)
        log_prob = sum_independent_dims(distribution.log_prob(actions))
        entropy = sum_independent_dims(distribution.entropy())
        return log_prob, entropy

    def forward(self, features):
        distribution = self.get_distribution(features)
        action = distribution.rsample()
        log_prob = sum_independent_dims(distribution.log_prob(action))
        return action, log_prob

    def predict(self, features):
        with torch.no_grad():
            return self.get_distribution(features).mean


class Actor(nn.Module):
    def __init__(self, h_dim, num_discrete, num_continuous, log_std_init=-0.001):
        super().__init__()
        self.categorical_actor = CategoricalActor(h_dim=h_dim, act_dim=num_discrete)
        self.gaussian_actor = GaussianActor(h_dim=h_dim, act_dim=num_continuous, log_std_init=log_std_init)

        weights_init(self, gain=0.01)

    def evaluate_actions(self, features, categorical_actions, gaussian_actions):
        categorical_log_probs, categorical_entropy = self.categorical_actor.evaluate_actions(features, categorical_actions)
        gaussian_log_probs, gaussian_entropy = self.gaussian_actor.evaluate_actions(features, gaussian_actions)
        return categorical_log_probs, categorical_entropy, gaussian_log_probs, gaussian_entropy

    def forward(self, features):
        categorical_actions, categorical_log_probs = self.categorical_actor(features)
        gaussian_actions, gaussian_log_probs = self.gaussian_actor(features)
        return categorical_actions, categorical_log_probs, gaussian_actions, gaussian_log_probs

    def predict(self, features):
        with torch.no_grad():
            categorical_actions = self.categorical_actor.predict(features)
            gaussian_actions = self.gaussian_actor.predict(features)
            return categorical_actions, gaussian_actions


class Critic(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(True),
            nn.Linear(h_dim // 2, 1),
        )

        weights_init(self, 1)

    def forward(self, features):
        return self.value(features)


class ActorCritic(nn.Module):
    def __init__(self, input_dims, h_dim, num_discrete, num_continuous, lr=0.001, T_max=1000000, log_std_init=0, weight_decay=1e-6, feature_extractor=NatureCNN):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            feature_extractor(input_dims, h_dim),
        )
        self.actor = Actor(h_dim=h_dim, num_discrete=num_discrete, num_continuous=num_continuous, log_std_init=log_std_init)
        self.critic = Critic(h_dim=h_dim)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        weights_init(self.feature_extractor, np.sqrt(2))

    def evaluate_actions(self, obs, categorical_actions, gaussian_actions):
        features = self.feature_extractor(obs)
        categorical_log_probs, categorical_entropy, gaussian_log_probs, gaussian_entropy = self.actor.evaluate_actions(features, categorical_actions, gaussian_actions)
        values = self.critic(features)
        return categorical_log_probs, categorical_entropy, gaussian_log_probs, gaussian_entropy, values

    def predict(self, obs):
        """
        Inference mode
        Only queries actor, not critic
        uses just means, no std for distributions if applicable
        """
        with torch.no_grad():
            features = self.feature_extractor(obs)
            return self.actor.predict(features)

    def forward(self, obs):
        features = self.feature_extractor(obs)
        categorical_actions, categorical_log_probs, gaussian_actions, gaussian_log_probs = self.actor(features)
        values = self.critic(features)
        return categorical_actions, categorical_log_probs, gaussian_actions, gaussian_log_probs, values

    def predict_values(self, obs):
        """
        Inference mode
        Only queries critic, not actor
        """
        with torch.no_grad():
            features = self.feature_extractor(obs)
            return self.critic(features)


def weights_init(model, gain=1):
    """
    Orthogonal initialization
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.zero_()


class DummyActorCritic(nn.Module):
    def __init__(self):
        super(DummyActorCritic, self).__init__()

    def evaluate_actions(self, obs, categorical_actions, gaussian_actions, t):
        return None

    def predict(self, obs, t):
        return None

    def forward(self, obs, t):
        return None

    def predict_values(self, obs, t):
        return None
