import random
import pickle as pkl
import torch
from os import path
import numpy as np


class MemoryBuffer:
    """
        This class acts as a ring buffer for our agent to store previous observations,
        actions, rewards, next observations, and done values.
        capacity: max amount of values we want to store in the buffer
    """
    def __init__(self, buffer_size, obs_shape, action_dim, device=torch.device("cpu")):
        self.buffer_size = buffer_size
        self.episode_start = 0
        self.ptr = 0
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.float32)
        self.actions2 = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs2 = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        # self.time = np.zeros((self.buffer_size,1) , dtype=np.float32)
        self.device = device

    # Number of records in the buffer
    def __len__(self):
        return self.ptr

    # Add one record to our buffer
    # expects torch tensors for everything
    def add(self, observation, categorical_action, categorical_log_prob, gaussian_action, gaussian_log_prob, reward, value):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = categorical_action
        self.log_probs[self.ptr] = categorical_log_prob
        self.actions2[self.ptr] = gaussian_action
        self.log_probs2[self.ptr] = gaussian_log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        # self.time[self.ptr] = time
        self.ptr += 1

    # Receive every transition sequence currently storing
    def get(self):
        # , torch.from_numpy(self.time)
        return torch.from_numpy(self.observations), torch.from_numpy(self.actions), torch.from_numpy(self.log_probs), torch.from_numpy(self.actions2), torch.from_numpy(self.log_probs2), torch.from_numpy(self.values), torch.from_numpy(self.advantages), torch.from_numpy(self.returns)

    def add_episode(self, gae, last_value):
        path_slice = slice(self.episode_start, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        advantages, returns = gae.gae(rewards, values, last_value)
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        self.episode_start = self.ptr

    def reset(self):
        self.episode_start = 0
        self.ptr = 0
        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.float32)
        self.actions2 = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs2 = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        # self.time = np.zeros((self.buffer_size, 1), dtype=np.float32)

    def save(self, name='memory_buffer.pkl', directory='saves'):
        l = []
        l.append(self.observations)
        l.append(self.actions)
        l.append(self.actions2)
        l.append(self.log_probs)
        l.append(self.log_probs2)
        l.append(self.rewards)
        l.append(self.values)
        l.append(self.advantages)
        l.append(self.returns)
        l.append(self.time)
        l.append(self.episode_start)
        l.append(self.ptr)
        with open(path.join(directory, name), 'wb') as file:
            pkl.dump(l, file)

    def load(self, name='memory_buffer.pkl', directory='saves'):
        with open(path.join(directory, name), 'rb') as file:
            l = pkl.load(file)
        self.observations = l[0]
        self.actions = l[1]
        self.actions2 = l[2]
        self.log_probs = l[3]
        self.log_probs2 = l[4]
        self.rewards = l[5]
        self.values = l[6]
        self.advantages = l[7]
        self.returns = l[8]
        self.time = l[9]
        self.episode_start = l[10]
        self.ptr = l[11]
