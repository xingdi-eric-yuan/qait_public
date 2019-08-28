import random
import numpy as np
from collections import namedtuple


# a snapshot of state to be stored in replay memory
qa_Transition = namedtuple('qa_Transition', ('observation_list', 'quest_list', 'answer_strings'))


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0
        self.alpha_rewards, self.beta_rewards = [], []

    def push(self, is_prior=False, reward=0.0, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = qa_Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
            self.alpha_rewards.append(reward)
            if len(self.alpha_rewards) > self.alpha_capacity:
                self.alpha_rewards = self.alpha_rewards[1:]
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = qa_Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity
            self.beta_rewards.append(reward)
            if len(self.beta_rewards) > self.beta_capacity:
                self.beta_rewards = self.beta_rewards[1:]

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        return res

    def avg_rewards(self):
        if len(self.alpha_rewards) == 0 and len(self.beta_rewards) == 0 :
            return 0.0
        return np.mean(self.alpha_rewards + self.beta_rewards)

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
