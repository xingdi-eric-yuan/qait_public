from collections import namedtuple
import numpy as np
import torch


# a snapshot of state to be stored in replay memory
Transition = namedtuple('Transition', ('observation_list', 'quest_list', 'possible_words', 'word_indices', 'reward', 'is_final'))


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0, discount_gamma=1.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.discount_gamma = discount_gamma
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def push(self, is_prior=False, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def get_next_final_pos(self, which_memory, head):
        i = head
        while True:
            if i >= len(which_memory):
                return None
            if which_memory[i].is_final:
                return i
            i += 1
        return None

    def _get_single_transition(self, n, which_memory):
        assert n > 0
        tried_times = 0
        while True:
            tried_times += 1
            if tried_times >= 50:
                return None
            if len(which_memory) <= n:
                return None

            head = np.random.randint(0, len(which_memory) - n)
            # if n is 1, then head can't be is_final
            if n == 1:
                if which_memory[head].is_final:
                    continue
            #  if n > 1, then all except tail can't be is_final
            else:
                if np.any([item.is_final for item in which_memory[head: head + n]]):
                    continue

            next_final = self.get_next_final_pos(which_memory, head)
            if next_final is None:
                continue

            # all good
            obs = which_memory[head].observation_list
            quest = which_memory[head].quest_list
            possible_words = which_memory[head].possible_words
            word_indices = which_memory[head].word_indices
            
            next_obs = which_memory[head + n].observation_list
            next_possible_words = which_memory[head + n].possible_words

            rewards_up_to_next_final = [self.discount_gamma ** i * which_memory[head + i].reward for i in range(next_final - head + 1)]
            reward = torch.sum(torch.stack(rewards_up_to_next_final))

            return (obs, quest, possible_words, word_indices, reward, next_obs, next_possible_words, n)

    def _get_batch(self, n_list, which_memory):
        res = []
        for i in range(len(n_list)):
            output = self._get_single_transition(n_list[i], which_memory)
            if output is None:
                continue
            res.append(output)

        if len(res) == 0:
            return None
        return res

    def get_batch(self, batch_size, multi_step=1):
        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))
        res = []
        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch(np.random.randint(1, multi_step + 1, size=from_alpha), self.alpha_memory)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch(np.random.randint(1, multi_step + 1, size=from_beta), self.beta_memory)
        if res_alpha is None and res_beta is None:
            return None
        if res_alpha is not None:
            res += res_alpha
        if res_beta is not None:
            res += res_beta

        obs_list, quest_list, possible_words_list, word_indices_list = [], [], [], []
        reward_list, next_obs_list, next_possible_words_list, actual_n_list = [], [], [], []

        for item in res:
            obs, quest, possible_words, word_indices, reward, next_obs, next_possible_words, n = item

            obs_list.append(obs)
            quest_list.append(quest)
            possible_words_list.append(possible_words)
            word_indices_list.append(word_indices)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            next_possible_words_list.append(next_possible_words)
            actual_n_list.append(n)

        chosen_indices = list(zip(*word_indices_list))
        chosen_indices = [torch.stack(item, 0) for item in chosen_indices]  # list of batch x 1
        rewards = torch.stack(reward_list, 0)  # batch
        actual_n_list = np.array(actual_n_list)

        return obs_list, quest_list, possible_words_list, chosen_indices, rewards, next_obs_list, next_possible_words_list, actual_n_list

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
