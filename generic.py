import numpy as np
import torch
import copy
import random
import uuid
import os
import time
import multiprocessing as mp
from os.path import join as pjoin
missing_words = set()


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, enable_cuda=False, type='long'):
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix.copy()).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix.copy()).type(torch.FloatTensor))


def _word_to_id(word, word2id):
    try:
        return word2id[word]
    except KeyError:
        key = word + "_" + str(len(word2id))
        if key not in missing_words:
            print("Warning... %s is not in vocab, vocab size is %d..." % (word, len(word2id)))
            missing_words.add(key)
            with open("missing_words.txt", 'a+') as outfile:
                outfile.write(key + '\n')
                outfile.flush()
        return 1


def _words_to_ids(words, word2id):
    ids = []
    for word in words:
        ids.append(_word_to_id(word, word2id))
    return ids


def preproc(s, tokenizer=None):
    if s is None:
        return "nothing"
    if "$$$$$$$" in s:
        s = s.split("$$$$$$$")[-1]
    if "are carrying:" in s:
        s = " -= inventory =- " + s
    s = s.replace("\n", ' ')
    if s.strip() == "":
        return "nothing"
    s = s.strip()
    if len(s) == 0:
        return "nothing"
    s = " ".join([t.text for t in tokenizer(s)])
    s = s.lower()
    return s


def max_len(list_of_list):
    return max(map(len, list_of_list))


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def ez_gather_dim_1(input, index):
    if len(input.size()) == len(index.size()):
        return input.gather(1, index)
    res = []
    for i in range(input.size(0)):
        res.append(input[i][index[i][0]])
    return torch.stack(res, 0)


def list_of_token_list_to_char_input(list_of_token_list, char2id):
    batch_size = len(list_of_token_list)
    max_token_number = max_len(list_of_token_list)
    max_char_number = max([max_len(item) for item in list_of_token_list])
    if max_char_number < 6:
        max_char_number = 6
    res = np.zeros((batch_size, max_token_number, max_char_number), dtype='int32')
    for i in range(batch_size):
        for j in range(len(list_of_token_list[i])):
            for k in range(len(list_of_token_list[i][j])):
                res[i][j][k] = _word_to_id(list_of_token_list[i][j][k], char2id)
    return res


class HistoryScoreCache(object):

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class ObservationPool(object):

    def __init__(self, capacity=1):
        self.capacity = capacity

    def identical_with_history(self, new_stuff, list_of_old_stuff):
        new_obs = new_stuff.split("<|>")[1].strip()
        new_feedback = new_stuff.split("<|>")[2].strip()
        for i in range(len(list_of_old_stuff)):
            if new_stuff == list_of_old_stuff[i]:
                return True
            # prev_act <|> obs <|> feedback
            # if obs and feedback are seen before, drop it
            if new_obs in list_of_old_stuff[i] and new_feedback in list_of_old_stuff[i]:
                return True
        return False

    def push_batch(self, stuff):
        assert len(stuff) == len(self.memory)
        for i in range(len(stuff)):
            if not self.identical_with_history(stuff[i], self.memory[i]):
                self.memory[i].append(stuff[i])
            if len(self.memory[i]) > self.capacity:
                self.memory[i] = self.memory[i][-self.capacity:]

    def push_one(self, which, stuff):
        assert which < len(self.memory)
        if not self.identical_with_history(stuff, self.memory[which]):
            self.memory[which].append(stuff)
        if len(self.memory[which]) > self.capacity:
            self.memory[which] = self.memory[which][-self.capacity:]

    def get_last(self):
        return [item[-1] for item in self.memory]

    def get(self, which=None):
        if which is not None:
            assert which < len(self.memory)
            # prev_act <|> obs <|> feedback
            output = " <|> ".join(self.memory[which])
            return output

        output = []
        for i in range(len(self.memory)):
            output.append(" <|> ".join(self.memory[i]))
        return output

    def get_sent_list(self):
        return copy.copy(self.memory)

    def reset(self, batch_size):
        self.memory = []
        for _ in range(batch_size):
            self.memory.append([])

    def __len__(self):
        return len(self.memory)
