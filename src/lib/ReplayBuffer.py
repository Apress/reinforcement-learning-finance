from __future__ import absolute_import, print_function

import bisect
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from .Sample import Sample


class ReplayBuffer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def addExperience(self, sample, priority=None):
        raise NotImplementedError("Method not implemeted")

    @abstractmethod
    def sampleMinibatch(self, batch_size):
        raise NotImplementedError("Method not implemeted")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Method not implemeted")


class SimpleReplayBuffer(ReplayBuffer):
    ''' Simple replay buffer. If upper limit on capacity is set, will remove elements from beginning (oldest elements)
        before inserting new ones if size of buffer exceeds the specified capacity
    '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def addExperience(self, sample, priority=None):
        self.buffer.append(sample)

    def sampleMinibatch(self, batch_size):
        assert batch_size <= self.capacity
        index = np.random.randint(0, high=len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in index]

    def __len__(self):
        return len(self.buffer)


class MostRecentReplayBuffer(SimpleReplayBuffer):
    """ Returns the most recently added elements """
    def sampleMinibatch(self, batch_size):
        assert batch_size <= self.capacity
        sz = len(self.buffer)
        return [self.buffer[i] for i in range(sz-batch_size, sz)]


class PrioritizedBuffer(ReplayBuffer):
    '''
        Prioritized experience replay buffer based on the buffer proposed by T. Schaul, et. al., 2016
        Priority queue as buffer. Discards the least important experiences.
        Importance is determined by TD error
    '''

    def __init__(self, capacity=None, alpha=0.5, by_rank=True, epsilon=0, seed=None):
        self.alpha = alpha
        self.byRank = by_rank
        self.buffer = {}
        self.capacity = capacity
        self.epsilon = epsilon
        if seed is not None:
            np.random.seed(seed)

    def addExperience(self, sample, priority=None):
        assert isinstance(sample, Sample)
        if priority is None:
            priority = 0
        key = np.abs(priority) + self.epsilon
        if len(self.buffer) == self.capacity:
            del self.buffer[self.buffer.keys()[0]]

        self.buffer[sample] = key

    def sampleMinibatchByPriority(self, batch_size):
        experiences = list(self.buffer.items())
        probs = np.array([e[1] for e in experiences], dtype=np.float32)
        probs = np.divide(probs, probs.sum())
        sample = np.random.random(batch_size)
        index = [bisect.bisect_left(probs, s) for s in sample]
        return [experiences[i][0] for i in index]

    def sampleMinibatchByRank(self, batch_size):
        experiences = sorted(list(self.buffer.items()), key=lambda x: -x[1])
        ranks = [(i+1,e) for i,e in enumerate(experiences)]
        probs = [r[0]**(-self.alpha) for r in ranks]
        sum_probs = sum(probs)
        probs = [p/sum_probs for p in probs]
        probs = np.cumsum(np.array(probs, dtype=np.float32))
        samples = np.random.random(batch_size)
        index = [bisect.bisect_left(probs, s) for s in samples]
        return [experiences[i][0] for i in index]

    def sampleMinibatch(self, batch_size):
        if self.byRank:
            return self.sampleMinibatchByRank(batch_size)
        return self.sampleMinibatchByPriority(batch_size)

    def updateBuffer(self, minibatch_samples, new_priorities):
        for sample, priority in zip(minibatch_samples, new_priorities):
            self.buffer[sample] = priority

    def __len__(self):
        return len(self.buffer)