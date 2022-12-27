from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
import gym
import numpy as np
from typing import List, Tuple
import tensorflow as tf


class StateAndRewardEmulator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, state, action):
        raise NotImplementedError("Subclass needs to implement")

    @abstractmethod
    def tfEnvStep(self, action: tf.Tensor) -> List[tf.Tensor]:
        raise NotImplementedError("Subclass needs to implement")

    def setInitialState(self, state):
        pass


class AIGymEmulator(StateAndRewardEmulator):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.reset()

    def step(self, state, action):
        """ Returns next_state, reward, done, info """
        if (state is not None) and (np.equal(self.env.state, state).sum() != state.shape[0]):
            self.env.reset()
            self.env.state = self.env.unwrapped.state = state
        return self.env.step(action)

    def envStep(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, _ = self.env.step(action)
        return state.astype(np.float32), np.array(reward, dtype=np.float32), np.array(done, dtype=np.int32)

    # wrap open AI gym's env.step call as an operation in Tensorflow function to include in a callable TensorFlow graph
    def tfEnvStep(self, action: tf.Tensor) -> List[tf.Tensor]:
        """ Return next state, reward, done . Ignores the last info value """
        return tf.numpy_function(self.envStep, [action], [tf.float32, tf.float32, tf.int32])