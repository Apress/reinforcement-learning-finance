import time

import gym
import numpy as np
import pandas as pd
import tensorflow as tf

from src.learner.DQN import DQN
from src.lib.Emulator import AIGymEmulator
from src.lib.Episode import Episode
from src.lib.QFunction import QNeuralNet
from src.lib.Sample import Sample
from src.lib.ReplayBuffer import MostRecentReplayBuffer

tf.random.set_seed(10)
np.random.seed(10)


class CarpoleV0DQN(object):
    def name(self):
        name = self.__class__.__name__
        if name.startswith("CartpoleV0"):
            name = name[10:]
        return name

    def qfuncnetwork(self):
        optimizer = tf.keras.optimizers.Adam()
        #loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        loss = tf.keras.losses.MeanSquaredError()
        qnet = tf.keras.models.Sequential()
        qnet.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(self.nfeatures,)))
        qnet.add(tf.keras.layers.Dense(50, activation='relu'))
        qnet.add(tf.keras.layers.Dense(self.nactions, activation="linear"))
        qnet.compile(optimizer=optimizer, loss=loss)
        return qnet

    def createAgent(self):
        replay_buf = MostRecentReplayBuffer(2*self.minibatchSize)
        return DQN(self.qfunc, self.emulator, self.nfeatures, self.nactions, replay_buffer=replay_buf,
                         discount_factor=self.discountFactor, minibatch_size=self.minibatchSize, epochs_training=20)

    def __init__(self):
        self.nfeatures = 4
        self.nactions = 2
        self.testEpisodes = 50
        self.maxTimeStepsInEpisode = 50
        self.discountFactor = 1
        self.minibatchSize = 20
        qnet = self.qfuncnetwork()
        self.qfunc = QNeuralNet(qnet, self.nfeatures, self.nactions)
        # create emulator
        self.envName = "CartPole-v0"
        self.env = gym.make(self.envName)
        self.emulator = AIGymEmulator(env_name=self.envName)
        self.agent = self.createAgent()
        self.train()

    def generateTrainingEpisodes(self):
        pos = np.arange(-2.0, 2.0, 4.0/10, dtype=np.float32)
        vel = np.array([-0.5, 0, 0.5], dtype=np.float32)
        angle = np.array([-0.4, -0.3, 0, 0.3, 0.4], dtype=np.float32)
        angvel = np.array([-0.5, 0, 0.5], dtype=np.float32)
        episodes = []
        for p in pos:
            for v in vel:
                for a in angle:
                    for aa in angvel:
                        state = np.array([p, v, a, aa], dtype=np.float32)
                        sample = Sample(state, 0, 1, None)
                        episode = Episode([sample])
                        episodes.append(episode)
        return episodes

    def train(self):
        episodes = self.generateTrainingEpisodes()
        return self.agent.fit(episodes)

    def balance(self):
        test_env = self.env
        rewards = []
        for i in range(self.testEpisodes):
            obs0 = test_env.reset()
            tot_reward = 0
            fac = 1
            for j in range(self.maxTimeStepsInEpisode):
                action, qval = self.agent.predict(obs0)
                obs1, reward, done, info = test_env.step(action)
                if done:
                    break
                tot_reward += fac * reward
                obs0 = obs1
                fac *= self.discountFactor
            rewards.append(tot_reward)

        random_agent_rewards = []
        for i in range(self.testEpisodes):
            test_env.reset()
            tot_reward = 0
            fac = 1
            for j in range(self.maxTimeStepsInEpisode):
                action = test_env.action_space.sample()
                obs1, reward, done, info = test_env.step(action)
                if done:
                    break
                tot_reward += fac * reward
                fac *= self.discountFactor
            random_agent_rewards.append(tot_reward)

        result_df = pd.DataFrame({self.name(): rewards, "RandomAgent": random_agent_rewards})
        print(result_df)
        assert (np.mean(rewards) > np.mean(random_agent_rewards))


if __name__ == "__main__":
    cartpole = CarpoleV0DQN()
    cartpole.balance()
