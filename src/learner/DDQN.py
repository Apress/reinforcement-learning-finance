from copy import deepcopy

import numpy as np
import pandas as pd

from src.learner.DQN import DQN


class DDQN(DQN):
    """ Double Deep Q Network learner """
    def __init__(self, q_function, emulator, state_size, num_actions, discount_factor=0.9, minibatch_size=10,
                 replay_buffer=None, epochs_training=10, decay_rate=10, max_samples_per_episode=50, sync_period=2):
        """
        Initialize double deep Q-network (DDQN) learner.
        :param q_function: Action value function. Object of type QFunction. Could be a TensorFlow model wrapped in
        QNeuralNet class
        :param emulator: An object of type StateAndRewardEmulator gives the reward and next state from
        (state, action) input
        :param state_size: Dimension of state space
        :param num_actions: Number of actions
        :param discount_factor:
        :param minibatch_size: Size of minibatch for stochastic gradient descent
        :param replay_buffer: An object of type ReplayBuffer
        :param epochs_training: Number of epochs used in training
        :param sync_period: Synchronize target network weights from learned network after a specified number of steps
        :param: max_samples_per_episode: Maximum samples in an episode
        """
        super(DDQN, self).__init__(q_function, emulator, state_size, num_actions, discount_factor,
                                   minibatch_size, replay_buffer, epochs_training, decay_rate, max_samples_per_episode)
        self.qFuncTarget = deepcopy(q_function)
        self.syncPeriod = sync_period

    def _improveQFunc(self, df):
        """ Improve Q function using TD(0) learning
            :param df: dataframe to hold or append history of fitting
            :return: dataframe containing the appended history
         """
        if len(self.replayBuffer) < self.minibatchSize:
            return
        minibatch = self.replayBuffer.sampleMinibatch(self.minibatchSize)
        targets = self.trainingTargets
        inputs = self.trainingInputs
        for i, sample in enumerate(minibatch):
            state, action, reward, next_state = sample
            qval = reward
            if next_state is not None:
                optimum_action = np.argmax(self.qFunc.predict(next_state[np.newaxis, :])[0])
                val = self.qFuncTarget.predict(next_state[np.newaxis, :])[0, optimum_action]
                qval = reward + self.discountFactor * val
            targets[i, :] = self.qFunc.predict(state[np.newaxis, :])[0]
            targets[i, action] = qval
            inputs[i, :] = state
        history = self.qFunc.fit(inputs, targets, batch_size=self.minibatchSize, epochs=self.epochs)
        if df is None:
            df = pd.DataFrame(history.history)
        else:
            df = df.append(history.history, ignore_index=True)
        return df

    def _postTimeStepProcessing(self, time_step):
        """
        Post time step processing. Every syncPeriod time steps, update target network weights from lerned network
        :param time_step: time step in the lerning process
        """
        if time_step % self.syncPeriod == 0:
            params = self.qFunc.getParameters()
            self.qFuncTarget.setParameters(params)