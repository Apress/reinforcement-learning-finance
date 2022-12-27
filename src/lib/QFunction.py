from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class QFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, training_inputs, training_targets, *args, **kwargs):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def predict(self, inputs, *args, **kwargs):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def nextOptimumAction(self, states):
        raise NotImplementedError('Not Implemented')

    @abstractmethod
    def setParameters(self, params):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def getParameters(self):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def getParamCorrections(self, params, new_params):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def applyParamCorrections(self, corrections):
        raise NotImplementedError('Not implemented')

    def softUpdateParams(self, target_params, weight):
        params = self.getParameters()
        params = weight * target_params + (1.0 - weight) * params
        self.setParameters(params)


class QTable(QFunction):
    """ Q function represented as a table """

    def __init__(self, num_states, num_actions, learning_rate=0.02):
        """
        Initialize Q table
        :param num_states:
        :param num_actions:
        """
        self.qTable = np.zeros((num_states, num_actions), dtype=np.float32)
        self.nStates = num_states
        self.nActions = num_actions
        self.alpha = learning_rate

    def predict(self, inputs, *args, **kwargs):
        """
        Lookup Q table and get action values
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """
        return np.argmax(self.qTable[inputs, :], axis=1)

    def fit(self, training_inputs, training_targets, *args, **kwargs):
        """
        Fit Q table to the training data
        :param training_inputs:
        :param training_targets:
        :param args:
        :param kwargs:
        """
        for input, target in zip(training_inputs, training_targets):
            max_value = np.max(self.qTable[input, :])
            self.qTable[input, target] = (1 - self.alpha)*self.qTable[input, target] + self.alpha*max_value

    def randomAction(self):
        """
        Returns a random action
        :return:
        """
        return np.random.choice(self.nActions)

    def nextOptimumAction(self, states):
        """
        Returns the next optimum action from a state
        :param state: Specified state (array or list of states)
        :return: Optimum action
        """
        return np.argmax(self.qTable[states, :], axis=1)

    def setParameters(self, params):
        self.qTable[:, :] = params

    def getParameters(self):
        return self.qTable

    def getParamCorrections(self, params, new_params):
        return np.subtract(new_params, params)

    def applyParamCorrections(self, corrections):
        self.setParameters(np.add(self.qTable, corrections))


class QNeuralNet(QFunction):
    """ Action value function represented by a neural network.
        Input dimension is the number of features, output dimension is equal to the number of actions
        Output represents the probability (un-normalized) of each action.
        Dimension of action space is assumed to be 1
        Actions values are {0, 1, 2, ..., nactions-1}
    """
    def __init__(self, neuralnet, state_dimension, num_actions):
        """
        Initialize the action value function with a neural network
        :param neuralnet: keras.Sequential network. Must be compiled model
        :param state_dimension:
        :param num_actions: number of actions
        """
        assert isinstance(neuralnet, tf.keras.Sequential)
        self.neuralNet = neuralnet
        self.stateDim = state_dimension
        self.nActions = num_actions

    def __deepcopy__(self, memodict={}):
        hid = id(self)
        if hid in memodict:
            return memodict[hid]
        cls = self.__class__
        result = cls.__new__(cls)
        neural_net = tf.keras.models.model_from_json(self.neuralNet.to_json())
        result.__init__(neural_net, self.stateDim, self.nActions)
        memodict[hid] = result
        return result

    def fit(self, training_inputs, training_targets, *args, **kwargs):
        """
        Call fit on neural network.
        Fit the Q value function to a list of Q(state, action) = target values
        :param training_inputs:
        :param training_targets:
        :param args:
        :param kwargs:
        :return: history of fit (metrics)
        """
        return self.neuralNet.fit(training_inputs, training_targets, *args, **kwargs)

    def predict(self, inputs, *args, **kwargs):
        """
        Call predict on neural network
        :param inputs:
        :param args:
        :param kwargs:
        """
        return self.neuralNet.predict(inputs, *args, **kwargs)

    def nextOptimumAction(self, states):
        """
        :param state: nd array of (#observations, input_dimension) shape
        :return: (observations) array with index of best action
        """
        output = self.neuralNet(states)
        return np.argmax(output, axis=1)

    def randomAction(self):
        return np.random.randint(0, high=self.nActions)

    def setParameters(self, params):
        """
        Set configurable parameters of a neural network
        :param params: configurable parameters of value function
        """
        for i, param in enumerate(params):
            self.neuralNet.layers[i].set_weights(param)

    def getParameters(self):
        """
        Get configurable parameters of the neural network representing the value function
        :return: parameters
        """
        params = []
        for layer in self.neuralNet.layers:
            params.append(layer.get_weights())
        return params

    def getParamCorrections(self, params, new_params):
        if isinstance(params, np.ndarray):
            return np.subtract(new_params, params)
        result = []
        for pm, npm in zip(params, new_params):
            result.append(self.getParamCorrections(pm, npm))
        return result

    def applyCorrectionToParam(self, param, correction):
        if isinstance(param, np.ndarray):
            return np.add(param, correction)

        count = 0
        for pm, corr in zip(param, correction):
            param[count] = self.applyCorrectionToParam(pm, corr)
            count += 1
        return param

    def applyParamCorrections(self, corrections):
        new_params = self.getParameters()
        new_params = self.applyCorrectionToParam(new_params, corrections)
        self.setParameters(new_params)
