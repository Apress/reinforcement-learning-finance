from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf

import src.lib.State as st


class ValFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, training_inputs, training_targets, *args, **kwargs):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def predict(self, input, *args, **kwargs):
        raise NotImplementedError('Not implemented')

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


class VFuncTable(ValFunction):
    """ Value function represented as a mapping from states to values """
    def __init__(self, states, learning_rate=0.1):
        """
        Initialize the value function table
        :param states:
        :param learning_rate:
        """
        self.table = np.zeros(len(states), dtype=np.float32)
        self.states =states
        self.alpha = learning_rate
        self.stateDict = {state: i for i, state in enumerate(states)}

    def fit(self, training_inputs, training_targets, *args, **kwargs):
        """
        Train the value function
        :param training_inputs:
        :param training_targets:
        :param args:
        :param kwargs:
        """
        for i in range(training_inputs.shape[0]):
            state = self.stateDict[training_inputs[i,0]]
            self.table[state] = (1 - self.alpha) * self.table[state] + self.alpha * training_targets[i]

    def predict(self, input, *args, **kwargs):
        """
        Predict the value function for specified inputs
        :param input:
        :param args:
        :param kwargs:
        :return:
        """
        return np.array([self.table[self.stateDict[state]] for state in input], dtype=np.float32)

    def setParameters(self, params):
        self.table[:] = params

    def getParameters(self):
        return self.table

    def getParamCorrections(self, params, new_params):
        return np.subtract(new_params, params)

    def applyParamCorrections(self, corrections):
        self.setParameters(np.add(self.table, corrections))


class VNeuralNet(ValFunction):
    """ State value function represented by a neural network.
        Output is the state value
    """
    def __init__(self, neuralnet):
        """
        Initialize the action value function with a neural network
        :param neuralnet: keras.Sequential network. Must be compiled model
        """
        assert isinstance(neuralnet, tf.keras.Sequential)
        self.neuralNet = neuralnet

    def __deepcopy__(self, memodict={}):
        hid = id(self)
        if hid in memodict:
            return memodict[hid]
        cls = self.__class__
        result = cls.__new__(cls)
        neural_net = tf.keras.models.model_from_json(self.neuralNet.to_json())
        result.__init__(neural_net)
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
        weights = [layer.get_weights() for layer in self.neuralNet.layers]
        return weights

    def getParamCorrections(self, params, new_params):
        if isinstance(params, np.ndarray):
            return np.subtract(new_params, params)
        result = []
        for pm, npm in zip(params, new_params):
            if len(pm):
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


class VFuncCustomNeuralNet(VNeuralNet):
    """ If neural netwrok supports a continuum of actions, assume it lies between [0, 1) """

    def __init__(self, neuralnet, optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber()):
        """
        Initialize with a neural net
        :param neuralnet:
        """
        super(VFuncCustomNeuralNet, self).__init__(neuralnet)
        self.optimizer = optimizer
        self.loss = loss

    def value(self, state):
        """
        Get value for a given state
        :param state:
        :return:
        """
        assert isinstance(state, st.State)
        xinp = np.array(state.values(), dtype=np.float32)
        val = self.neuralNet.predict(xinp[np.newaxis, :])[0]
        return val

    def fit(self, training_inputs, training_targets, *args, **kwargs):
        """
        Call fit on neural network.
        Fit the value function to a list of V(state) = target values
        :param training_inputs:
        :param training_targets:
        :param args:
        :param kwargs:
        :return: history of fit (metrics)
        """
        epochs = kwargs.get("epochs", 10)
        results = []
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                output = self.neuralNet(training_inputs)
                loss = self.loss(training_targets, output)
                # loss += sum(self.neuralNet.losses)   # add any losses added by the model

            grads = tape.gradient(loss, self.neuralNet.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.neuralNet.trainable_weights))
            results.append(tf.keras.metrics.Mean(loss).result())

        return results
