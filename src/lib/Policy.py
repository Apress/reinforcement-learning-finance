from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np
import tensorflow as tf

import src.lib.QFunction
import src.lib.ValFunction


class PolicyType(Enum):
    DETERMINISTIC = 0,
    STOCHASTIC = 1


class Policy(object):
    __metaclass__ = ABCMeta

    TYPE = PolicyType.STOCHASTIC

    @abstractmethod
    def fit(self, training_inputs, *args, **kwargs):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def predict(self, inputs, *args, **kwargs):
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

    @abstractmethod
    def getNextAction(self, states):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def randomAction(self):
        raise NotImplementedError('Not implemented')


class EpsilonSoftGreedyPolicy(Policy):
    """ Epsilon soft greedy policy. Selects the best action from a q function with probability 1 - epsilon,
        and a random action with probability epsilon """

    def __init__(self, num_actions, qfunc, epsilon=1E-4):
        assert isinstance(qfunc, src.lib.QFunction.QFunction)
        self.epsilon = epsilon
        self.qFunc = qfunc
        self.nActions = num_actions

    def getNextAction(self, states):
        observations = states.shape[0]
        ractions = np.random.choice(self.nActions, size=observations)
        rdraws = np.random.random(observations)
        vals = self.qFunc.nextOptimumAction(states)
        return np.where(rdraws < self.epsilon, ractions, vals)

    def randomAction(self):
        return np.random.choice(self.nActions)

    def predict(self, inputs, *args, **kwargs):
        return self.qFunc.predict(inputs, *args, **kwargs)

    def setParameters(self, params):
        self.qFunc.setParameters(params)

    def getParameters(self):
        return self.qFunc.getParameters()

    def fit(self, training_inputs, *args, **kwargs):
        pass

    def getParamCorrections(self, params, new_params):
        return self.qFunc.getParamCorrections

    def applyParamCorrections(self, corrections):
        pass


class GibbsPolicy(Policy):
    """ Stochastic policy over a discrete action space
    Probability of taking an action a in state s is given by:
    p(s,a) = exp(x_s * beta_{s,a}) / sum_a exp(x_s * beta_{s,a})
    """
    def __init__(self, num_states, state_dimension, num_actions, learning_rate=0.1):
        self.coeffArr = np.ndarray((state_dimension, num_actions), dtype=np.float32)
        self.coeffArr = np.random.random(self.coeffArr.shape) * 1e-3
        self.nStates = state_dimension
        self.nActions = num_actions
        self.alpha = learning_rate
        self.TYPE = PolicyType.STOCHASTIC

    def getNextAction(self, states):
        """

        :param states: Two dimensional ndarray of shape (#observations, state_dimension)
        :return:
        """
        vals = np.einsum("ij,jk->ik", states, self.coeffArr)
        return np.argmax(vals, axis=1)

    def getNextActionProbVals(self, states):
        """
        Get probability values for all actions in a given state (processes a batch of states)
        :param states:
        :return: 2 dimensional ndarray giving probability of taking an action in each state
        """
        vals = np.exp(np.einsum("ij,jk->ik", states, self.coeffArr))
        svals = np.sum(vals, axis=1)
        return np.divide(vals, svals[:, np.newaxis])

    def policyGradientStep(self, states, advantages):
        probs = self.getNextActionProbVals(states)
        val = np.multiply(states, probs)
        val = np.multiply(val, advantages)
        self.coeffArr += self.alpha * val

    def setParameters(self, params):
        self.coeffArr = np.copy(params)

    def getParameters(self):
        return self.coeffArr

    def randomAction(self):
        return np.random.choice(self.nActions)

    def fit(self, training_inputs, *args, **kwargs):
        pass

    def predict(self, input, *args, **kwargs):
        return self.getNextActionProbVals(input)

    def getParamCorrections(self, params, new_params):
        return np.subtract(new_params, params)

    def applyParamCorrections(self, corrections):
        self.coeffArr = np.add(self.getParameters(), corrections)


class PolicyNeuralNet(Policy):
    """ Policy represented by a tensorflow neural network """

    def __init__(self, neuralnet, optimizer=tf.keras.optimizers.Adam(), is_stochastic=True,
                 num_actions=None, deriv_delta=0.2, discount_factor=0.99):
        """
        Initialize a tensorflow neural network based policy
        If the policy is deterministic, it takes states as input and predicts the optimum action
        If the policy is stochastic, it predicts the probabilities of different actions. Action space is discrete
        :param neuralnet: An uncompiled tensorflow model.
        :param is_stochastic: Is the policy stochastic or deterministic
        """
        assert isinstance(neuralnet, tf.keras.Model)
        if is_stochastic:  # stochastic policy
            assert isinstance(num_actions, int), "num_actions for discrete policy must be int"
        self.neuralNet = neuralnet
        self.optimizer = optimizer
        self.TYPE = PolicyType.STOCHASTIC if is_stochastic else PolicyType.DETERMINISTIC
        self.nActions = num_actions
        self.gamma = discount_factor
        self.derivDelta = deriv_delta
        self.action_num_to_val = None

    def mapActionNums(self, func):
        self.action_num_to_val = func

    def getNextAction(self, states):
        assert isinstance(states, np.ndarray)
        val = self.neuralNet.predict(states)
        if self.TYPE == PolicyType.STOCHASTIC:
            val = np.argmax(val, axis=1)  #tf.random.categorical(val, 1)
        else:
            val = np.rint(val)
        return val

    def randomAction(self):
        return np.random.random()

    def sacFit(self, training_inputs, epochs, *args, **kwargs):
        learned_q1 = kwargs["learned_q1"]
        learned_q2 = kwargs["learned_q2"]
        results = [None] * epochs
        for epoch in epochs:
            with tf.GradientTape() as tape:
                action_probs = self.neuralNet(training_inputs)
                actions = tf.argmax(action_probs, exis=1)
                values = tf.map_fn(fn=self.action_num_to_val, elems=actions)
                tf.concat(training_inputs, values)
                q1_output = learned_q1(training_inputs)
                q2_output = learned_q2(training_inputs)
                q_val = tf.reduce_min(q1_output, q2_output)
                targets = tf.subtract(q_val, tf.log(action_probs))
                loss = tf.reduce_mean(targets)
            grads = tape.gradient(loss, self.neuralNet.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.neuralNet.trainable_weights))
            results[epoch] = loss.numpy()
        return targets

    def fit(self, training_inputs, *args, **kwargs):
        """
        Use supervised learning to fit the policy
        :param training_inputs: For stochastic policy, needs to be states
        :param args:
        :param kwargs: Must contain advantages
        :return:
        """
        epochs = kwargs.get("epochs", 10)
        results = [None] * epochs
        # Soft Actor-Critic uses a different objective function: Q(s, a) - log pi(a|s)
        if kwargs.get("type", None) == "SAC":
            self.sacFit(training_inputs, epochs, *args, **kwargs)
            return

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                output = self.neuralNet(training_inputs)
                if self.TYPE == PolicyType.STOCHASTIC:
                    advantages = kwargs["advantages"]
                    action_probs = tf.math.sigmoid(output)
                    sum_vals = tf.math.reduce_sum(action_probs, axis=1)
                    sum_vals = tf.math.add(sum_vals, 1E-4)  # avoid 0
                    probs = tf.math.divide(action_probs, tf.expand_dims(sum_vals, axis=1))
                    neg_log = -tf.math.log(probs)
                    mult = tf.math.multiply(neg_log, advantages)
                    loss = tf.math.reduce_mean(mult)
                else:
                    values = kwargs["actionValues"]  # = Q^pi(s, a=\mu_\theta(s))
                    mult = -tf.math.multiply(output, values)
                    loss = tf.math.reduce_mean(mult)
            grads = tape.gradient(loss, self.neuralNet.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.neuralNet.trainable_weights))
            results[epoch] = loss.numpy()
        return results

    def predict(self, input, *args, **kwargs):
        res = self.neuralNet.predict(input, *args, **kwargs)
        return res

    def setParameters(self, params):
        """
        Set configurable parameters of a neural network
        :param params: configurable parameters of policy function
        """
        for i, param in enumerate(params):
            self.neuralNet.layers[i].set_weights(param)

    def getParameters(self):
        """
         Get configurable parameters of the neural network representing the policy
         :return: parameters
        """
        weights = [layer.get_weights() for layer in self.neuralNet.layers]
        return weights

    def getParamCorrections(self, params, new_params):
        """
        Get corrections from new_params and params by effectively taking a diff between them. Can have ragged shapes
        :param params:
        :param new_params:
        :return: corrections calculated as new_params - params
        """
        if isinstance(params, np.ndarray):
            return np.subtract(new_params, params)
        result = []
        for pm, npm in zip(params, new_params):
            if len(pm):
                result.append(self.getParamCorrections(pm, npm))
        return result

    def _applyCorrectionToParam(self, param, correction):
        """
        Apply provided correction to parameters passed in param
        :param param: Parameters to apply correction to
        :param correction: Correction to apply
        :return: parameters with applied correction
        """
        if isinstance(param, np.ndarray):
            return np.add(param, correction)

        count = 0
        for pm, corr in zip(param, correction):
            param[count] = self._applyCorrectionToParam(pm, corr)
            count += 1
        return param

    def applyParamCorrections(self, corrections):
        """
        Apply provided corrections to network parameters and change the network parameters to corrected ones
        :param corrections: Corrections to apply
        """
        new_params = self.getParameters()
        new_params = self._applyCorrectionToParam(new_params, corrections)
        self.setParameters(new_params)
