import logging
from enum import Enum

import numpy as np

logging.basicConfig(level=logging.DEBUG)


class ActionSpaceType(Enum):
    DISCRETE = 0,
    CONTINUOUS = 1


class ActionSpace(object):
    def __init__(self, action_space_type, num_discrete_actions=None):
        """
        Initialize the action space
        :param action_space_type: ActionSpaceType
        :param num_discrete_actions: Number of discrete actions. Actions are [0, 1, ... N-1]
        """
        assert isinstance(action_space_type, ActionSpaceType)
        self.actionSpaceType = action_space_type
        self.nActions = num_discrete_actions
        self.logger = logging.getLogger(self.__class__.__name__)

    def getActionSpaceType(self):
        return self.actionSpaceType

    def numActions(self):
        return self.nActions

    def getAllActions(self):
        if self.actionSpaceType == ActionSpaceType.CONTINUOUS:
            raise ValueError("Action space is continuous, cannot get all actions")
        return np.arange(self.nActions)

    def geOptimumActionFromValFunc(self, q_func, curr_state):
        """
        Get optimum action in a state using Q value function
        If action space is discrete, value function takes atet as input and produces value for each discrete action
        If action space is continuous, value function takes (state, action) as input and produces value
        :param q_func: Value function
        :param curr_state: current state: numpy ndarray of 2 dimensions: [#observations, state_dimension] shape
        :return: action yielding optimum reward
        """
        if self.actionSpaceType == ActionSpaceType.CONTINUOUS:
            alpha = 0.1
            max_iter = 1000
            threshold = 1E-2
            curr_action = np.random.random()
            state = curr_state.values()
            input = np.array((2, state.shape[0]+1), dtype=np.float32)
            input[:, 0:-1] = state
            delta = threshold * np.random.random()
            for iter in range(max_iter):
                input[0, -2:] = curr_action
                input[1, -2:] = curr_action + delta
                output = q_func.predict(input)
                deriv = np.divide(output[1] - output[0], delta)
                correction = alpha*deriv
                corr = np.sqrt(np.dot(correction, correction))
                if corr < threshold:
                    return curr_action
                curr_action += correction
            return curr_action
        elif self.actionSpaceType == ActionSpaceType.DISCRETE:
            action_vals = q_func.predict(curr_state)
            return np.argmax(action_vals, axis=1)

    def randomAction(self):
        """
        Generate a random action in the action space
        :return: random action
        """
        if self.actionSpaceType == ActionSpaceType.DISCRETE:
            return np.random.choice(self.nActions)
        elif self.actionSpaceType == ActionSpaceType.CONTINUOUS:
            return np.random.random()
        raise ValueError("Unsupported action space type")
