from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod


class ReinforcementLearner(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, episodes):
        raise NotImplementedError("Subclass needs to implement")

    @abstractmethod
    def predict(self, curr_state):
        """
        Predict the next optimal action and total discounted reward starting from curr_state
        :param curr_state: np.ndarray containing a set of current states. Dimension: #batches X #features in a state
        :return Next optimal action and total discounted rewards for each batch
        """
        raise NotImplementedError("Subclass needs to implement")