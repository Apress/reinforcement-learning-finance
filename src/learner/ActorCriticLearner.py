from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

from src.lib.Episode import Episode
from src.lib.ReinforcementLearner import ReinforcementLearner
from src.lib.ActorCriticNetwork import ACNetworkBase
import tensorflow as tf


class AdvantageActorCriticLearner(ReinforcementLearner):
    """
    A2C learner. Needs a value function because it uses the policy being learned by actor.
    Batch A2:
    1. Sample {si, ai} from pi_theta(a|s)
    2. Fit value function V_phi_pi(s) to samples reward sums
    3. Calculate advantage: A_pi(si, ai) = r(si, ai) + gamma*V_phi_pi(s_i+1) - V_phi_pi(si)
    4. grad_theta(J(theta)) = sum_i(grad_theta log(pi_theta(ai|si) * A_pi(si, ai)
    5. theta += alpha * grad_theta J(theta)

    Online A2C:
    1. Take action a ~ pi_theta(a|s) to get (s, a, r, s')
    2. Update V_phi_pi(s) using target r + gamma * V_phi_pi(s')
    3. Calculate advantage: A_pi(si, ai) = r(si, ai) + gamma*V_phi_pi(s_i+1) - V_phi_pi(si)
    4. grad_theta(J(theta)) = sum_i(grad_theta log(pi_theta(ai|si) * A_pi(si, ai)
    5. theta += alpha * grad_theta J(theta)

    Handles the batch A2C version. Online version is a special case of batch version with batch size = 1
    """

    def __init__(self, ac_network, discrete_actions=True):
        """
        Initialize A2C learner
        :param ac_network: Actor-Critic network. Must be an instance of ACNetworkBase
        :param discrete_actions: Is action space discrete?
        """
        assert isinstance(ac_network, ACNetworkBase)
        self.acNetwork = ac_network
        self.discreteActions = discrete_actions

    def fit(self, episodes: list) -> pd.DataFrame:
        assert len(episodes)
        nelements = sum([len(ep) for ep in episodes])
        rewards = np.zeros(nelements, dtype=np.float32)
        losses = np.zeros(nelements, dtype=np.float32)
        assert isinstance(episodes[0], Episode)
        count = 0
        for episode in episodes:
            for initial_sample in episode:
                state, action, reward, next_state = initial_sample
                initial_state = tf.constant(state, dtype=tf.float32)
                episode_reward, loss = self.acNetwork.train(initial_state)
                rewards[count] = episode_reward.numpy()
                losses[count] = loss.numpy()
                count += 1

        return pd.DataFrame({"rewards": rewards, "loss": losses})

    def predict(self, curr_state):
        assert len(curr_state.shape) == 1
        state = tf.constant(curr_state[np.newaxis, :], dtype=tf.float32)
        action_logits_t, value = self.acNetwork.call(state)
        if self.discreteActions:
            action = tf.math.argmax(action_logits_t, axis=1)
            return action.numpy()[0], value.numpy()[0, 0]
        return action_logits_t.numpy()[0, :], value.numpy()[0, 0]

