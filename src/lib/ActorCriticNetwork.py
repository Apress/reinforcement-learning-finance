from __future__ import absolute_import, print_function

import src.lib.Emulator as Emulator
import tensorflow as tf
from typing import Tuple


class ACNetworkBase(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, ** kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("Subclass needs to implement")

    def train(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError("Subclass needs to implement")

    def getExpectedReturns(self, rewards: tf.Tensor) -> tf.Tensor:
        """ Expected returns """
        ntime = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=ntime)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(ntime):
            reward = rewards[i]
            discounted_sum = reward + self.discountFactor * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]

        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1E-4))
        return returns


class ACNetwork(ACNetworkBase):
    """ Actor-Critic network with separate networks for actor and critic
    Critic produces state value function as output
    Actor is deterministic and produces the action
    """
    def __init__(self, actor_network, critic_network, emulator, discount_factor, max_steps_per_episode,
                 actor_optimizer, critic_optimizer):
        super().__init__()
        assert isinstance(emulator, Emulator.StateAndRewardEmulator)
        self.actor = actor_network
        self.critic = critic_network
        self.emulator = emulator
        self.discountFactor = discount_factor
        self.maxStepsPerEpisode = max_steps_per_episode
        self.actorOptimizer = actor_optimizer
        self.criticOptimizer = critic_optimizer
        self.criticLoss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.actor(inputs), self.critic(inputs)

    def train(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as critic_tape:
            with tf.GradientTape() as actor_tape:
                actions, values, rewards = self.runEpisode(initial_state)
                returns = self.getExpectedReturns(rewards)
                actions, values, returns = [tf.expand_dims(x, 1) for x in [actions, values, returns]]
                advantage = returns - values
                actor_loss = -tf.math.reduce_sum(advantage)
            critic_loss = self.criticLoss(values, returns)

        grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.criticOptimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actorOptimizer.apply_gradients(zip(grads, self.actor.trainable_weights))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward, critic_loss + actor_loss

    def runEpisode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        state = initial_state
        self.emulator.setInitialState(initial_state)
        for t in tf.range(self.maxStepsPerEpisode):
            state = tf.expand_dims(state, 0)
            action, value = self.call(state)

            values = values.write(t, tf.squeeze(value))
            actions = actions.write(t, tf.squeeze(action))

            state, reward, done = self.emulator.tfEnvStep(action)
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break
        return actions.stack(), values.stack(), rewards.stack()


class CombinedACNetwork(ACNetworkBase):
    """ Combined actor critic network
    Actor and critic networks share common layers
    Critic network outputs one value: state value function
    Actor network outputs the probability (unnormalized) of each of a set of DISCRETE actions
    """
    def __init__(self, common_layers, actor_output_layer, critic_output_layer, emulator, discount_factor, optimizer,
                 max_steps_per_episode):
        super(CombinedACNetwork, self).__init__()
        assert isinstance(emulator, Emulator.StateAndRewardEmulator)
        self.commonLayers = common_layers
        self.actor = actor_output_layer
        self.critic = critic_output_layer
        self.emulator = emulator
        self.discountFactor = discount_factor
        self.optimizer = optimizer
        self.criticLoss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.maxStepsPerEpisode = max_steps_per_episode

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.commonLayers[0](inputs)
        for layer in self.commonLayers[1:]:
            x = layer(x)
        return self.actor(x), self.critic(x)

    def loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """ combined actor-critic loss. returns is the target value """
        advantage = returns - values
        action_log_prob = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_prob * advantage)
        critic_loss = self.criticLoss(values, returns)
        return actor_loss + critic_loss

    def runEpisode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        state = initial_state
        self.emulator.setInitialState(state)
        for t in tf.range(self.maxStepsPerEpisode):
            state = tf.expand_dims(state, 0)
            action_logits_t, value = self.call(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            values = values.write(t, tf.squeeze(value))
            action_probs = action_probs.write(t, action_probs_t[0, action])

            state, reward, done = self.emulator.tfEnvStep(action)
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        return action_probs, values, rewards

    def train(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.runEpisode(initial_state)
            returns = self.getExpectedReturns(rewards)
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
            loss = self.loss(action_probs, values, returns)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward, loss

