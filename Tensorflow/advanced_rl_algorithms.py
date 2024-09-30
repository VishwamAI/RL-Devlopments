# MIT License
#
# Copyright (c) 2024 VishwamAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Advanced Reinforcement Learning Algorithms Module - TensorFlow Version

This module implements advanced reinforcement learning algorithms including
Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG) using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Actor(tf.keras.Model):
    """Actor network for PPO and DDPG in TensorFlow."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.layer_norm1 = layers.LayerNormalization()
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.layer_norm2 = layers.LayerNormalization()
        self.output_layer = layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.layer_norm1(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    """Critic network for PPO and DDPG in TensorFlow."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.layer_norm1 = layers.LayerNormalization()
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.layer_norm2 = layers.LayerNormalization()
        self.output_layer = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        return self.output_layer(x)


class Critic(tf.keras.Model):
    """Critic network with improved architecture for SAC and TD3 in TensorFlow."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.layer_norm1 = layers.LayerNormalization()
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.layer_norm2 = layers.LayerNormalization()
        self.output_layer = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        return self.output_layer(x)


class SACAgent:
    """Soft Actor-Critic (SAC) agent implementation in TensorFlow."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)

        # Initialize target networks
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def select_action(self, state):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action = self.actor(state)
        return action.numpy().flatten()

    def update(self, replay_buffer, batch_size=64):
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward.reshape(-1, 1), dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        done = tf.convert_to_tensor(done.reshape(-1, 1), dtype=tf.float32)

        # Critic update
        with tf.GradientTape(persistent=True) as tape:
            next_action = self.actor(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * tf.minimum(target_q1, target_q2)

            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)

            critic_loss = tf.reduce_mean(tf.square(current_q1 - target_q)) + tf.reduce_mean(tf.square(current_q2 - target_q))

        critic_grads1 = tape.gradient(critic_loss, self.critic1.trainable_variables)
        critic_grads2 = tape.gradient(critic_loss, self.critic2.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(critic_grads1, self.critic1.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads2, self.critic2.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            new_actions = self.actor(state)
            actor_loss = -tf.reduce_mean(self.critic1(state, new_actions))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update target networks
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

    def _soft_update(self, source_net, target_net):
        target_weights = target_net.get_weights()
        source_weights = source_net.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * source_weights[i] + (1 - self.tau) * target_weights[i]
        target_net.set_weights(target_weights)


# Example for creating an SAC Agent
if __name__ == "__main__":
    state_dim = 33  # Example state dimension
    action_dim = 4  # Example action dimension
    agent = SACAgent(state_dim, action_dim)

    # Example usage:
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print("Selected Action:", action)
