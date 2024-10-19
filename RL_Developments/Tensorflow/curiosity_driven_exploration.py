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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import gymnasium as gym

class ICM(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim=64, beta=0.2):
        super(ICM, self).__init__()
        self.beta = beta
        self.action_dim = action_dim

        # Define the feature encoder
        self.feature_encoder = models.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            layers.Dense(hidden_dim, activation='relu')
        ])

        # Define the inverse model
        self.inverse_model = models.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(hidden_dim * 2,)),
            layers.Dense(action_dim, activation='softmax')
        ])

        # Define the forward model
        self.forward_model = models.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(hidden_dim + action_dim,)),
            layers.Dense(hidden_dim, activation='relu')
        ])

    def call(self, state, next_state, action):
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)

        action_one_hot = tf.one_hot(action, self.action_dim, dtype=tf.float32)
        action_one_hot = tf.reshape(action_one_hot, (tf.shape(state_feat)[0], -1))

        pred_next_state_feat = self.forward_model(tf.concat([state_feat, action_one_hot], axis=1))
        pred_action = self.inverse_model(tf.concat([state_feat, next_state_feat], axis=1))

        return pred_action, pred_next_state_feat, next_state_feat

    def compute_intrinsic_reward(self, state, next_state, action):
        _, pred_next_state_feat, next_state_feat = self.call(state, next_state, action)
        intrinsic_reward = self.beta * 0.5 * tf.reduce_mean(tf.square(pred_next_state_feat - next_state_feat), axis=1)
        return intrinsic_reward

class NoveltyDetector:
    def __init__(self, state_dim, memory_size=1000, novelty_threshold=0.1):
        self.memory = np.zeros((memory_size, state_dim))
        self.memory_index = 0
        self.memory_size = memory_size
        self.novelty_threshold = novelty_threshold

    def compute_novelty(self, state):
        if self.memory_index < self.memory_size:
            distances = np.mean(np.abs(self.memory[:self.memory_index] - state), axis=1)
        else:
            distances = np.mean(np.abs(self.memory - state), axis=1)

        novelty = np.min(distances)
        return novelty

    def update_memory(self, state):
        self.memory[self.memory_index % self.memory_size] = state
        self.memory_index += 1

    def is_novel(self, state):
        novelty = self.compute_novelty(state)
        return novelty > self.novelty_threshold

class CuriosityDrivenAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor model
        self.actor = models.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(action_dim, activation='tanh')
        ])

        # Critic model
        self.critic = models.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(1)
        ])

        # Instantiate the ICM model
        self.icm = ICM(state_dim, action_dim, hidden_dim=hidden_dim)

        # Novelty detector
        self.novelty_detector = NoveltyDetector(state_dim)

        # Optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.icm_optimizer = optimizers.Adam(learning_rate=learning_rate)

    def act(self, state):
        state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_probs = self.actor(state)
        action = tf.argmax(action_probs, axis=1)
        action_one_hot = tf.one_hot(action, self.action_dim, dtype=tf.float32)
        return int(action.numpy()[0]), action_one_hot.numpy().flatten()

    def compute_total_reward(self, state, next_state, action, extrinsic_reward):
        intrinsic_reward = self.icm.compute_intrinsic_reward(state, next_state, action)
        self.novelty_detector.update_memory(next_state)
        novelty = self.novelty_detector.compute_novelty(next_state)
        novelty_reward = novelty if self.novelty_detector.is_novel(next_state) else 0
        total_reward = extrinsic_reward + intrinsic_reward.numpy() + novelty_reward
        return total_reward

    @tf.function
    def update(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        # Update ICM
        with tf.GradientTape() as tape:
            pred_action, pred_next_state_feat, next_state_feat = self.icm(state[None, :], next_state[None, :], action[None])
            inverse_loss = tf.reduce_mean(tf.keras.losses.MSE(action, pred_action))
            forward_loss = 0.5 * tf.reduce_mean(tf.square(pred_next_state_feat - next_state_feat))
            icm_loss = inverse_loss + forward_loss

        gradients = tape.gradient(icm_loss, self.icm.trainable_variables)
        self.icm_optimizer.apply_gradients(zip(gradients, self.icm.trainable_variables))

        # Update critic
        with tf.GradientTape() as tape:
            value = self.critic(state[None, :])
            next_value = self.critic(next_state[None, :])
            td_error = reward + (1 - done) * 0.99 * next_value - value
            critic_loss = tf.reduce_mean(tf.square(td_error))

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(self.critic(state[None, :]))

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        return icm_loss.numpy(), critic_loss.numpy(), actor_loss.numpy()

def train_curiosity_driven_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        episode_reward = 0
        done = False

        while not done:
            action, action_one_hot = agent.act(state)
            next_state, extrinsic_reward, done, _, _ = env.step(action)
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state

            total_reward = agent.compute_total_reward(state, next_state, action, extrinsic_reward)
            agent.update(state, action, total_reward, next_state, done)

            episode_reward += total_reward
            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    return agent

def main():
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = CuriosityDrivenAgent(state_dim, action_dim)
    trained_agent = train_curiosity_driven_agent(env, agent)

if __name__ == "__main__":
    main()
