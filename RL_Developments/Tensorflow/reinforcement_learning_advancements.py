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

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import gym
import logging
import time
from ..utils import utils
from ..core_neural_networks import TensorFlowModel, CNN, LSTMModule, LRNN, MachineLearning
from .rl_module import PrioritizedReplayBuffer, RLAgent, RLEnvironment, train_rl_agent

class QNetwork(Model):
    def __init__(self, action_dim: int, features: list):
        super(QNetwork, self).__init__()
        self.dense_layers = [layers.Dense(f, activation='relu') for f in features]
        self.output_layer = layers.Dense(action_dim)

    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

class AdvancedRLAgent(RLAgent):
    def __init__(self, observation_dim: int, action_dim: int, features: list = [64, 64],
                 learning_rate: float = 1e-4, gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                 performance_threshold: float = 0.8, update_interval: int = 86400,
                 buffer_size: int = 100000, batch_size: int = 32):
        super().__init__(observation_dim, action_dim)
        self.features = features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.performance_threshold = performance_threshold
        self.update_interval = update_interval
        self.batch_size = batch_size

        self.q_network = QNetwork(action_dim, features)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, (observation_dim,), (action_dim,), batch_size)
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.q_network(state[np.newaxis, :], training=False)
            return np.argmax(q_values.numpy())

    def update(self, batch: dict) -> float:
        states = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_observations']
        dones = batch['dones']

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values = tf.gather(q_values, actions, axis=1, batch_dims=0)

            next_q_values = self.q_network(next_states)
            targets = rewards + self.gamma * tf.reduce_max(next_q_values, axis=1) * (1 - dones)

            loss = tf.reduce_mean(tf.square(q_values - targets))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        return loss.numpy()

    def train(self, env, num_episodes: int, max_steps: int) -> dict:
        episode_rewards = []
        moving_avg_reward = 0
        best_performance = float('-inf')
        window_size = 100  # Size of the moving average window
        no_improvement_count = 0
        max_no_improvement = 50  # Maximum number of episodes without improvement

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                state_tensor = np.array(state, dtype=np.float32)
                action = self.select_action(state_tensor, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > self.replay_buffer.batch_size:
                    batch = self.replay_buffer.sample()
                    loss = self.update(batch)

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Update moving average
            if episode < window_size:
                moving_avg_reward = np.mean(episode_rewards[:episode + 1])
            else:
                moving_avg_reward = np.mean(episode_rewards[-window_size:])

            if moving_avg_reward > best_performance:
                best_performance = moving_avg_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if (episode + 1) % 10 == 0:
                logging.info(f"Episode {episode + 1}, Avg Reward: {moving_avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")

            # Check for early stopping based on performance threshold and improvement
            if moving_avg_reward >= self.performance_threshold:
                if no_improvement_count >= max_no_improvement:
                    logging.info(f"Performance threshold reached and no improvement for {max_no_improvement} episodes. Stopping at episode {episode + 1}")
                    break
            elif no_improvement_count >= max_no_improvement * 2:
                logging.info(f"No significant improvement for {max_no_improvement * 2} episodes. Stopping at episode {episode + 1}")
                break

        self.is_trained = True
        self.performance = moving_avg_reward  # Use the final moving average as performance
        self.last_update = time.time()

        return {"final_reward": self.performance, "episode_rewards": episode_rewards}

    def diagnose(self) -> list:
        issues = []
        if not self.is_trained:
            issues.append("Model is not trained")
        if self.performance < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")
        return issues

    def heal(self, env, num_episodes: int, max_steps: int, max_attempts: int = 5):
        issues = self.diagnose()
        if issues:
            logging.info(f"Healing issues: {issues}")
            initial_performance = self.performance
            for attempt in range(max_attempts):
                training_info = self.train(env, num_episodes, max_steps)
                new_performance = training_info['final_reward']
                if new_performance > self.performance:
                    self.performance = new_performance
                    self.last_update = time.time()
                    logging.info(f"Healing successful after {attempt + 1} attempts. New performance: {self.performance}")
                    return
                logging.info(f"Healing attempt {attempt + 1} failed. Current performance: {new_performance}")
            logging.warning(f"Failed to improve performance after {max_attempts} attempts. Best performance: {self.performance}")

    def update_model(self, env, num_episodes: int, max_steps: int):
        num_episodes = max(1, num_episodes)
        training_info = self.train(env, num_episodes, max_steps)
        self.performance = training_info['final_reward']
        self.last_update = time.time()
        logging.info(f"Model updated. Current performance: {self.performance}")

# Usage Example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = gym.make("CartPole-v1")
    agent = AdvancedRLAgent(observation_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    agent.train(env, num_episodes=1000, max_steps=200)
