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

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import logging
import time
from typing import Dict, Any, List
from .rl_module import PrioritizedReplayBuffer, RLEnvironment
from ..utils import utils

class QNetwork(Model):
    def __init__(self, features: List[int], action_dim: int):
        super(QNetwork, self).__init__()
        self.hidden_layers = [layers.Dense(f, activation='relu') for f in features]
        self.output_layer = layers.Dense(action_dim)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class SelfCuringRLAgent:
    def __init__(self, features: List[int], action_dim: int, learning_rate: float = 1e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, performance_threshold: float = 0.8,
                 update_interval: int = 86400):  # 24 hours in seconds
        self.features = features
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.performance_threshold = performance_threshold
        self.update_interval = update_interval

        self.q_network = QNetwork(features, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(100000, (features[0],), (action_dim,))
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.q_network(np.expand_dims(state, axis=0))
            return tf.argmax(q_values[0]).numpy()

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        states = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_observations']
        dones = batch['dones']

        # Compute Q-values for current states and actions
        q_values = self.q_network(states)
        q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions.flatten(), self.action_dim), axis=1)

        # Compute next state Q-values and select the best actions
        next_q_values = self.q_network(next_states)
        targets = rewards + self.gamma * tf.reduce_max(next_q_values, axis=1) * (1 - dones)

        # Compute loss
        loss = tf.reduce_mean(tf.keras.losses.Huber()(targets, q_values_selected))

        # Update the model
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions.flatten(), self.action_dim), axis=1)
            loss = tf.keras.losses.Huber()(targets, q_values_selected)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        return loss.numpy()

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        episode_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > self.replay_buffer.batch_size:
                    batch = self.replay_buffer.sample(self.replay_buffer.batch_size)
                    loss = self.update(batch)

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
                logging.info(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")

        self.is_trained = True
        self.performance = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
        self.last_update = time.time()

        return {"final_reward": self.performance, "episode_rewards": episode_rewards}

    def diagnose(self) -> List[str]:
        issues = []
        if not self.is_trained:
            issues.append("Model is not trained")
        if self.performance < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")
        return issues

    def heal(self, env, num_episodes: int, max_steps: int):
        issues = self.diagnose()
        min_episodes = max(5, num_episodes // 10)  # Ensure at least 5 episodes for healing
        for issue in issues:
            if issue == "Model is not trained" or issue == "Model performance is below threshold":
                logging.info(f"Healing issue: {issue}")
                self.train(env, num_episodes, max_steps)
            elif issue == "Model hasn't been updated in 24 hours":
                logging.info(f"Healing issue: {issue}")
                self.update_model(env, min_episodes, max_steps)  # Perform a shorter training session

    def update_model(self, env, num_episodes: int, max_steps: int):
        num_episodes = max(1, num_episodes)  # Ensure at least 1 episode
        training_info = self.train(env, num_episodes, max_steps)
        self.performance = training_info['final_reward']
        self.last_update = time.time()


def create_self_curing_rl_agent(features: List[int], action_dim: int) -> SelfCuringRLAgent:
    return SelfCuringRLAgent(features=features, action_dim=action_dim)


if __name__ == "__main__":
    from .rl_module import RLEnvironment

    logging.basicConfig(level=logging.INFO)
    env = RLEnvironment("CartPole-v1")
    agent = create_self_curing_rl_agent([64, 64], env.action_space.n)

    # Initial training
    training_info = agent.train(env, num_episodes=1000, max_steps=500)
    logging.info(f"Initial training completed. Final reward: {training_info['final_reward']}")

    # Simulate some time passing and performance degradation
    agent.last_update -= 100000  # Simulate 27+ hours passing
    agent.performance = 0.7  # Simulate performance drop

    # Diagnose and heal
    issues = agent.diagnose()
    if issues:
        logging.info(f"Detected issues: {issues}")
        agent.heal(env, num_episodes=500, max_steps=500)
        logging.info(f"Healing completed. New performance: {agent.performance}")
