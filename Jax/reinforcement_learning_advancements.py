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

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import List, Dict, Any
import gym
import logging
import time
from ..utils import utils
from ..core_neural_networks import JAXModel, CNN, LSTMModule, LRNN, MachineLearning
from .rl_module import PrioritizedReplayBuffer, RLAgent, RLEnvironment, train_rl_agent

class QNetwork(nn.Module):
    features: List[int]
    action_dim: int

    @nn.compact
    def __call__(self, x):
        for feature in self.features:
            x = nn.Dense(feature)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

class AdvancedRLAgent(RLAgent):
    def __init__(self, observation_dim: int, action_dim: int, features: List[int] = [64, 64],
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

        self.q_network = QNetwork(self.features, action_dim)
        self.state = None
        self.optimizer = optax.adam(learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, (observation_dim,), (action_dim,), batch_size)
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()

    def init_model(self, key):
        # Initialize the model parameters
        dummy_input = jnp.ones((1, self.observation_dim))
        params = self.q_network.init(key, dummy_input)
        return params

    def select_action(self, state: jnp.ndarray, training: bool = False) -> int:
        if training and jax.random.uniform(jax.random.PRNGKey(0)) < self.epsilon:
            return jax.random.randint(jax.random.PRNGKey(0), (1,), 0, self.action_dim)[0]
        else:
            q_values = self.q_network.apply(self.state.params, state)
            return jnp.argmax(q_values).item()

    def update(self, batch: Dict[str, jnp.ndarray]) -> float:
        def loss_fn(params):
            states = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            next_states = batch['next_observations']
            dones = batch['dones']

            q_values = self.q_network.apply(params, states)
            q_values = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze()

            next_q_values = self.q_network.apply(params, next_states).max(axis=1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

            return jax.numpy.mean(optax.l2_loss(q_values, targets))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)
        return loss

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        episode_rewards = []
        moving_avg_reward = 0
        best_performance = float('-inf')
        window_size = 100  # Size of the moving average window
        no_improvement_count = 0
        max_no_improvement = 50  # Maximum number of episodes without improvement

        key = jax.random.PRNGKey(0)
        self.state = train_state.TrainState.create(apply_fn=self.q_network.apply, params=self.init_model(key), tx=self.optimizer)

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                state_tensor = jnp.array(state, dtype=jnp.float32)
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
                moving_avg_reward = sum(episode_rewards) / (episode + 1)
            else:
                moving_avg_reward = sum(episode_rewards[-window_size:]) / window_size

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

    def diagnose(self) -> List[str]:
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
