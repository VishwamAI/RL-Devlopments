
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
import flax.linen as nn
import optax
import logging
import time
from typing import Dict, Any, List, Tuple
from .rl_module import PrioritizedReplayBuffer, RLEnvironment


class QNetwork(nn.Module):
    action_dim: int
    features: List[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for f in self.features:
            x = nn.Dense(f)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

class SelfCuringRL:
    def __init__(self, env: RLEnvironment, agent: SelfCuringRLAgent):
        self.env = env
        self.agent = agent
        self.logger = logging.getLogger(__name__)

    def train(self, num_episodes: int, max_steps: int) -> None:
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.agent.replay_buffer.add(state, action, reward, next_state, done)

                if len(self.agent.replay_buffer) > self.agent.replay_buffer.batch_size:
                    batch = self.agent.replay_buffer.sample()
                    loss = self.agent.update(batch)

                state = next_state

                if done:
                    break

            self.agent.epsilon = max(self.agent.epsilon_end, self.agent.epsilon * self.agent.epsilon_decay)

            if episode % 10 == 0:
                self.logger.info(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.4f}")

            self._check_and_update_agent()

    def _check_and_update_agent(self) -> None:
        current_time = time.time()
        if current_time - self.agent.last_update >= self.agent.update_interval:
            self.agent.performance = self._evaluate_agent()
            if self.agent.performance < self.agent.performance_threshold:
                self._self_cure()
            self.agent.last_update = current_time

    def _evaluate_agent(self, num_eval_episodes: int = 10) -> float:
        total_rewards = []
        for _ in range(num_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        return jnp.mean(jnp.array(total_rewards))

    def _self_cure(self) -> None:
        self.logger.warning("Performance below threshold. Initiating self-curing process.")
        
        # Reset epsilon for more exploration
        self.agent.epsilon = self.agent.epsilon_start
        
        # Reinitialize the Q-network
        self.agent.params = self.agent.q_network.init(jax.random.PRNGKey(int(time.time())), jnp.ones((1, self.agent.features[0])))
        self.agent.opt_state = self.agent.optimizer.init(self.agent.params)
        
        # Clear a portion of the replay buffer
        self.agent.replay_buffer.clear(fraction=0.5)
        
        self.logger.info("Self-curing process completed. Resuming training with reset parameters.")

    def run(self, num_episodes: int, max_steps: int) -> None:
        self.train(num_episodes, max_steps)
        final_performance = self._evaluate_agent()
        self.logger.info(f"Training completed. Final performance: {final_performance:.2f}")

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

        self.q_network = QNetwork(action_dim=action_dim, features=features)
        self.params = self.q_network.init(jax.random.PRNGKey(0), jnp.ones((1, features[0])))  # Dummy input for init
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self.replay_buffer = PrioritizedReplayBuffer(100000, (features[0],), (action_dim,))
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()

    def select_action(self, state: jnp.ndarray, training: bool = False) -> int:
        if training and jax.random.uniform(jax.random.PRNGKey(0)) < self.epsilon:
            return jax.random.randint(jax.random.PRNGKey(0), (1,), 0, self.action_dim).item()
        else:
            q_values = self.q_network.apply(self.params, state[None, :])
            return jnp.argmax(q_values).item()

    def update(self, batch: Dict[str, jnp.ndarray]) -> float:
        states = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_observations']
        dones = batch['dones']

        # Compute Q-values for current states and actions
        q_values = self.q_network.apply(self.params, states)
        q_values_selected = jnp.take_along_axis(q_values, actions[:, None], axis=1)

        # Compute next state Q-values and select the best actions
        next_q_values = self.q_network.apply(self.params, next_states)
        targets = rewards + self.gamma * jnp.max(next_q_values, axis=1) * (1 - dones)

        # Compute loss
        loss = jax.numpy.mean(optax.l2_loss(q_values_selected, targets[:, None]))

        # Compute gradients
        loss_grad = jax.grad(lambda p: self.q_network.apply(p, states).mean())(self.params)

        # Update parameters
        updates, self.opt_state = self.optimizer.update(loss_grad, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

        return loss

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        episode_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(max_steps):
                state_tensor = jnp.array(state, dtype=jnp.float32)
                action = self.select_action(state_tensor, training=True)
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
