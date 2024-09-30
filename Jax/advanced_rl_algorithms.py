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
Advanced Reinforcement Learning Algorithms Module

This module implements advanced reinforcement learning algorithms including
Soft Actor-Critic (SAC), Twin Delayed DDPG (TD3), Proximal Policy Optimization (PPO),
and Deep Q-Network (DQN).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Callable
import numpy as np

class SACAgent:
    """Soft Actor-Critic (SAC) agent implementation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Initialize actor and critic networks
        self.actor = Actor(action_dim, hidden_dim)
        self.critic1 = Critic(hidden_dim)
        self.critic2 = Critic(hidden_dim)
        self.target_critic1 = Critic(hidden_dim)
        self.target_critic2 = Critic(hidden_dim)

        # Initialize parameters
        self.actor_params = self.actor.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.critic1_params = self.critic1.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.critic2_params = self.critic2.init(jax.random.PRNGKey(2), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.target_critic1_params = self.critic1_params
        self.target_critic2_params = self.critic2_params

        # Initialize optimizers
        self.actor_optimizer = optax.adam(learning_rate)
        self.critic_optimizer = optax.adam(learning_rate)

        # Initialize optimizer states
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic1_opt_state = self.critic_optimizer.init(self.critic1_params)
        self.critic2_opt_state = self.critic_optimizer.init(self.critic2_params)

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select an action using the current policy."""
        action = self.actor.apply(self.actor_params, state)
        return action

    def update(self, batch: Tuple[jnp.ndarray, ...]) -> Tuple[float, float]:
        """Update the SAC agent using a batch of experiences."""
        states, actions, rewards, next_states, dones = batch

        # Update critic
        next_actions = self.actor.apply(self.actor_params, next_states)
        target_q1 = self.target_critic1.apply(self.target_critic1_params, next_states, next_actions)
        target_q2 = self.target_critic2.apply(self.target_critic2_params, next_states, next_actions)
        target_q = jnp.minimum(target_q1, target_q2)
        target_q = rewards + (1 - dones) * self.gamma * target_q

        def critic_loss_fn(critic_params):
            q1 = self.critic1.apply(critic_params, states, actions)
            q2 = self.critic2.apply(critic_params, states, actions)
            critic_loss = jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
            return critic_loss

        critic_loss, critic1_grads = jax.value_and_grad(critic_loss_fn)(self.critic1_params)
        critic_loss, critic2_grads = jax.value_and_grad(critic_loss_fn)(self.critic2_params)
        self.critic1_params, self.critic1_opt_state = self.critic_optimizer.update(critic1_grads, self.critic1_opt_state, self.critic1_params)
        self.critic2_params, self.critic2_opt_state = self.critic_optimizer.update(critic2_grads, self.critic2_opt_state, self.critic2_params)

        # Update actor
        def actor_loss_fn(actor_params):
            actions = self.actor.apply(actor_params, states)
            q1 = self.critic1.apply(self.critic1_params, states, actions)
            q2 = self.critic2.apply(self.critic2_params, states, actions)
            q = jnp.minimum(q1, q2)
            actor_loss = -jnp.mean(q)
            return actor_loss

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(self.actor_params)
        self.actor_params, self.actor_opt_state = self.actor_optimizer.update(actor_grads, self.actor_opt_state, self.actor_params)

        # Update target networks
        self.target_critic1_params = self.soft_update(self.critic1_params, self.target_critic1_params)
        self.target_critic2_params = self.soft_update(self.critic2_params, self.target_critic2_params)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, source_params: dict, target_params: dict) -> dict:
        """Soft update of target network parameters."""
        return jax.tree_map(lambda s, t: self.tau * s + (1 - self.tau) * t, source_params, target_params)

class TD3Agent:
    """Twin Delayed DDPG (TD3) agent implementation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, policy_noise: float = 0.2,
                 noise_clip: float = 0.5, policy_freq: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # Initialize actor and critic networks
        self.actor = Actor(action_dim, hidden_dim)
        self.critic1 = Critic(hidden_dim)
        self.critic2 = Critic(hidden_dim)
        self.target_actor = Actor(action_dim, hidden_dim)
        self.target_critic1 = Critic(hidden_dim)
        self.target_critic2 = Critic(hidden_dim)

        # Initialize parameters
        self.actor_params = self.actor.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.critic1_params = self.critic1.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.critic2_params = self.critic2.init(jax.random.PRNGKey(2), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.target_actor_params = self.actor_params
        self.target_critic1_params = self.critic1_params
        self.target_critic2_params = self.critic2_params

        # Initialize optimizers
        self.actor_optimizer = optax.adam(learning_rate)
        self.critic_optimizer = optax.adam(learning_rate)

        # Initialize optimizer states
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic1_opt_state = self.critic_optimizer.init(self.critic1_params)
        self.critic2_opt_state = self.critic_optimizer.init(self.critic2_params)

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select an action using the current policy."""
        return self.actor.apply(self.actor_params, state)

    def update(self, batch: Tuple[jnp.ndarray, ...], step: int) -> Tuple[float, float]:
        """Update the TD3 agent using a batch of experiences."""
        states, actions, rewards, next_states, dones = batch

        # Add noise to target policy
        noise = jnp.clip(jax.random.normal(jax.random.PRNGKey(step), shape=actions.shape) * self.policy_noise, -self.noise_clip, self.noise_clip)
        next_actions = jnp.clip(self.actor.apply(self.target_actor_params, next_states) + noise, -1, 1)

        # Compute target Q-values
        target_q1 = self.critic1.apply(self.target_critic1_params, next_states, next_actions)
        target_q2 = self.critic2.apply(self.target_critic2_params, next_states, next_actions)
        target_q = jnp.minimum(target_q1, target_q2)
        target_q = rewards + (1 - dones) * self.gamma * target_q

        # Update critics
        def critic_loss_fn(critic1_params, critic2_params):
            q1 = self.critic1.apply(critic1_params, states, actions)
            q2 = self.critic2.apply(critic2_params, states, actions)
            critic_loss = jnp.mean((q1 - target_q)**2 + (q2 - target_q)**2)
            return critic_loss

        (critic_loss, (critic1_grads, critic2_grads)) = jax.value_and_grad(critic_loss_fn, argnums=(0, 1))(self.critic1_params, self.critic2_params)
        self.critic1_params, self.critic1_opt_state = self.critic_optimizer.update(critic1_grads, self.critic1_opt_state, self.critic1_params)
        self.critic2_params, self.critic2_opt_state = self.critic_optimizer.update(critic2_grads, self.critic2_opt_state, self.critic2_params)

        # Delayed policy updates
        if step % self.policy_freq == 0:
            # Update actor
            def actor_loss_fn(actor_params):
                actions = self.actor.apply(actor_params, states)
                actor_loss = -jnp.mean(self.critic1.apply(self.critic1_params, states, actions))
                return actor_loss

            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(self.actor_params)
            self.actor_params, self.actor_opt_state = self.actor_optimizer.update(actor_grads, self.actor_opt_state, self.actor_params)

            # Update target networks
            self.target_actor_params = self.soft_update(self.actor_params, self.target_actor_params)
            self.target_critic1_params = self.soft_update(self.critic1_params, self.target_critic1_params)
            self.target_critic2_params = self.soft_update(self.critic2_params, self.target_critic2_params)
        else:
            actor_loss = 0.0

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, source_params: dict, target_params: dict) -> dict:
        """Soft update of target network parameters."""
        return jax.tree_map(lambda s, t: self.tau * s + (1 - self.tau) * t, source_params, target_params)

class PPOAgent:
    """Proximal Policy Optimization (PPO) agent implementation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 clip_ratio: float = 0.2, max_grad_norm: float = 0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm

        # Initialize actor and critic networks
        self.actor = Actor(action_dim, hidden_dim)
        self.critic = Critic(hidden_dim)

        # Initialize parameters
        self.actor_params = self.actor.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.critic_params = self.critic.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))

        # Initialize optimizers
        self.actor_optimizer = optax.adam(learning_rate)
        self.critic_optimizer = optax.adam(learning_rate)

        # Initialize optimizer states
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

    @jax.jit
    def select_action(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select an action using the current policy."""
        mean, log_std = self.actor.apply(self.actor_params, state)
        std = jnp.exp(log_std)
        action = mean + std * jax.random.normal(jax.random.PRNGKey(0), shape=mean.shape)
        log_prob = self._log_prob(mean, std, action)
        return action, log_prob

    @jax.jit
    def _log_prob(self, mean, std, action):
        return -0.5 * (((action - mean) / std)**2).sum(axis=-1) - 0.5 * np.log(2.0 * np.pi) * action.shape[-1] - std.sum(axis=-1)

    @jax.jit
    def update(self, states: jnp.ndarray, actions: jnp.ndarray, old_log_probs: jnp.ndarray,
               advantages: jnp.ndarray, returns: jnp.ndarray) -> Tuple[float, float]:
        """Update the PPO agent using collected experiences."""
        def loss_fn(actor_params, critic_params):
            mean, log_std = self.actor.apply(actor_params, states)
            std = jnp.exp(log_std)
            new_log_probs = self._log_prob(mean, std, actions)
            ratio = jnp.exp(new_log_probs - old_log_probs)
            clip_adv = jnp.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clip_adv))

            value_pred = self.critic.apply(critic_params, states, actions)
            critic_loss = jnp.mean((returns - value_pred)**2)

            return actor_loss + 0.5 * critic_loss

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
        loss, (actor_grads, critic_grads) = grad_fn(self.actor_params, self.critic_params)

        # Clip gradients
        actor_grads = jax.tree_map(lambda g: jnp.clip(g, -self.max_grad_norm, self.max_grad_norm), actor_grads)
        critic_grads = jax.tree_map(lambda g: jnp.clip(g, -self.max_grad_norm, self.max_grad_norm), critic_grads)

        # Update parameters
        self.actor_params, self.actor_opt_state = self.actor_optimizer.update(actor_grads, self.actor_opt_state, self.actor_params)
        self.critic_params, self.critic_opt_state = self.critic_optimizer.update(critic_grads, self.critic_opt_state, self.critic_params)

        return loss.item()

class DQNAgent:
    """Deep Q-Network (DQN) agent implementation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64,
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, target_update_freq: int = 100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Initialize Q-networks
        self.q_network = QNetwork(action_dim, hidden_dim)
        self.target_q_network = QNetwork(action_dim, hidden_dim)

        # Initialize parameters
        self.q_params = self.q_network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.target_q_params = self.q_params

        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.q_params)

    @jax.jit
    def select_action(self, state: jnp.ndarray, eval_mode: bool = False) -> int:
        """Select an action using epsilon-greedy policy."""
        if eval_mode or jax.random.uniform(jax.random.PRNGKey(0)) > self.epsilon:
            q_values = self.q_network.apply(self.q_params, state)
            return jnp.argmax(q_values).item()
        else:
            return jax.random.randint(jax.random.PRNGKey(0), (), 0, self.action_dim).item()

    @jax.jit
    def update(self, batch: Tuple[jnp.ndarray, ...]) -> float:
        """Update the DQN agent using a batch of experiences."""
        states, actions, rewards, next_states, dones = batch

        def loss_fn(q_params):
            q_values = self.q_network.apply(q_params, states)
            q_values = q_values[jnp.arange(q_values.shape[0]), actions]

            next_q_values = self.target_q_network.apply(self.target_q_params, next_states)
            next_q_values = jnp.max(next_q_values, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            loss = jnp.mean((q_values - target_q_values)**2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(self.q_params)
        self.q_params, self.opt_state = self.optimizer.update(grads, self.opt_state, self.q_params)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_q_params = self.q_params

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

class Actor(nn.Module):
    """Actor network for SAC, TD3, and PPO."""
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20, 2)
        return mean, log_std

class Critic(nn.Module):
    """Critic network for SAC, TD3, and PPO."""
    hidden_dim: int

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class QNetwork(nn.Module):
    """Q-network for DQN."""
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

def create_sac_agent(state_dim: int, action_dim: int, **kwargs) -> SACAgent:
    """Create a Soft Actor-Critic (SAC) agent."""
    return SACAgent(state_dim, action_dim, **kwargs)

def create_td3_agent(state_dim: int, action_dim: int, **kwargs) -> TD3Agent:
    """Create a Twin Delayed DDPG (TD3) agent."""
    return TD3Agent(state_dim, action_dim, **kwargs)

def create_ppo_agent(state_dim: int, action_dim: int, **kwargs) -> PPOAgent:
    """Create a Proximal Policy Optimization (PPO) agent."""
    return PPOAgent(state_dim, action_dim, **kwargs)

def create_dqn_agent(state_dim: int, action_dim: int, **kwargs) -> DQNAgent:
    """Create a Deep Q-Network (DQN) agent."""
    return DQNAgent(state_dim, action_dim, **kwargs)
