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
from typing import Tuple, Callable

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = DQNetwork(action_dim)
        self.params = self.network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def get_action(self, state: jnp.ndarray) -> int:
        q_values = self.network.apply(self.params, state)
        return jnp.argmax(q_values).item()

    def update(self, state: jnp.ndarray, action: int, reward: float, next_state: jnp.ndarray, done: bool):
        loss, grads = jax.value_and_grad(self._loss_fn)(self.params, state, action, reward, next_state, done)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def _loss_fn(self, params, state, action, reward, next_state, done):
        q_values = self.network.apply(params, state)
        next_q_values = self.network.apply(params, next_state)
        target = reward + (1 - done) * 0.99 * jnp.max(next_q_values)
        predicted = q_values[action]
        return jnp.square(target - predicted)

class DQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = ActorCritic(action_dim)
        self.params = self.network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def get_action(self, state: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
        action_probs, _ = self.network.apply(self.params, state)
        action = jax.random.categorical(jax.random.PRNGKey(0), action_probs)
        return action.item(), action_probs

    def update(self, states: jnp.ndarray, actions: jnp.ndarray, rewards: jnp.ndarray, 
               dones: jnp.ndarray, old_log_probs: jnp.ndarray):
        loss, grads = jax.value_and_grad(self._loss_fn)(self.params, states, actions, rewards, dones, old_log_probs)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def _loss_fn(self, params, states, actions, rewards, dones, old_log_probs):
        action_probs, values = self.network.apply(params, states)
        new_log_probs = jnp.log(action_probs[jnp.arange(actions.shape[0]), actions])
        
        advantages = rewards - values
        ratio = jnp.exp(new_log_probs - old_log_probs)
        clip_adv = jnp.clip(ratio, 0.8, 1.2) * advantages
        loss = -jnp.minimum(ratio * advantages, clip_adv)
        
        value_loss = jnp.square(rewards - values)
        entropy = -jnp.sum(action_probs * jnp.log(action_probs), axis=-1)
        
        return jnp.mean(loss + 0.5 * value_loss - 0.01 * entropy)

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        actor_hidden = nn.Dense(64)(x)
        actor_hidden = nn.relu(actor_hidden)
        actor_output = nn.Dense(self.action_dim)(actor_hidden)
        action_probs = nn.softmax(actor_output)

        critic_hidden = nn.Dense(64)(x)
        critic_hidden = nn.relu(critic_hidden)
        value = nn.Dense(1)(critic_hidden)

        return action_probs, value.squeeze()
