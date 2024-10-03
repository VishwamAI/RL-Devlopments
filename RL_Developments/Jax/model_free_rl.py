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
from typing import Tuple, List

class QNetwork(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

class QLearningAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.1):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_network = QNetwork(action_dim)
        self.params = self.q_network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def get_action(self, state: jnp.ndarray) -> int:
        if jax.random.uniform(jax.random.PRNGKey(0)) < self.epsilon:
            return jax.random.randint(jax.random.PRNGKey(0), (), 0, self.action_dim)
        else:
            q_values = self.q_network.apply(self.params, state)
            return jnp.argmax(q_values).item()

    def update(self, state: jnp.ndarray, action: int, reward: float, next_state: jnp.ndarray, done: bool):
        def loss_fn(params):
            q_values = self.q_network.apply(params, state)
            next_q_values = self.q_network.apply(params, next_state)
            target = reward + self.gamma * jnp.max(next_q_values) * (1 - done)
            return jnp.square(target - q_values[action])

        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

class SARSAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.1):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_network = QNetwork(action_dim)
        self.params = self.q_network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def get_action(self, state: jnp.ndarray) -> int:
        if jax.random.uniform(jax.random.PRNGKey(0)) < self.epsilon:
            return jax.random.randint(jax.random.PRNGKey(0), (), 0, self.action_dim)
        else:
            q_values = self.q_network.apply(self.params, state)
            return jnp.argmax(q_values).item()

    def update(self, state: jnp.ndarray, action: int, reward: float, next_state: jnp.ndarray, next_action: int, done: bool):
        def loss_fn(params):
            q_values = self.q_network.apply(params, state)
            next_q_values = self.q_network.apply(params, next_state)
            target = reward + self.gamma * next_q_values[next_action] * (1 - done)
            return jnp.square(target - q_values[action])

        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss
