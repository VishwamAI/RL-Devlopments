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
from typing import Tuple, List, Callable

class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.softmax(x)

class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)

class PolicyGradient:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.policy = PolicyNetwork(action_dim)
        self.params = self.policy.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    @jax.jit
    def get_action(self, params, state):
        action_probs = self.policy.apply(params, state)
        return jax.random.categorical(jax.random.PRNGKey(0), action_probs)

    @jax.jit
    def update(self, params, opt_state, states, actions, rewards):
        def loss_fn(p):
            action_probs = self.policy.apply(p, states)
            log_probs = jnp.log(action_probs[jnp.arange(actions.shape[0]), actions])
            return -jnp.mean(log_probs * rewards)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

class QLearning:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3, gamma: float = 0.99):
        self.q_network = nn.Dense(action_dim)
        self.params = self.q_network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self.gamma = gamma

    @jax.jit
    def get_action(self, params, state, epsilon):
        q_values = self.q_network.apply(params, state)
        return jax.lax.cond(
            jax.random.uniform(jax.random.PRNGKey(0)) < epsilon,
            lambda: jax.random.randint(jax.random.PRNGKey(0), (), 0, q_values.shape[-1]),
            lambda: jnp.argmax(q_values)
        )

    @jax.jit
    def update(self, params, opt_state, state, action, reward, next_state, done):
        def loss_fn(p):
            q_values = self.q_network.apply(p, state)
            next_q_values = self.q_network.apply(p, next_state)
            target = reward + self.gamma * jnp.max(next_q_values) * (1 - done)
            return jnp.square(target - q_values[action])

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

class ActorCritic:
    def __init__(self, state_dim: int, action_dim: int, actor_lr: float = 1e-3, critic_lr: float = 1e-3):
        self.actor = PolicyNetwork(action_dim)
        self.critic = ValueNetwork()
        self.actor_params = self.actor.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.critic_params = self.critic.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)))
        self.actor_optimizer = optax.adam(actor_lr)
        self.critic_optimizer = optax.adam(critic_lr)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

    @jax.jit
    def get_action(self, actor_params, state):
        action_probs = self.actor.apply(actor_params, state)
        return jax.random.categorical(jax.random.PRNGKey(0), action_probs)

    @jax.jit
    def update(self, actor_params, critic_params, actor_opt_state, critic_opt_state, 
               states, actions, rewards, next_states, dones):
        def actor_loss_fn(ap, cp):
            action_probs = self.actor.apply(ap, states)
            log_probs = jnp.log(action_probs[jnp.arange(actions.shape[0]), actions])
            values = self.critic.apply(cp, states).squeeze()
            advantages = rewards + 0.99 * self.critic.apply(cp, next_states).squeeze() * (1 - dones) - values
            return -jnp.mean(log_probs * advantages)

        def critic_loss_fn(cp):
            values = self.critic.apply(cp, states).squeeze()
            targets = rewards + 0.99 * self.critic.apply(cp, next_states).squeeze() * (1 - dones)
            return jnp.mean(jnp.square(targets - values))

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params, critic_params)
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)

        actor_updates, actor_opt_state = self.actor_optimizer.update(actor_grads, actor_opt_state)
        critic_updates, critic_opt_state = self.critic_optimizer.update(critic_grads, critic_opt_state)

        actor_params = optax.apply_updates(actor_params, actor_updates)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        return actor_params, critic_params, actor_opt_state, critic_opt_state, actor_loss, critic_loss
