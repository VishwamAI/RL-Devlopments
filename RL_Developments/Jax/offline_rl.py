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
from typing import Any, Tuple, List
import numpy as np

# Define the policy network
class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x  # Direct action output

# Define a simple value network for Q-function
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x, a):
        x = jnp.concatenate([x, a], axis=-1)  # Concatenate state and action
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze()

class OfflineRL:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize policy and Q networks
        self.policy = PolicyNetwork(action_dim)
        self.q_network = QNetwork()

        # Initialize parameters
        self.policy_params = self.policy.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.q_params = self.q_network.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))

        # Set up optimizers
        self.policy_optimizer = optax.adam(learning_rate)
        self.q_optimizer = optax.adam(learning_rate)
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)
        self.q_opt_state = self.q_optimizer.init(self.q_params)

    @jax.jit
    def behavior_cloning_loss(self, policy_params, states, actions):
        pred_actions = self.policy.apply(policy_params, states)
        return jnp.mean(jnp.square(pred_actions - actions))  # MSE Loss

    @jax.jit
    def q_loss(self, q_params, states, actions, targets):
        q_values = self.q_network.apply(q_params, states, actions)
        return jnp.mean((q_values - targets) ** 2)

    @jax.jit
    def update(self, policy_params, q_params, policy_opt_state, q_opt_state, 
               states, actions, rewards, next_states, dones):
        # Behavior Cloning update
        def policy_loss_fn(pp):
            return self.behavior_cloning_loss(pp, states, actions)
        
        # Q-value update
        def q_loss_fn(qp):
            next_actions = self.policy.apply(policy_params, next_states)
            target_q_values = rewards + 0.99 * self.q_network.apply(qp, next_states, next_actions) * (1 - dones)
            return self.q_loss(qp, states, actions, target_q_values)

        policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
        q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_params)

        # Update the policy network
        policy_updates, policy_opt_state = self.policy_optimizer.update(policy_grads, policy_opt_state)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        # Update the Q network
        q_updates, q_opt_state = self.q_optimizer.update(q_grads, q_opt_state)
        q_params = optax.apply_updates(q_params, q_updates)

        return policy_params, q_params, policy_opt_state, q_opt_state, policy_loss, q_loss

    def train(self, dataset: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]], epochs: int = 100):
        for epoch in range(epochs):
            # Sample data
            states, actions, rewards, next_states, dones = map(jnp.array, zip(*dataset))
            
            # Update step
            self.policy_params, self.q_params, self.policy_opt_state, self.q_opt_state, policy_loss, q_loss = self.update(
                self.policy_params, self.q_params, self.policy_opt_state, self.q_opt_state,
                states, actions, rewards, next_states, dones
            )
            
            print(f"Epoch {epoch+1}/{epochs}, Policy Loss: {policy_loss}, Q Loss: {q_loss}")

    def select_action(self, state: np.ndarray):
        state = jnp.array(state)
        action = self.policy.apply(self.policy_params, state)
        return np.array(action)
