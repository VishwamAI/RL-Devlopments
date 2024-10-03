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
import pickle  # We'll use this for saving/loading the model parameters

# Define the Policy Network
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

# Define the Value Network
class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Define the HierarchicalRL class
class HierarchicalRL:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Instantiate policy and value networks
        self.actor = PolicyNetwork(action_dim)
        self.critic = ValueNetwork()
        
        # Initialize model parameters
        self.actor_params = self.actor.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.critic_params = self.critic.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)))
        
        # Set up optimizers
        self.actor_optimizer = optax.adam(learning_rate)
        self.critic_optimizer = optax.adam(learning_rate)
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

    def train(self, states, actions, rewards, next_states, dones, epochs: int = 100):
        for epoch in range(epochs):
            actor_params, critic_params, actor_opt_state, critic_opt_state, actor_loss, critic_loss = self.update(
                self.actor_params, self.critic_params, self.actor_opt_state, self.critic_opt_state,
                states, actions, rewards, next_states, dones
            )
            self.actor_params, self.critic_params = actor_params, critic_params
            self.actor_opt_state, self.critic_opt_state = actor_opt_state, critic_opt_state
            print(f"Epoch {epoch+1}/{epochs}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

    def save_model(self, path: str):
        # Saving parameters using pickle
        with open(path, 'wb') as f:
            pickle.dump({"actor_params": self.actor_params, "critic_params": self.critic_params}, f)
        print(f"Model saved at {path}")

    def load_model(self, path: str):
        # Loading parameters using pickle
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.actor_params = params["actor_params"]
        self.critic_params = params["critic_params"]
        print(f"Model loaded from {path}")
