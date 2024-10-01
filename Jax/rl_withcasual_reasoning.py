import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, List
import numpy as np
# Define the Policy Network with Causal Reasoning
class CausalPolicyNetwork(nn.Module):
    action_dim: int
    state_dim: int
    causal_attention_dim: int = 64  # Dimension for the causal attention layer

    @nn.compact
    def __call__(self, x):
        # Causal attention mechanism
        query = nn.Dense(self.causal_attention_dim)(x)
        key = nn.Dense(self.causal_attention_dim)(x)
        value = nn.Dense(self.causal_attention_dim)(x)
        
        # Scaled dot-product attention
        attention_scores = jnp.dot(query, key.T) / jnp.sqrt(self.causal_attention_dim)
        attention_weights = nn.softmax(attention_scores, axis=-1)
        causal_context = jnp.dot(attention_weights, value)
        
        # Concatenate original input with causal context
        combined_input = jnp.concatenate([x, causal_context], axis=-1)
        
        # Policy network layers
        x = nn.Dense(256)(combined_input)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        
        return nn.softmax(x)

# Define the Value Network with Causal Reasoning
class CausalValueNetwork(nn.Module):
    state_dim: int
    causal_attention_dim: int = 64

    @nn.compact
    def __call__(self, x, action):
        # Apply causal attention
        query = nn.Dense(self.causal_attention_dim)(x)
        key = nn.Dense(self.causal_attention_dim)(x)
        value = nn.Dense(self.causal_attention_dim)(x)

        attention_scores = jnp.dot(query, key.T) / jnp.sqrt(self.causal_attention_dim)
        attention_weights = nn.softmax(attention_scores, axis=-1)
        causal_context = jnp.dot(attention_weights, value)

        combined_input = jnp.concatenate([x, action, causal_context], axis=-1)
        
        # Value network layers
        x = nn.Dense(256)(combined_input)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        
        return x.squeeze()

# RL Agent with Causal Reasoning
class RLWithCasualReasoning:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize networks
        self.policy_network = CausalPolicyNetwork(action_dim, state_dim)
        self.value_network = CausalValueNetwork(state_dim)
        
        # Initialize parameters
        self.policy_params = self.policy_network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.value_params = self.value_network.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))

        # Set up optimizers
        self.policy_optimizer = optax.adam(learning_rate)
        self.value_optimizer = optax.adam(learning_rate)
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)
        self.value_opt_state = self.value_optimizer.init(self.value_params)

    @jax.jit
    def get_action(self, policy_params, state):
        action_probs = self.policy_network.apply(policy_params, state)
        return jax.random.categorical(jax.random.PRNGKey(0), action_probs)

    @jax.jit
    def update(self, policy_params, value_params, policy_opt_state, value_opt_state, 
               states, actions, rewards, next_states, dones):
        def policy_loss_fn(pp):
            action_probs = self.policy_network.apply(pp, states)
            log_probs = jnp.log(action_probs[jnp.arange(actions.shape[0]), actions])
            values = self.value_network.apply(value_params, states, actions).squeeze()
            advantages = rewards + 0.99 * self.value_network.apply(value_params, next_states, actions).squeeze() * (1 - dones) - values
            return -jnp.mean(log_probs * advantages)

        def value_loss_fn(vp):
            values = self.value_network.apply(vp, states, actions).squeeze()
            targets = rewards + 0.99 * self.value_network.apply(vp, next_states, actions).squeeze() * (1 - dones)
            return jnp.mean((targets - values) ** 2)

        policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
        value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_params)

        # Update the policy network
        policy_updates, policy_opt_state = self.policy_optimizer.update(policy_grads, policy_opt_state)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        # Update the value network
        value_updates, value_opt_state = self.value_optimizer.update(value_grads, value_opt_state)
        value_params = optax.apply_updates(value_params, value_updates)

        return policy_params, value_params, policy_opt_state, value_opt_state, policy_loss, value_loss

    def train(self, dataset: List[Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray, bool]], epochs: int = 100):
        for epoch in range(epochs):
            # Sample data
            states, actions, rewards, next_states, dones = map(jnp.array, zip(*dataset))
            
            # Update step
            self.policy_params, self.value_params, self.policy_opt_state, self.value_opt_state, policy_loss, value_loss = self.update(
                self.policy_params, self.value_params, self.policy_opt_state, self.value_opt_state,
                states, actions, rewards, next_states, dones
            )
            
            print(f"Epoch {epoch + 1}/{epochs}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")

    def select_action(self, state: jnp.ndarray):
        state = jnp.array(state)
        action = self.policy_network.apply(self.policy_params, state)
        return np.array(action)
