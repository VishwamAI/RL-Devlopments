import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Callable

# Define a Quantum-Inspired Q-Network
class QuantumQNetwork(nn.Module):
    action_dim: int
    state_dim: int
    num_qubits: int = 4  # Number of qubits for the quantum encoding
    circuit_layers: int = 2  # Number of layers in the parameterized quantum circuit

    @nn.compact
    def __call__(self, x):
        # Quantum Encoding Layer
        quantum_encoding = nn.Dense(self.num_qubits)(x)
        quantum_encoding = nn.sigmoid(quantum_encoding)
        
        # Parameterized Quantum Circuit Simulation
        for _ in range(self.circuit_layers):
            quantum_encoding = nn.Dense(self.num_qubits)(quantum_encoding)
            quantum_encoding = nn.relu(quantum_encoding)

        # Classical Layers
        x = nn.Dense(128)(quantum_encoding)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        q_values = nn.Dense(self.action_dim)(x)
        return q_values

# QRL Agent
class QRLAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Initialize the Q-network
        self.q_network = QuantumQNetwork(action_dim, state_dim)
        self.q_params = self.q_network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        
        # Set up the optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.q_params)
        
    @jax.jit
    def select_action(self, q_params, state, epsilon: float = 0.1):
        q_values = self.q_network.apply(q_params, state)
        greedy_action = jnp.argmax(q_values)
        random_action = jax.random.randint(jax.random.PRNGKey(0), (), 0, self.action_dim)
        return jnp.where(jax.random.uniform(jax.random.PRNGKey(0)) < epsilon, random_action, greedy_action)
    
    @jax.jit
    def update(self, q_params, opt_state, states, actions, rewards, next_states, dones):
        def loss_fn(params):
            q_values = self.q_network.apply(params, states)
            q_action_values = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze()

            # Compute the target using the Bellman equation
            next_q_values = self.q_network.apply(params, next_states)
            next_q_max = jnp.max(next_q_values, axis=1)
            targets = rewards + self.gamma * next_q_max * (1 - dones)
            
            # Mean Squared Error loss
            loss = jnp.mean((targets - q_action_values) ** 2)
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(q_params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        q_params = optax.apply_updates(q_params, updates)
        
        return q_params, opt_state, loss
    
    def train(self, dataset: list, epochs: int = 100):
        for epoch in range(epochs):
            states, actions, rewards, next_states, dones = map(jnp.array, zip(*dataset))
            
            self.q_params, self.opt_state, loss = self.update(self.q_params, self.opt_state, states, actions, rewards, next_states, dones)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    
    def save_model(self, path: str):
        jax.save(path, {"q_params": self.q_params})
    
    def load_model(self, path: str):
        params = jax.load(path)
        self.q_params = params["q_params"]
