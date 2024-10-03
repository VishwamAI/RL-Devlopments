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

# Define the Cognition Network
class CognitionNetwork(nn.Module):
    input_dim: int
    memory_dim: int = 128  # Memory dimension for internal processing
    output_dim: int = 10   # Output dimension for decision-making

    @nn.compact
    def __call__(self, x, memory_state):
        # Attention mechanism to focus on relevant features
        attention_weights = nn.Dense(self.memory_dim)(x)
        attention_weights = nn.softmax(attention_weights)
        
        # Update the memory state using attention-weighted input
        updated_memory = memory_state + attention_weights * nn.Dense(self.memory_dim)(x)
        
        # Process information in memory through a multi-layer perceptron (MLP)
        x = nn.Dense(128)(updated_memory)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        # Decision-making output
        output = nn.Dense(self.output_dim)(x)
        return output, updated_memory

# Cognition Agent using the Cognition Network
class CognitionAgent:
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize the cognition network
        self.cognition_network = CognitionNetwork(input_dim, output_dim=output_dim)
        self.memory_state = jnp.zeros((1, 128))  # Initialize memory state
        
        self.params = self.cognition_network.init(jax.random.PRNGKey(0), jnp.zeros((1, input_dim)), self.memory_state)
        
        # Set up the optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
    @jax.jit
    def forward(self, params, x, memory_state):
        return self.cognition_network.apply(params, x, memory_state)
    
    @jax.jit
    def update(self, params, opt_state, x, y):
        def loss_fn(params):
            predicted, _ = self.cognition_network.apply(params, x, self.memory_state)
            loss = jnp.mean((predicted - y) ** 2)
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    def train(self, dataset: list, epochs: int = 100):
        for epoch in range(epochs):
            inputs, targets = map(jnp.array, zip(*dataset))
            self.params, self.opt_state, loss = self.update(self.params, self.opt_state, inputs, targets)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    
    def predict(self, x):
        output, self.memory_state = self.forward(self.params, x, self.memory_state)
        return output
    
    def save_model(self, path: str):
        jax.save(path, {"params": self.params})
    
    def load_model(self, path: str):
        params = jax.load(path)
        self.params = params["params"]
