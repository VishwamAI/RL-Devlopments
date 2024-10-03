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
from typing import List, Tuple, Callable

class MAMLModel(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class MAML:
    def __init__(self, input_dim: int, output_dim: int, alpha: float = 0.01, beta: float = 0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        self.model = MAMLModel(output_dim)
        self.params = self.model.init(jax.random.PRNGKey(0), jnp.zeros((1, input_dim)))
        self.meta_optimizer = optax.adam(self.beta)
        self.meta_opt_state = self.meta_optimizer.init(self.params)

    @jax.jit
    def inner_update(self, params, x, y):
        def loss_fn(p):
            pred = self.model.apply(p, x)
            return jnp.mean(optax.l2_loss(pred, y))
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        inner_params = jax.tree_map(lambda p, g: p - self.alpha * g, params, grads)
        return inner_params, loss

    @jax.jit
    def outer_update(self, params, tasks):
        def task_loss(p, task):
            support_x, support_y, query_x, query_y = task
            adapted_params, _ = self.inner_update(p, support_x, support_y)
            pred = self.model.apply(adapted_params, query_x)
            return jnp.mean(optax.l2_loss(pred, query_y))
        
        mean_loss, grads = jax.value_and_grad(lambda p: jnp.mean(jax.vmap(lambda t: task_loss(p, t))(tasks)))(params)
        updates, self.meta_opt_state = self.meta_optimizer.update(grads, self.meta_opt_state)
        params = optax.apply_updates(params, updates)
        return params, mean_loss

    def meta_train(self, tasks: List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]], num_epochs: int):
        for epoch in range(num_epochs):
            self.params, loss = self.outer_update(self.params, tasks)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")

class Reptile:
    def __init__(self, input_dim: int, output_dim: int, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.model = MAMLModel(output_dim)
        self.params = self.model.init(jax.random.PRNGKey(0), jnp.zeros((1, input_dim)))

    @jax.jit
    def inner_update(self, params, x, y, num_steps: int):
        def loss_fn(p):
            pred = self.model.apply(p, x)
            return jnp.mean(optax.l2_loss(pred, y))
        
        def update_step(p, _):
            loss, grads = jax.value_and_grad(loss_fn)(p)
            p = jax.tree_map(lambda param, grad: param - self.inner_lr * grad, p, grads)
            return p, loss

        final_params, losses = jax.lax.scan(update_step, params, jnp.arange(num_steps))
        return final_params, losses[-1]

    @jax.jit
    def outer_update(self, params, tasks, num_inner_steps: int):
        def task_update(p, task):
            x, y = task
            updated_params, _ = self.inner_update(p, x, y, num_inner_steps)
            return updated_params

        task_params = jax.vmap(lambda t: task_update(params, t))(tasks)
        mean_params = jax.tree_map(lambda *args: jnp.mean(jnp.stack(args), axis=0), *task_params)
        new_params = jax.tree_map(lambda p, mp: p + self.meta_lr * (mp - p), params, mean_params)
        return new_params

    def meta_train(self, tasks: List[Tuple[jnp.ndarray, jnp.ndarray]], num_epochs: int, num_inner_steps: int):
        for epoch in range(num_epochs):
            self.params = self.outer_update(self.params, tasks, num_inner_steps)
            # Compute validation loss for monitoring
            val_losses = []
            for x, y in tasks:
                _, loss = self.inner_update(self.params, x, y, num_inner_steps)
                val_losses.append(loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Mean Validation Loss: {jnp.mean(jnp.array(val_losses))}")
