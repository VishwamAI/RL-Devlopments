import jax
import jax.numpy as jnp

class PPOBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=1000):
        self.state_buf = jnp.zeros((buffer_size, state_dim))
        self.action_buf = jnp.zeros((buffer_size, action_dim))
        self.reward_buf = jnp.zeros(buffer_size)
        self.next_state_buf = jnp.zeros((buffer_size, state_dim))
        self.done_buf = jnp.zeros(buffer_size, dtype=bool)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

    def add(self, state, action, reward, next_state, done):
        self.state_buf = self.state_buf.at[self.ptr].set(state)
        self.action_buf = self.action_buf.at[self.ptr].set(action)
        self.reward_buf = self.reward_buf.at[self.ptr].set(reward)
        self.next_state_buf = self.next_state_buf.at[self.ptr].set(next_state)
        self.done_buf = self.done_buf.at[self.ptr].set(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get(self):
        return (
            self.state_buf[:self.size],
            self.action_buf[:self.size],
            self.reward_buf[:self.size],
            self.next_state_buf[:self.size],
            self.done_buf[:self.size]
        )

    def clear(self):
        self.ptr, self.size = 0, 0

    def is_full(self):
        return self.size == self.max_size
