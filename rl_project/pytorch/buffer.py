import torch

class PPOBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=1000):
        self.state_buf = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.action_buf = torch.zeros((buffer_size, action_dim), dtype=torch.float32)
        self.reward_buf = torch.zeros(buffer_size, dtype=torch.float32)
        self.next_state_buf = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.done_buf = torch.zeros(buffer_size, dtype=torch.bool)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

    def add(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        self.action_buf[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.reward_buf[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_state_buf[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.bool)
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
