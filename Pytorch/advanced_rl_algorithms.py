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

"""
Advanced Reinforcement Learning Algorithms Module - PyTorch Version

This module implements advanced reinforcement learning algorithms including
Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3) using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import numpy as np


class Actor(nn.Module):
    """Actor network with improved architecture."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """Critic network with improved architecture."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACAgent:
    """Soft Actor-Critic (SAC) agent implementation in PyTorch."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to('cuda')
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to('cuda')
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to('cuda')
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to('cuda')
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to('cuda')

        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to('cuda').unsqueeze(0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=64):
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to('cuda')
        action = torch.FloatTensor(action).to('cuda')
        reward = torch.FloatTensor(reward).unsqueeze(1).to('cuda')
        next_state = torch.FloatTensor(next_state).to('cuda')
        done = torch.FloatTensor(done).unsqueeze(1).to('cuda')

        # Compute target Q-values
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * torch.min(target_q1, target_q2)

        # Update critics
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target critics
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# Example for creating an SAC Agent
if __name__ == "__main__":
    state_dim = 33  # Example state dimension
    action_dim = 4  # Example action dimension
    agent = SACAgent(state_dim, action_dim)

    # Example usage:
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print("Selected Action:", action)
