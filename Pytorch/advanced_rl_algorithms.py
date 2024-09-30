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
Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG) using PyTorch.
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


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent implementation in PyTorch."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to('cuda')
        self.critic = Critic(state_dim, action_dim, hidden_dim).to('cuda')

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to('cuda').unsqueeze(0)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update(self, states, actions, old_log_probs, rewards, dones, next_states):
        states = torch.FloatTensor(states).to('cuda')
        actions = torch.LongTensor(actions).to('cuda')
        old_log_probs = torch.FloatTensor(old_log_probs).to('cuda')
        rewards = torch.FloatTensor(rewards).to('cuda')
        dones = torch.FloatTensor(dones).to('cuda')
        next_states = torch.FloatTensor(next_states).to('cuda')

        # Compute advantages
        with torch.no_grad():
            values = self.critic(states, self.actor(states)).squeeze()
            next_values = self.critic(next_states, self.actor(next_states)).squeeze()
            advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # PPO update
        for _ in range(10):  # Number of epochs
            # Actor loss
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            value_pred = self.critic(states, self.actor(states)).squeeze()
            value_loss = nn.MSELoss()(value_pred, rewards + self.gamma * next_values * (1 - dones))

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss.item(), value_loss.item(), entropy.item()


class DDPGAgent:
    """Deep Deterministic Policy Gradient (DDPG) agent implementation in PyTorch."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-4, gamma=0.99, tau=0.001):
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to('cuda')
        self.critic = Critic(state_dim, action_dim, hidden_dim).to('cuda')
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).to('cuda')
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to('cuda')

        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

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

        # Compute target Q-value
        with torch.no_grad():
            target_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, target_action)
            target_q = reward + self.gamma * target_q * (1 - done)

        # Update critic
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()


# Example for creating PPO and DDPG Agents
if __name__ == "__main__":
    state_dim = 33  # Example state dimension
    action_dim = 4  # Example action dimension

    ppo_agent = PPOAgent(state_dim, action_dim)
    ddpg_agent = DDPGAgent(state_dim, action_dim)

    # Example usage:
    state = np.random.randn(state_dim)
    ppo_action, _ = ppo_agent.select_action(state)
    ddpg_action = ddpg_agent.select_action(state)
    print("PPO Selected Action:", ppo_action)
    print("DDPG Selected Action:", ddpg_action)
