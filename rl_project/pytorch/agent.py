import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        value = self.critic(state)
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        return action_mean, action_std, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, action_std, _ = self.actor_critic(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        return action.detach().numpy().flatten()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor([float(d) for d in dones]).unsqueeze(1)

        # Compute advantages
        action_mean, action_std, values = self.actor_critic(states)
        _, _, next_values = self.actor_critic(next_states)

        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_error = td_target - values
        advantages = td_error.detach()

        # Compute actor loss
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
        old_log_probs = log_probs.detach()

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Compute critic loss
        critic_loss = F.mse_loss(values, td_target.detach())

        # Compute total loss
        loss = actor_loss + 0.5 * critic_loss

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Note: This implementation assumes a continuous action space.
# For discrete action spaces, modifications would be needed in the action selection and loss computation.
