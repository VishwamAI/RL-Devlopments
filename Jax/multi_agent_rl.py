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
from functools import partial
import numpy as np

class Actor(nn.Module):
    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.relu(nn.Dense(self.hidden_dim)(state))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        action = nn.tanh(nn.Dense(self.action_dim)(x))
        return action

class Critic(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        q_value = nn.Dense(1)(x)
        return q_value

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(hidden_dim, action_dim)
        self.critic = Critic(hidden_dim)

        # Initialize model parameters
        self.actor_params = self.actor.init(jax.random.PRNGKey(0), jnp.ones((1, state_dim)))
        self.critic_params = self.critic.init(jax.random.PRNGKey(1), jnp.ones((1, state_dim)), jnp.ones((1, action_dim)))

        # Set up optimizers
        self.actor_optimizer = optax.adam(learning_rate)
        self.critic_optimizer = optax.adam(learning_rate)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

    def act(self, state):
        return jax.jit(lambda params, x: self.actor.apply(params, x))(self.actor_params, state)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, states, actions, rewards, next_states, dones):
        def critic_loss_fn(critic_params, actor_params, states, actions, rewards, next_states, dones):
            q_values = self.critic.apply(critic_params, states, actions).squeeze()
            next_actions = self.actor.apply(actor_params, next_states)
            next_q_values = self.critic.apply(critic_params, next_states, next_actions).squeeze()
            target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
            return jnp.mean((q_values - target_q_values) ** 2)

        # Update critic
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(self.critic_params, self.actor_params, states, actions, rewards, next_states, dones)
        self.critic_params, self.critic_opt_state = optax.apply_updates(self.critic_params, self.critic_optimizer.update(critic_grads, self.critic_opt_state))

        def actor_loss_fn(actor_params, critic_params, states):
            actions = self.actor.apply(actor_params, states)
            return -jnp.mean(self.critic.apply(critic_params, states, actions))

        # Update actor
        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(self.actor_params, self.critic_params, states)
        self.actor_params, self.actor_opt_state = optax.apply_updates(self.actor_params, self.actor_optimizer.update(actor_grads, self.actor_opt_state))

        return critic_loss, actor_loss

class MultiAgentEnvironment:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        return [np.random.rand(self.state_dim) for _ in range(self.num_agents)]

    def step(self, actions):
        # Simplified environment dynamics
        next_states = [np.random.rand(self.state_dim) for _ in range(self.num_agents)]
        rewards = [np.random.rand() for _ in range(self.num_agents)]
        dones = [False for _ in range(self.num_agents)]
        return next_states, rewards, dones, {}

def train_maddpg(num_agents, state_dim, action_dim, num_episodes=1000):
    env = MultiAgentEnvironment(num_agents, state_dim, action_dim)
    agents = [MADDPGAgent(state_dim, action_dim) for _ in range(num_agents)]

    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = 0

        while True:
            actions = [agent.act(jnp.array(state)) for agent, state in zip(agents, states)]
            actions = [action.flatten() for action in actions]
            next_states, rewards, dones, _ = env.step(actions)
            episode_reward += sum(rewards)

            for i, agent in enumerate(agents):
                agent.update(jnp.array([states[i]]), jnp.array([actions[i]]), jnp.array([rewards[i]]), jnp.array([next_states[i]]), jnp.array([dones[i]]))

            states = next_states

            if any(dones):
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {episode_reward / num_agents:.2f}")

    return agents

def inter_agent_communication(agents, messages):
    """
    Simulates inter-agent communication by exchanging messages between agents.
    """
    for i, agent in enumerate(agents):
        received_messages = [msg for j, msg in enumerate(messages) if j != i]
        # Process received messages (placeholder)
        pass  # In a real scenario, add agent message handling logic here

def main():
    num_agents = 3
    state_dim = 5
    action_dim = 2

    trained_agents = train_maddpg(num_agents, state_dim, action_dim)

    # Simulate inter-agent communication
    messages = [f"Message from Agent {i}" for i in range(num_agents)]
    inter_agent_communication(trained_agents, messages)

if __name__ == "__main__":
    main()
