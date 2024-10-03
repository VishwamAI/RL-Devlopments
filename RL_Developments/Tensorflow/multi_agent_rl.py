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

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class Actor(Model):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.out = layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        action = self.out(x)
        return action

class Critic(Model):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.out = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.out(x)
        return q_value

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)

        # Define optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def act(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = self.actor(tf.expand_dims(state, axis=0))
        return action.numpy().flatten()

    @tf.function
    def update(self, states, actions, rewards, next_states, dones):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update Critic
        with tf.GradientTape() as tape:
            next_actions = self.actor(next_states)
            next_q_values = self.critic(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
            q_values = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return critic_loss.numpy(), actor_loss.numpy()

class MultiAgentEnvironment:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        return [np.random.rand(self.state_dim) for _ in range(self.num_agents)]

    def step(self, actions):
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
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones, _ = env.step(actions)
            episode_reward += sum(rewards)

            for i, agent in enumerate(agents):
                agent.update([states[i]], [actions[i]], [rewards[i]], [next_states[i]], [dones[i]])

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
        pass  # You can extend this with inter-agent logic

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
