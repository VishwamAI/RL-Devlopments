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

import tensorflow as tf
import numpy as np
import gym

class EnvironmentModel(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(EnvironmentModel, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim + action_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(state_dim)
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, state, action):
        inputs = tf.concat([state, action], axis=-1)
        return self.model(inputs)

    def train_step(self, states, actions, next_states):
        with tf.GradientTape() as tape:
            predictions = self(states, actions)
            loss = tf.reduce_mean(tf.square(predictions - next_states))
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()

class MBPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='tanh')
        ])
        
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim + action_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        self.env_model = EnvironmentModel(state_dim, action_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def act(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self.actor(state)
        return action.numpy().flatten()

    def update(self, states, actions, rewards, next_states, dones):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update Critic
        with tf.GradientTape() as tape:
            Q_values = tf.squeeze(self.critic(tf.concat([states, actions], axis=1)))
            next_actions = self.actor(next_states)
            next_Q_values = tf.squeeze(self.critic(tf.concat([next_states, next_actions], axis=1)))
            target_Q_values = rewards + (1 - dones) * 0.99 * next_Q_values
            critic_loss = tf.reduce_mean(tf.square(Q_values - tf.stop_gradient(target_Q_values)))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(self.critic(tf.concat([states, self.actor(states)], axis=1)))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Train Environment Model
        model_loss = self.env_model.train_step(states, actions, next_states)

        return critic_loss.numpy(), actor_loss.numpy(), model_loss

def model_based_planning(agent, initial_state, planning_horizon=5, num_simulations=10):
    best_action_sequence = None
    best_reward = float('-inf')

    for _ in range(num_simulations):
        state = tf.convert_to_tensor(np.expand_dims(initial_state, axis=0), dtype=tf.float32)
        action_sequence = []
        total_reward = 0

        for _ in range(planning_horizon):
            action = agent.act(state.numpy().flatten())
            action_sequence.append(action)

            next_state = agent.env_model(state, tf.convert_to_tensor(np.expand_dims(action, axis=0), dtype=tf.float32))
            reward = agent.critic(tf.concat([state, tf.convert_to_tensor(np.expand_dims(action, axis=0), dtype=tf.float32)], axis=1))

            total_reward += reward.numpy().flatten()[0]
            state = next_state

        if total_reward > best_reward:
            best_reward = total_reward
            best_action_sequence = action_sequence

    return best_action_sequence[0]

def train_mbpo(env, agent, num_episodes=1000, planning_horizon=5, num_simulations=10):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = model_based_planning(agent, state, planning_horizon, num_simulations)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            agent.update([state], [action], [reward], [next_state], [done])

            state = next_state

            if done:
                break

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    return agent

def main():
    # Example usage
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = MBPOAgent(state_dim, action_dim)
    trained_agent = train_mbpo(env, agent)

    # Test the trained agent
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = trained_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test episode reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
