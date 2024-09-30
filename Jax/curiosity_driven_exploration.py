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
import numpy as np
import gymnasium as gym
from flax.training import train_state

class ICM(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    beta: float = 0.2

    def setup(self):
        self.feature_encoder = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim)
        ])
        self.inverse_model = nn.Sequential([
            nn.Dense(self.hidden_dim * 2),
            nn.relu,
            nn.Dense(self.action_dim)
        ])
        self.forward_model = nn.Sequential([
            nn.Dense(self.hidden_dim + self.action_dim),
            nn.relu,
            nn.Dense(self.hidden_dim)
        ])

    def __call__(self, state, next_state, action):
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)

        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        action_one_hot = action_one_hot.reshape((state_feat.shape[0], -1))

        pred_next_state_feat = self.forward_model(jnp.concatenate([state_feat, action_one_hot], axis=1))
        pred_action = self.inverse_model(jnp.concatenate([state_feat, next_state_feat], axis=1))

        return pred_action, pred_next_state_feat, next_state_feat

    def compute_intrinsic_reward(self, state, next_state, action):
        _, pred_next_state_feat, next_state_feat = self(state, next_state, action)
        intrinsic_reward = self.beta * 0.5 * jnp.mean(jnp.square(pred_next_state_feat - next_state_feat), axis=1)
        return intrinsic_reward

class NoveltyDetector:
    def __init__(self, state_dim, memory_size=1000, novelty_threshold=0.1):
        self.memory = np.zeros((memory_size, state_dim))
        self.memory_index = 0
        self.memory_size = memory_size
        self.novelty_threshold = novelty_threshold

    def compute_novelty(self, state):
        if self.memory_index < self.memory_size:
            distances = np.mean(np.abs(self.memory[:self.memory_index] - state), axis=1)
        else:
            distances = np.mean(np.abs(self.memory - state), axis=1)

        novelty = np.min(distances)
        return novelty

    def update_memory(self, state):
        self.memory[self.memory_index % self.memory_size] = state
        self.memory_index += 1

    def is_novel(self, state):
        novelty = self.compute_novelty(state)
        return novelty > self.novelty_threshold

class CuriosityDrivenAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the actor and critic using Flax modules
        self.actor = nn.Sequential([
            nn.Dense(hidden_dim),
            nn.relu,
            nn.Dense(hidden_dim),
            nn.relu,
            nn.Dense(action_dim),
            nn.tanh
        ])
        
        self.critic = nn.Sequential([
            nn.Dense(hidden_dim),
            nn.relu,
            nn.Dense(hidden_dim),
            nn.relu,
            nn.Dense(1)
        ])
        
        # Initialize states for actor, critic, and ICM
        rng = jax.random.PRNGKey(42)
        self.actor_state = train_state.TrainState.create(apply_fn=self.actor.apply, 
                                                        params=self.actor.init(rng, jnp.ones((1, state_dim))),
                                                        tx=optax.adam(learning_rate))
        self.critic_state = train_state.TrainState.create(apply_fn=self.critic.apply, 
                                                         params=self.critic.init(rng, jnp.ones((1, state_dim))),
                                                         tx=optax.adam(learning_rate))
        self.icm = ICM(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        self.novelty_detector = NoveltyDetector(state_dim)

    def act(self, state):
        state = jnp.array(state).reshape(1, -1)
        action_probs = self.actor.apply(self.actor_state.params, state)
        action = jnp.argmax(action_probs, axis=1)
        action_one_hot = jax.nn.one_hot(action, self.action_dim).astype(float)
        return int(action), action_one_hot.squeeze()

    def update(self, state, action, reward, next_state, done):
        state = jnp.array(state)
        action = jnp.array(action)
        reward = jnp.array([reward])
        next_state = jnp.array(next_state)
        
        # ICM update logic
        pred_action, pred_next_state_feat, next_state_feat = self.icm(state, next_state, action)
        
        inverse_loss = jnp.mean((pred_action - action) ** 2)
        forward_loss = 0.5 * jnp.mean(jnp.square(pred_next_state_feat - next_state_feat))
        
        total_loss = inverse_loss + forward_loss
        
        self.novelty_detector.update_memory(next_state)

        # Critic update using TD-error
        value = self.critic.apply(self.critic_state.params, state)
        next_value = self.critic.apply(self.critic_state.params, next_state)
        td_error = reward + (1 - done) * 0.99 * next_value - value
        
        critic_loss = jnp.square(td_error).mean()

        return total_loss, critic_loss

def train_curiosity_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, action_one_hot = agent.act(state)
            next_state, extrinsic_reward, done, _ = env.step(action)
            total_reward += extrinsic_reward

            total_loss, critic_loss = agent.update(state, action_one_hot, extrinsic_reward, next_state, done)
            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Loss: {total_loss}, Critic Loss: {critic_loss}")

def main():
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = CuriosityDrivenAgent(state_dim=state_dim, action_dim=action_dim)
    train_curiosity_agent(env, agent)

if __name__ == "__main__":
    main()
