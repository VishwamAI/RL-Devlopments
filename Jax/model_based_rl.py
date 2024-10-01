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
from jax import grad, jit, vmap
import optax  # JAX equivalent for optimization algorithms
import gym
import numpy as np
import jax.nn as nn
import math

# Define the Environment Model in JAX
class EnvironmentModel:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.params = {
            'w1': jax.random.normal(jax.random.PRNGKey(0), (state_dim + action_dim, hidden_dim)),
            'b1': jnp.zeros(hidden_dim),
            'w2': jax.random.normal(jax.random.PRNGKey(1), (hidden_dim, hidden_dim)),
            'b2': jnp.zeros(hidden_dim),
            'w3': jax.random.normal(jax.random.PRNGKey(2), (hidden_dim, state_dim)),
            'b3': jnp.zeros(state_dim)
        }
        self.optimizer = optax.adam(learning_rate=1e-3)
        self.opt_state = self.optimizer.init(self.params)

    def predict(self, state, action):
        x = jnp.concatenate([state, action])
        x = jnp.tanh(jnp.dot(x, self.params['w1']) + self.params['b1'])
        x = jnp.tanh(jnp.dot(x, self.params['w2']) + self.params['b2'])
        next_state = jnp.dot(x, self.params['w3']) + self.params['b3']
        return next_state

    @jit
    def update(self, states, actions, next_states):
        def loss_fn(params):
            pred_next_states = jax.vmap(lambda s, a: self.predict(s, a))(states, actions)
            return jnp.mean(jnp.square(pred_next_states - next_states))

        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

# Define the MBPO Agent
class MBPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.actor_params = {
            'w1': jax.random.normal(jax.random.PRNGKey(3), (state_dim, hidden_dim)),
            'b1': jnp.zeros(hidden_dim),
            'w2': jax.random.normal(jax.random.PRNGKey(4), (hidden_dim, hidden_dim)),
            'b2': jnp.zeros(hidden_dim),
            'w3': jax.random.normal(jax.random.PRNGKey(5), (hidden_dim, action_dim)),
            'b3': jnp.zeros(action_dim)
        }
        
        self.critic_params = {
            'w1': jax.random.normal(jax.random.PRNGKey(6), (state_dim + action_dim, hidden_dim)),
            'b1': jnp.zeros(hidden_dim),
            'w2': jax.random.normal(jax.random.PRNGKey(7), (hidden_dim, hidden_dim)),
            'b2': jnp.zeros(hidden_dim),
            'w3': jax.random.normal(jax.random.PRNGKey(8), (hidden_dim, 1)),
            'b3': jnp.zeros(1)
        }

        self.actor_optimizer = optax.adam(learning_rate=1e-3)
        self.critic_optimizer = optax.adam(learning_rate=1e-3)

        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

        self.env_model = EnvironmentModel(state_dim, action_dim)

    def policy(self, state):
        x = jnp.tanh(jnp.dot(state, self.actor_params['w1']) + self.actor_params['b1'])
        x = jnp.tanh(jnp.dot(x, self.actor_params['w2']) + self.actor_params['b2'])
        return jnp.tanh(jnp.dot(x, self.actor_params['w3']) + self.actor_params['b3'])

    def value(self, state, action):
        x = jnp.concatenate([state, action])
        x = jnp.tanh(jnp.dot(x, self.critic_params['w1']) + self.critic_params['b1'])
        x = jnp.tanh(jnp.dot(x, self.critic_params['w2']) + self.critic_params['b2'])
        return jnp.dot(x, self.critic_params['w3']) + self.critic_params['b3']

    @jit
    def update(self, states, actions, rewards, next_states, dones):
        # Update Critic
        def critic_loss_fn(critic_params):
            Q_vals = jax.vmap(lambda s, a: self.value(s, a))(states, actions)
            next_actions = jax.vmap(self.policy)(next_states)
            next_Q_vals = jax.vmap(lambda s, a: self.value(s, a))(next_states, next_actions)
            target_Q_vals = rewards + (1 - dones) * 0.99 * next_Q_vals
            return jnp.mean((Q_vals - target_Q_vals) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(self.critic_params)
        critic_updates, self.critic_opt_state = self.critic_optimizer.update(critic_grads, self.critic_opt_state)
        self.critic_params = optax.apply_updates(self.critic_params, critic_updates)

        # Update Actor
        def actor_loss_fn(actor_params):
            actions = jax.vmap(lambda s: self.policy(s))(states)
            return -jnp.mean(jax.vmap(lambda s, a: self.value(s, a))(states, actions))

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(self.actor_params)
        actor_updates, self.actor_opt_state = self.actor_optimizer.update(actor_grads, self.actor_opt_state)
        self.actor_params = optax.apply_updates(self.actor_params, actor_updates)

        return critic_loss, actor_loss

def train_agent(env, agent, num_episodes=1000, planning_horizon=5, num_simulations=10):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state).flatten()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.update(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))
            agent.env_model.update(np.array([state]), np.array([action]), np.array([next_state]))

            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")

    return agent
class MCTSAgent:
    def __init__(self, state_dim, action_dim, num_simulations=100, exploration_constant=1.41):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.env_model = EnvironmentModel(state_dim, action_dim)
        self.value_network = nn.Sequential(
            nn.Dense(64),
            nn.relu,
            nn.Dense(64),
            nn.relu,
            nn.Dense(1)
        )
        self.policy_network = nn.Sequential(
            nn.Dense(64),
            nn.relu,
            nn.Dense(64),
            nn.relu,
            nn.Dense(action_dim)
        )
        self.optimizer = optax.adam(learning_rate=1e-3)
        self.value_params = self.value_network.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.policy_params = self.policy_network.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)))
        self.value_opt_state = self.optimizer.init(self.value_params)
        self.policy_opt_state = self.optimizer.init(self.policy_params)

    @jax.jit
    def value(self, state):
        return self.value_network.apply(self.value_params, state)

    @jax.jit
    def policy(self, state):
        return self.policy_network.apply(self.policy_params, state)

    def select_action(self, state):
        root = MCTSNode(state)
        for _ in range(self.num_simulations):
            node = root
            while not node.is_leaf():
                node = node.select_child(self.exploration_constant)
            value = self.simulate(node.state)
            node.backpropagate(value)
        return root.best_child().action

    def simulate(self, state):
        action = self.policy(state)
        next_state = self.env_model.predict(state, action)
        return self.value(next_state)

    @jax.jit
    def update(self, states, actions, rewards, next_states, dones):
        def value_loss_fn(params):
            predicted_values = jax.vmap(lambda s: self.value_network.apply(params, s))(states)
            target_values = rewards + (1 - dones) * 0.99 * jax.vmap(lambda s: self.value_network.apply(params, s))(next_states)
            return jnp.mean((predicted_values - target_values) ** 2)

        def policy_loss_fn(params):
            log_probs = jax.vmap(lambda s: jax.nn.log_softmax(self.policy_network.apply(params, s)))(states)
            advantages = rewards + (1 - dones) * 0.99 * jax.vmap(lambda s: self.value(s))(next_states) - jax.vmap(lambda s: self.value(s))(states)
            return -jnp.mean(jnp.sum(log_probs * advantages, axis=-1))

        value_loss, value_grads = jax.value_and_grad(value_loss_fn)(self.value_params)
        policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(self.policy_params)

        value_updates, self.value_opt_state = self.optimizer.update(value_grads, self.value_opt_state)
        policy_updates, self.policy_opt_state = self.optimizer.update(policy_grads, self.policy_opt_state)

        self.value_params = optax.apply_updates(self.value_params, value_updates)
        self.policy_params = optax.apply_updates(self.policy_params, policy_updates)

        return value_loss, policy_loss

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, exploration_constant):
        return max(self.children, key=lambda c: c.ucb_score(exploration_constant))

    def ucb_score(self, exploration_constant):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self, action, next_state):
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)

    def best_child(self):
        return max(self.children, key=lambda c: c.visits)

def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = MBPOAgent(state_dim, action_dim)
    trained_agent = train_agent(env, agent)

    # Test the trained agent
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = trained_agent.policy(state).flatten()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test episode reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
