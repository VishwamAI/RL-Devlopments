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
from typing import List, Tuple, Dict, Any, Sequence
import gym
import logging
import numpy as np
import time
import scipy.signal

# Constants for PPO
gamma = 0.99
lam = 0.95
value_loss_coef = 0.5
entropy_coef = 0.01

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PPOBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs_buf = jnp.zeros((size, *obs_dim))
        self.act_buf = jnp.zeros((size, *act_dim))
        self.adv_buf = jnp.zeros(size)
        self.rew_buf = jnp.zeros(size)
        self.ret_buf = jnp.zeros(size)
        self.val_buf = jnp.zeros(size)
        self.logp_buf = jnp.zeros(size)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def add(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf = self.obs_buf.at[self.ptr].set(obs)
        self.act_buf = self.act_buf.at[self.ptr].set(act)
        self.rew_buf = self.rew_buf.at[self.ptr].set(rew)
        self.val_buf = self.val_buf.at[self.ptr].set(val)
        self.logp_buf = self.logp_buf.at[self.ptr].set(logp)
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = jnp.append(self.rew_buf[path_slice], last_val)
        vals = jnp.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        self.adv_buf = self.adv_buf.at[path_slice].set(discount_cumsum(deltas, gamma * lam))
        self.ret_buf = self.ret_buf.at[path_slice].set(discount_cumsum(rews, gamma)[:-1])
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

@jax.jit
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(nn.Module):
    action_dim: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

class Critic(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(1)(x)

class RLAgent(nn.Module):
    observation_dim: int
    action_dim: int
    features: Sequence[int] = (256, 256, 256)

    def setup(self):
        self.actor = Actor(self.action_dim, self.features)
        self.critic = Critic(self.features)

    def __call__(self, x):
        return self.actor(x), self.critic(x)

    def actor_forward(self, x):
        return self.actor(x)

    def critic_forward(self, x):
        return self.critic(x)

@jax.jit
def select_action(params, observation, key):
    logits = RLAgent(params).actor_forward(observation)
    action = jax.random.categorical(key, logits)
    log_prob = jax.nn.log_softmax(logits)[action]
    return action, log_prob

@jax.jit
def update_ppo(params, batch, optimizer_state, clip_ratio):
    def loss_fn(params):
        pi, v = RLAgent(params)(batch['obs'])
        log_prob = jax.nn.log_softmax(pi).gather(1, batch['act'].unsqueeze(-1)).squeeze(-1)
        ratio = jnp.exp(log_prob - batch['logp'])
        clip_adv = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch['adv']
        policy_loss = -jnp.mean(jnp.minimum(ratio * batch['adv'], clip_adv))

        value_pred = v.squeeze(-1)
        value_loss = 0.5 * jnp.mean((value_pred - batch['ret'])**2)

        entropy = -jnp.mean(jnp.sum(jax.nn.softmax(pi) * jax.nn.log_softmax(pi), axis=-1))
        kl = jnp.mean(batch['logp'] - log_prob)

        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        return total_loss, (policy_loss, value_loss, kl, entropy)

    optimizer = optax.adam(learning_rate=3e-4)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(params)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, loss, aux

def get_minibatches(data, batch_size):
    indices = np.random.permutation(len(data['obs']))
    for start in range(0, len(data['obs']), batch_size):
        end = start + batch_size
        yield {k: v[indices[start:end]] for k, v in data.items()}

def train_rl_agent(
    agent: RLAgent,
    env: gym.Env,
    num_episodes: int = 10000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    clip_ratio: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 64,
    buffer_size: int = 2048,
    target_kl: float = 0.01,
):
    key = jax.random.PRNGKey(0)
    params = agent.init(key, jnp.zeros((1, env.observation_space.shape[0])))
    optimizer = optax.adam(learning_rate=3e-4)
    optimizer_state = optimizer.init(params)

    buffer = PPOBuffer(buffer_size, env.observation_space.shape, env.action_space.shape)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            key, subkey = jax.random.split(key)
            action, log_prob = select_action(params, jnp.array(obs), subkey)
            next_obs, reward, done, truncated, _ = env.step(action.item())

            episode_reward += reward

            value = agent.apply({'params': params}, jnp.array(obs))
            buffer.add(obs, action, reward, value, log_prob)

            if buffer.ptr == buffer.max_size:
                last_val = agent.apply({'params': params}, jnp.array(next_obs))
                buffer.finish_path(last_val)
                data = buffer.get()

                for _ in range(n_epochs):
                    for mini_batch in get_minibatches(data, batch_size):
                        params, optimizer_state, loss, (policy_loss, value_loss, kl, entropy) = update_ppo(
                            params, mini_batch, optimizer_state, clip_ratio)
                        if kl > 1.5 * target_kl:
                            break

                buffer = PPOBuffer(buffer_size, env.observation_space.shape, env.action_space.shape)

            obs = next_obs
            if done or truncated:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

    return agent, params
