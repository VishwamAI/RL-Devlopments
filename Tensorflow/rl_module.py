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
from tensorflow import keras
from typing import List, Dict, Any
import gym
import logging
import scipy.signal

# Constants for PPO
gamma = 0.99
lam = 0.95
value_loss_coef = 0.5
entropy_coef = 0.01
learning_rate = 3e-4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PPOBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, *act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def add(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, gamma * lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class ActorCritic(keras.Model):
    def __init__(self, observation_dim, action_dim, features):
        super(ActorCritic, self).__init__()
        self.actor_layers = [keras.layers.Dense(feat, activation='relu') for feat in features]
        self.actor_out = keras.layers.Dense(action_dim)

        self.critic_layers = [keras.layers.Dense(feat, activation='relu') for feat in features]
        self.critic_out = keras.layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.actor_layers:
            x = layer(x)
        action_logits = self.actor_out(x)

        x = inputs
        for layer in self.critic_layers:
            x = layer(x)
        value = self.critic_out(x)

        return action_logits, value

def select_action(model, observation):
    logits = model(np.array([observation]), training=False)[0]
    action = np.random.choice(logits.shape[1], p=tf.nn.softmax(logits).numpy()[0])
    log_prob = tf.nn.log_softmax(logits)[0, action]
    return action, log_prob.numpy()

@tf.function
def update_ppo(model, optimizer, batch, clip_ratio):
    with tf.GradientTape() as tape:
        pi, v = model(batch['obs'], training=True)
        log_prob = tf.nn.log_softmax(pi)
        log_prob_action = tf.gather(log_prob, batch['act'], axis=1, batch_dims=1)

        ratio = tf.exp(log_prob_action - batch['logp'])
        clip_adv = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch['adv']
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * batch['adv'], clip_adv))

        value_pred = tf.squeeze(v)
        value_loss = 0.5 * tf.reduce_mean((value_pred - batch['ret'])**2)

        entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(pi) * log_prob, axis=1))
        kl = tf.reduce_mean(batch['logp'] - log_prob_action)

        total_loss = (policy_loss + value_loss_coef * value_loss
                      - entropy_coef * entropy)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, policy_loss, value_loss, kl, entropy

def get_minibatches(data, batch_size):
    indices = np.random.permutation(len(data['obs']))
    for start in range(0, len(data['obs']), batch_size):
        end = start + batch_size
        yield {k: v[indices[start:end]] for k, v in data.items()}

def train_rl_agent(
    env: gym.Env,
    num_episodes: int = 10000,
    max_steps: int = 1000,
    clip_ratio: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 64,
    buffer_size: int = 2048,
):
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(observation_dim, action_dim, features=[256, 256, 256])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    buffer = PPOBuffer(buffer_size, (observation_dim,), (action_dim,))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action, log_prob = select_action(model, obs)
            next_obs, reward, done, truncated, _ = env.step(action)

            episode_reward += reward

            value = model(np.array([obs]))[1]
            buffer.add(obs, action, reward, value.numpy()[0][0], log_prob)

            if buffer.ptr == buffer.max_size:
                last_val = model(np.array([next_obs]))[1].numpy()[0][0]
                buffer.finish_path(last_val)
                data = buffer.get()

                for _ in range(n_epochs):
                    for mini_batch in get_minibatches(data, batch_size):
                        total_loss, policy_loss, value_loss, kl, entropy = update_ppo(
                            model, optimizer, mini_batch, clip_ratio)

                buffer = PPOBuffer(buffer_size, (observation_dim,), (action_dim,))

            obs = next_obs
            if done or truncated:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

    return model

# Example usage:
# env = gym.make('CartPole-v1')
# trained_model = train_rl_agent(env)
