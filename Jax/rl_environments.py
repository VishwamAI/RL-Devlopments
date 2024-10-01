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
import gym
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, List, Dict, Any

class GymEnvironment:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self) -> jnp.ndarray:
        return jnp.array(self.env.reset())

    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool, Dict[str, Any]]:
        next_state, reward, done, info = self.env.step(action)
        return jnp.array(next_state), float(reward), bool(done), info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

# Example usage:
# trained_agent = train_dqn("CartPole-v1", episodes=500, max_steps=200)
