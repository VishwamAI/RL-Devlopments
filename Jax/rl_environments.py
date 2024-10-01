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
