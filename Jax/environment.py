import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Dict, Any

class Environment:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = jnp.zeros(state_dim)
        self.rng = jax.random.PRNGKey(0)

    def reset(self) -> jnp.ndarray:
        self.rng, subkey = jax.random.split(self.rng)
        self.state = jax.random.normal(subkey, shape=(self.state_dim,))
        return self.state

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, Dict[str, Any]]:
        self.rng, subkey = jax.random.split(self.rng)
        
        # Simple dynamics: state changes based on action and some randomness
        self.state = jnp.clip(self.state + action + jax.random.normal(subkey, shape=self.state.shape) * 0.1, -1, 1)
        
        # Reward is negative distance from origin
        reward = -jnp.linalg.norm(self.state)
        
        # Episode ends if state is close to origin
        done = jnp.linalg.norm(self.state) < 0.1
        
        return self.state, reward, done, {}

class Agent(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)

def create_train_state(rng, state_dim, action_dim, learning_rate):
    agent = Agent(action_dim=action_dim)
    params = agent.init(rng, jnp.zeros((1, state_dim)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        actions = state.apply_fn(params, batch['states'])
        return -jnp.mean(batch['rewards'] * actions)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss

def train_agent(env, state_dim, action_dim, num_episodes, max_steps, batch_size, learning_rate):
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    train_state = create_train_state(init_rng, state_dim, action_dim, learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        states, actions, rewards = [], [], []

        for _ in range(max_steps):
            action = train_state.apply_fn(train_state.params, state[None, :])[0]
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        if len(states) >= batch_size:
            batch = {
                'states': jnp.array(states[-batch_size:]),
                'actions': jnp.array(actions[-batch_size:]),
                'rewards': jnp.array(rewards[-batch_size:])
            }
            train_state, loss = train_step(train_state, batch)

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    return train_state

# Usage
state_dim = 4
action_dim = 2
env = Environment(state_dim, action_dim)
trained_state = train_agent(env, state_dim, action_dim, num_episodes=1000, max_steps=200, batch_size=32, learning_rate=1e-3)
