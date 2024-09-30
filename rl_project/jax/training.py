import jax
import jax.numpy as jnp
import gym
import time
from agent import PPOAgent
from buffer import PPOBuffer
import tensorboardX as tbx

def train_ppo(env_name, num_episodes, max_steps_per_episode):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)
    buffer = PPOBuffer(state_dim, action_dim)

    writer = tbx.SummaryWriter(f'logs/jax_ppo_{env_name}')

    total_steps = 0
    start_time = time.time()

    key = jax.random.PRNGKey(0)  # Initialize random key

    for episode in range(num_episodes):
        state, _ = env.reset()  # Unpack the state from the reset return value
        print(f"Debug: State structure: {state}")  # Debug print statement
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps_per_episode):
            key, subkey = jax.random.split(key)  # Split key for randomness
            state_jax = jnp.array(state)  # Convert state to JAX array
            action = agent.get_action(state_jax, subkey)  # Pass JAX array and subkey to get_action
            next_state, reward, done, _, _ = env.step(action)  # Unpack the step return value

            buffer.add(state, action, reward, next_state, done)
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if buffer.is_full():
                loss = agent.update(*buffer.get())
                buffer.clear()
                writer.add_scalar('Loss', loss, total_steps)

            if done:
                break

            state = next_state

        writer.add_scalar('Episode Reward', episode_reward, episode)
        writer.add_scalar('Episode Length', episode_steps, episode)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Steps: {episode_steps}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    writer.add_scalar('Training Time', training_time, 0)

    writer.close()
    env.close()

if __name__ == "__main__":
    train_ppo("Pendulum-v1", num_episodes=1000, max_steps_per_episode=200)
