import torch
import gym
import time
from agent import PPOAgent
from buffer import PPOBuffer
from torch.utils.tensorboard import SummaryWriter
from environment import create_environment, process_action, process_state

def train_ppo(env_name, num_episodes, max_steps_per_episode):
    env = create_environment(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)
    buffer = PPOBuffer(state_dim, action_dim)

    writer = SummaryWriter(f'logs/pytorch_ppo_{env_name}')

    total_steps = 0
    start_time = time.time()

    for episode in range(num_episodes):
        state, _ = env.reset()  # Extract the state array from the tuple
        state = process_state(state)
        print(f"Debug: State structure: {state}")  # Debug print statement
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps_per_episode):
            action = agent.get_action(state)
            processed_action = process_action(action)
            next_state, reward, done, _, _ = env.step(processed_action)  # Updated to handle new gym API
            next_state = process_state(next_state)

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
