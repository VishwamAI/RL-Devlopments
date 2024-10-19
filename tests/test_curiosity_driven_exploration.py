import gymnasium as gym
import numpy as np
import torch
import pytest
from NeuroFlex.reinforcement_learning.curiosity_driven_exploration import CuriosityDrivenAgent, train_curiosity_driven_agent

@pytest.fixture
def env():
    return gym.make('MountainCar-v0')

@pytest.fixture
def agent(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return CuriosityDrivenAgent(state_dim, action_dim)

def test_curiosity_driven_exploration(env, agent):
    # Train the agent
    num_episodes = 100
    trained_agent = train_curiosity_driven_agent(env, agent, num_episodes)

    # Test the trained agent
    test_episodes = 10
    total_rewards = []

    for _ in range(test_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the state from the tuple if necessary
        state = np.array(state).flatten()  # Ensure state is a flat numpy array
        episode_reward = 0
        done = False

        while not done:
            action = trained_agent.act(state)
            next_state, reward, done, _ = env.step(np.argmax(action))
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the state from the tuple if necessary
            episode_reward += reward
            state = np.array(next_state).flatten()  # Ensure next_state is a flat numpy array

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    assert avg_reward > -200, f"Average test episode reward is too low: {avg_reward:.2f}"

    # Test ICM
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Extract the state from the tuple if necessary
    state = np.array(state).flatten()  # Ensure state is a flat numpy array
    next_state, _, _, _ = env.step(env.action_space.sample())
    if isinstance(next_state, tuple):
        next_state = next_state[0]  # Extract the state from the tuple if necessary
    next_state = np.array(next_state).flatten()  # Ensure next_state is a flat numpy array
    action = trained_agent.act(state)

    intrinsic_reward = trained_agent.icm.compute_intrinsic_reward(
        torch.FloatTensor(state).unsqueeze(0),
        torch.FloatTensor(next_state).unsqueeze(0),
        torch.FloatTensor(action).unsqueeze(0)
    )
    assert intrinsic_reward.item() > 0, f"Intrinsic reward is too low: {intrinsic_reward.item():.4f}"

    # Test novelty detection
    novelty = trained_agent.novelty_detector.compute_novelty(next_state)
    is_novel = trained_agent.novelty_detector.is_novel(next_state)
    assert novelty > 0, f"Novelty is too low: {novelty:.4f}"
    assert is_novel, "State is not novel"

    # Verify that the agent improves over time
    initial_rewards = total_rewards[:5]
    final_rewards = total_rewards[-5:]
    assert np.mean(final_rewards) > np.mean(initial_rewards), "Agent does not show significant improvement"

if __name__ == "__main__":
    pytest.main()
