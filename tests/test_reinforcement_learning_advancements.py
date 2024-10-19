import pytest
import gym
import time
from unittest.mock import patch, MagicMock
from NeuroFlex.reinforcement_learning.reinforcement_learning_advancements import (
    AdvancedRLAgent,
    MultiAgentEnvironment,
    create_ppo_agent,
    create_sac_agent,
    train_multi_agent_rl,
    advanced_rl_training
)

@pytest.fixture
def env_id():
    return "CartPole-v1"

@pytest.fixture
def num_agents():
    return 2

@pytest.fixture
def action_dim():
    return 2

@pytest.fixture
def observation_dim():
    return 4

@pytest.fixture
def features():
    return [64, 64]

def test_advanced_rl_agent_initialization(observation_dim, action_dim, features):
    agent = AdvancedRLAgent(observation_dim, action_dim, features)
    assert isinstance(agent, AdvancedRLAgent)
    assert agent.observation_dim == observation_dim
    assert agent.action_dim == action_dim
    assert agent.features == features
    assert not agent.is_trained
    assert isinstance(agent.q_network, dict)
    assert isinstance(agent.optimizer, dict)

def test_advanced_rl_agent_select_action(observation_dim, action_dim, features):
    agent = AdvancedRLAgent(observation_dim, action_dim, features)
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = agent.select_action(state)
    assert isinstance(action, int)
    assert 0 <= action < action_dim

def test_multi_agent_environment(num_agents, env_id, observation_dim):
    env = MultiAgentEnvironment(num_agents, env_id)
    assert len(env.envs) == num_agents
    observations = env.reset()
    assert len(observations) == num_agents
    for obs in observations:
        assert isinstance(obs, jnp.ndarray)
        assert obs.shape == (observation_dim,)

    actions = [0] * num_agents
    next_obs, rewards, dones, truncated, infos = env.step(actions)
    assert len(next_obs) == num_agents
    assert len(rewards) == num_agents
    assert len(dones) == num_agents
    assert len(truncated) == num_agents
    assert len(infos) == num_agents

def test_create_ppo_agent(env_id):
    env = gym.make(env_id)
    agent = create_ppo_agent(env)
    assert isinstance(agent, AdvancedRLAgent)
    assert agent.action_dim == env.action_space.n

def test_create_sac_agent(env_id):
    env = gym.make(env_id)
    agent = create_sac_agent(env)
    assert isinstance(agent, AdvancedRLAgent)
    assert agent.action_dim == env.action_space.n

@patch('NeuroFlex.reinforcement_learning.reinforcement_learning_advancements.AdvancedRLAgent')
def test_train_multi_agent_rl(mock_agent_class, num_agents, observation_dim, action_dim):
    mock_agent = MagicMock()
    mock_agent.device = jax.devices("cpu")[0]
    mock_agent.select_action.return_value = 0
    mock_agent.replay_buffer = MagicMock()
    mock_agent.replay_buffer.batch_size = 32
    mock_agent.replay_buffer.__len__.return_value = 100  # Ensure buffer appears to have enough samples
    mock_agent.replay_buffer.sample.return_value = {
        'observations': jax.random.normal(jax.random.PRNGKey(0), (32, observation_dim)),
        'actions': jax.random.randint(jax.random.PRNGKey(1), (32, 1), 0, action_dim),
        'rewards': jax.random.normal(jax.random.PRNGKey(2), (32,)),
        'next_observations': jax.random.normal(jax.random.PRNGKey(3), (32, observation_dim)),
        'dones': jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (32,))
    }
    mock_agent.to.return_value = mock_agent
    mock_agent.performance = 0.0
    mock_agent.performance_threshold = 0.8
    mock_agent.epsilon = 0.5
    mock_agent_class.return_value = mock_agent

    env = MultiAgentEnvironment(num_agents, "CartPole-v1")
    agents = [mock_agent_class() for _ in range(num_agents)]
    total_timesteps = 200

    trained_agents = train_multi_agent_rl(env, agents, total_timesteps)

    assert len(trained_agents) == num_agents
    for agent in trained_agents:
        assert agent.update.called
        assert agent.select_action.called
        assert agent.replay_buffer.__len__.call_count > 0
        assert agent.update.call_count > 0

@patch('NeuroFlex.reinforcement_learning.reinforcement_learning_advancements.train_multi_agent_rl')
def test_advanced_rl_training(mock_train, env_id, num_agents):
    mock_train.return_value = [MagicMock(is_trained=True) for _ in range(num_agents)]
    trained_agents = advanced_rl_training(env_id, num_agents, algorithm="PPO", total_timesteps=100)
    assert len(trained_agents) == num_agents
    for agent in trained_agents:
        assert agent.is_trained

def test_agent_self_healing(env_id, observation_dim, action_dim, features):
    agent = AdvancedRLAgent(observation_dim, action_dim, features)
    agent.is_trained = True
    agent.performance = 0.5
    agent.last_update = 0
    agent.performance_threshold = 0.8

    issues = agent.diagnose()
    assert "Model performance is below threshold" in issues
    assert "Model hasn't been updated in 24 hours" in issues

    with patch.object(agent, 'train') as mock_train:
        mock_train.return_value = {'final_reward': 0.9, 'episode_rewards': [0.5, 0.6, 0.7, 0.8, 0.9]}
        env = gym.make(env_id)
        agent.heal(env, num_episodes=10, max_steps=100)

    assert agent.performance > 0.8
    assert agent.last_update > 0
    mock_train.assert_called_once()

def test_agent_train(env_id, observation_dim, action_dim, features):
    agent = AdvancedRLAgent(observation_dim, action_dim, features)
    env = gym.make(env_id)

    with patch.object(agent, 'select_action', return_value=0), \
         patch.object(agent, 'update', return_value=0.1):
        result = agent.train(env, num_episodes=200, max_steps=100)

    assert 'final_reward' in result
    assert 'episode_rewards' in result
    assert len(result['episode_rewards']) <= 200  # May stop early due to performance threshold
    assert agent.is_trained
    assert agent.performance >= agent.performance_threshold
    assert agent.epsilon < agent.epsilon_start

    # Check if moving average calculation is working
    if len(result['episode_rewards']) >= 100:
        moving_avg = sum(result['episode_rewards'][-100:]) / 100
        assert pytest.approx(agent.performance, moving_avg, rel=1e-5)
    else:
        moving_avg = sum(result['episode_rewards']) / len(result['episode_rewards'])
        assert pytest.approx(agent.performance, moving_avg, rel=1e-5)

def test_agent_update(observation_dim, action_dim, features):
    agent = AdvancedRLAgent(observation_dim, action_dim, features)
    batch = {
        'observations': jax.random.normal(jax.random.PRNGKey(0), (32, observation_dim)),
        'actions': jax.random.randint(jax.random.PRNGKey(1), (32, 1), 0, action_dim),
        'rewards': jax.random.normal(jax.random.PRNGKey(2), (32,)),
        'next_observations': jax.random.normal(jax.random.PRNGKey(3), (32, observation_dim)),
        'dones': jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (32,))
    }
    loss = agent.update(batch)
    assert isinstance(loss, float)

if __name__ == '__main__':
    pytest.main()