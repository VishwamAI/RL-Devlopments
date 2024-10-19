import pytest
import jax
import jax.numpy as jnp
from NeuroFlex.reinforcement_learning.self_curing_rl import SelfCuringRLAgent
from NeuroFlex.reinforcement_learning.rl_module import RLEnvironment, PrioritizedReplayBuffer

@pytest.fixture
def agent():
    features = [4]  # CartPole-v1 has 4 observation dimensions
    action_dim = 2
    return SelfCuringRLAgent(features=features, action_dim=action_dim)

@pytest.fixture
def env():
    return RLEnvironment("CartPole-v1")

@pytest.fixture
def replay_buffer():
    return PrioritizedReplayBuffer(100000, (4,), (2,))

def test_agent_initialization(agent):
    assert isinstance(agent, SelfCuringRLAgent)
    assert agent.features == [4]
    assert agent.action_dim == 2
    assert not agent.is_trained

def test_select_action(agent):
    state = jax.random.uniform(jax.random.PRNGKey(0), (4,))
    action = agent.select_action(state)
    assert isinstance(action, int)
    assert 0 <= action < agent.action_dim

def test_update(agent):
    batch_size = 32
    states = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, 4))
    actions = jax.random.randint(jax.random.PRNGKey(1), (batch_size, 1), 0, agent.action_dim)
    rewards = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,))
    next_states = jax.random.uniform(jax.random.PRNGKey(3), (batch_size, 4))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size,))

    batch = {
        'observations': states,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_states,
        'dones': dones
    }

    loss = agent.update(batch)
    assert isinstance(loss, float)

def test_train(agent, env):
    num_episodes = 10
    max_steps = 100
    training_info = agent.train(env, num_episodes, max_steps)

    assert 'final_reward' in training_info
    assert 'episode_rewards' in training_info
    assert len(training_info['episode_rewards']) == num_episodes
    assert agent.is_trained

def test_diagnose(agent):
    issues = agent.diagnose()
    assert isinstance(issues, list)
    assert "Model is not trained" in issues

    agent.is_trained = True
    agent.performance = 0.7
    agent.last_update = 0
    issues = agent.diagnose()
    assert "Model performance is below threshold" in issues
    assert "Model hasn't been updated in 24 hours" in issues

def test_heal(agent, env):
    agent.is_trained = False
    agent.performance = 0.7
    agent.last_update = 0

    num_episodes = 5
    max_steps = 50
    agent.heal(env, num_episodes, max_steps)

    assert agent.is_trained
    assert agent.performance > 0.7
    assert agent.last_update > 0

if __name__ == '__main__':
    pytest.main()