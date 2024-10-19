import pytest
import jax.numpy as jnp
from RL_Developments.Jax.environment import Environment, Agent, create_train_state, train_agent

@pytest.fixture
def environment():
    state_dim = 4
    action_dim = 2
    return Environment(state_dim, action_dim)

@pytest.fixture
def agent(environment):
    action_dim = environment.action_dim
    return Agent(action_dim=action_dim)

def test_environment_reset(environment):
    state = environment.reset()
    assert state.shape == (4,), f"Expected state shape (4,), got {state.shape}"
    assert jnp.all(state >= -1) and jnp.all(state <= 1), "State values should be between -1 and 1"

def test_environment_step(environment):
    state = environment.reset()
    action = jnp.array([0.1, -0.2])
    next_state, reward, done, _ = environment.step(action)
    assert next_state.shape == (4,), f"Expected next_state shape (4,), got {next_state.shape}"
    assert isinstance(reward, float), f"Expected reward to be float, got {type(reward)}"
    assert isinstance(done, bool), f"Expected done to be bool, got {type(done)}"

def test_agent_initialization(agent):
    assert isinstance(agent, Agent)
    assert agent.action_dim == 2

def test_agent_call(agent):
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = agent(state)
    assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"

def test_create_train_state():
    state_dim = 4
    action_dim = 2
    learning_rate = 1e-3
    rng = jax.random.PRNGKey(0)
    train_state = create_train_state(rng, state_dim, action_dim, learning_rate)
    assert train_state is not None

def test_train_agent(environment):
    state_dim = 4
    action_dim = 2
    num_episodes = 10
    max_steps = 100
    batch_size = 32
    learning_rate = 1e-3
    trained_state = train_agent(environment, state_dim, action_dim, num_episodes, max_steps, batch_size, learning_rate)
    assert trained_state is not None

if __name__ == "__main__":
    pytest.main()
