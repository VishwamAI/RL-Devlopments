import jax
import jax.numpy as jnp
import pytest
from RL_Developments.Jax.advanced_rl_algorithms import SACAgent, TD3Agent

@pytest.fixture
def sac_agent():
    state_dim = 4
    action_dim = 2
    return SACAgent(state_dim, action_dim)

@pytest.fixture
def td3_agent():
    state_dim = 4
    action_dim = 2
    return TD3Agent(state_dim, action_dim)

def test_sac_agent_action_selection(sac_agent):
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = sac_agent.select_action(state)
    assert action.shape == (2,), f"Expected action shape {(2,)}, got {action.shape}"

def test_sac_agent_update(sac_agent):
    batch_size = 32
    state_dim = 4
    action_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 1))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size, 1))

    critic_loss, actor_loss = sac_agent.update((states, actions, rewards, next_states, dones))
    assert isinstance(critic_loss, float), f"Expected float for critic_loss, got {type(critic_loss)}"
    assert isinstance(actor_loss, float), f"Expected float for actor_loss, got {type(actor_loss)}"

def test_td3_agent_action_selection(td3_agent):
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = td3_agent.select_action(state)
    assert action.shape == (2,), f"Expected action shape {(2,)}, got {action.shape}"

def test_td3_agent_update(td3_agent):
    batch_size = 32
    state_dim = 4
    action_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 1))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size, 1))

    critic_loss, actor_loss = td3_agent.update((states, actions, rewards, next_states, dones), step=0)
    assert isinstance(critic_loss, float), f"Expected float for critic_loss, got {type(critic_loss)}"
    assert isinstance(actor_loss, float), f"Expected float for actor_loss, got {type(actor_loss)}"

if __name__ == "__main__":
    pytest.main()
