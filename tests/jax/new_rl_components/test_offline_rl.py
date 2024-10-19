import pytest
import jax.numpy as jnp
from RL_Developments.Jax.offline_rl import OfflineRL

@pytest.fixture
def offline_rl():
    state_dim = 4
    action_dim = 2
    return OfflineRL(state_dim, action_dim)

def test_offline_rl_initialization(offline_rl):
    assert isinstance(offline_rl, OfflineRL)
    assert offline_rl.state_dim == 4
    assert offline_rl.action_dim == 2

def test_offline_rl_select_action(offline_rl):
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = offline_rl.select_action(state)
    assert isinstance(action, jnp.ndarray)
    assert action.shape == (2,)

def test_offline_rl_behavior_cloning_loss(offline_rl):
    batch_size = 32
    state_dim = 4
    action_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    loss = offline_rl.behavior_cloning_loss(offline_rl.policy_params, states, actions)
    assert isinstance(loss, jnp.ndarray)

def test_offline_rl_q_loss(offline_rl):
    batch_size = 32
    state_dim = 4
    action_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    targets = jax.random.normal(jax.random.PRNGKey(2), (batch_size,))
    loss = offline_rl.q_loss(offline_rl.q_params, states, actions, targets)
    assert isinstance(loss, jnp.ndarray)

def test_offline_rl_update(offline_rl):
    batch_size = 32
    state_dim = 4
    action_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size,))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size,))
    policy_params, q_params, policy_opt_state, q_opt_state, policy_loss, q_loss = offline_rl.update(
        offline_rl.policy_params, offline_rl.q_params, offline_rl.policy_opt_state, offline_rl.q_opt_state,
        states, actions, rewards, next_states, dones
    )
    assert isinstance(policy_loss, jnp.ndarray)
    assert isinstance(q_loss, jnp.ndarray)

def test_offline_rl_train(offline_rl):
    state_dim = 4
    action_dim = 2
    num_episodes = 10
    max_steps = 100
    batch_size = 32
    learning_rate = 1e-3
    trained_state = offline_rl.train(state_dim, action_dim, num_episodes, max_steps, batch_size, learning_rate)
    assert trained_state is not None

if __name__ == "__main__":
    pytest.main()
