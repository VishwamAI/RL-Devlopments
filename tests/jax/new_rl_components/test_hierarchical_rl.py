import pytest
import jax.numpy as jnp
from RL_Developments.Jax.hierarchical_rl import HierarchicalRL

@pytest.fixture
def hierarchical_rl():
    state_dim = 4
    action_dim = 2
    return HierarchicalRL(state_dim, action_dim)

def test_hierarchical_rl_initialization(hierarchical_rl):
    assert isinstance(hierarchical_rl, HierarchicalRL)
    assert hierarchical_rl.state_dim == 4
    assert hierarchical_rl.action_dim == 2

def test_hierarchical_rl_get_action(hierarchical_rl):
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = hierarchical_rl.get_action(hierarchical_rl.actor_params, state)
    assert isinstance(action, jnp.ndarray)
    assert action.shape == ()

def test_hierarchical_rl_update(hierarchical_rl):
    batch_size = 32
    state_dim = 4
    action_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.randint(jax.random.PRNGKey(1), (batch_size,), 0, action_dim)
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size,))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size,))

    actor_params, critic_params, actor_opt_state, critic_opt_state, actor_loss, critic_loss = hierarchical_rl.update(
        hierarchical_rl.actor_params, hierarchical_rl.critic_params, hierarchical_rl.actor_opt_state, hierarchical_rl.critic_opt_state,
        states, actions, rewards, next_states, dones
    )
    assert isinstance(actor_loss, jnp.ndarray)
    assert isinstance(critic_loss, jnp.ndarray)

def test_hierarchical_rl_train(hierarchical_rl):
    state_dim = 4
    action_dim = 2
    num_episodes = 10
    max_steps = 100
    batch_size = 32
    learning_rate = 1e-3
    trained_state = hierarchical_rl.train(state_dim, action_dim, num_episodes, max_steps, batch_size, learning_rate)
    assert trained_state is not None

if __name__ == "__main__":
    pytest.main()
