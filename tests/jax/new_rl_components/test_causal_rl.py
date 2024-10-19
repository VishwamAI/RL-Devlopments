import pytest
import jax.numpy as jnp
from RL_Developments.Jax.rl_withcasual_reasoning import RLWithCasualReasoning

@pytest.fixture
def rl_with_causal_reasoning():
    state_dim = 4
    action_dim = 2
    return RLWithCasualReasoning(state_dim, action_dim)

def test_rl_with_causal_reasoning_initialization(rl_with_causal_reasoning):
    assert isinstance(rl_with_causal_reasoning, RLWithCasualReasoning)
    assert rl_with_causal_reasoning.state_dim == 4
    assert rl_with_causal_reasoning.action_dim == 2

def test_rl_with_causal_reasoning_get_action(rl_with_causal_reasoning):
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = rl_with_causal_reasoning.get_action(rl_with_causal_reasoning.policy_params, state)
    assert isinstance(action, jnp.ndarray)
    assert action.shape == ()

def test_rl_with_causal_reasoning_update(rl_with_causal_reasoning):
    batch_size = 32
    state_dim = 4
    action_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.randint(jax.random.PRNGKey(1), (batch_size,), 0, action_dim)
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size,))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size,))

    policy_params, value_params, policy_opt_state, value_opt_state, policy_loss, value_loss = rl_with_causal_reasoning.update(
        rl_with_causal_reasoning.policy_params, rl_with_causal_reasoning.value_params, rl_with_causal_reasoning.policy_opt_state, rl_with_causal_reasoning.value_opt_state,
        states, actions, rewards, next_states, dones
    )
    assert isinstance(policy_loss, jnp.ndarray)
    assert isinstance(value_loss, jnp.ndarray)

def test_rl_with_causal_reasoning_train(rl_with_causal_reasoning):
    state_dim = 4
    action_dim = 2
    num_episodes = 10
    max_steps = 100
    batch_size = 32
    learning_rate = 1e-3
    trained_state = rl_with_causal_reasoning.train(state_dim, action_dim, num_episodes, max_steps, batch_size, learning_rate)
    assert trained_state is not None

if __name__ == "__main__":
    pytest.main()
