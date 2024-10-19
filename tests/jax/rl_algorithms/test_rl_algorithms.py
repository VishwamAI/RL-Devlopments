import pytest
import jax.numpy as jnp
from RL_Developments.Jax.rl_algorithms import PolicyGradient, QLearning, ActorCritic

@pytest.fixture
def mock_env():
    class MockEnvironment:
        def reset(self):
            return jnp.array([0.1, -0.2, 0.3, -0.4])

        def step(self, action):
            next_state = jnp.array([0.2, -0.1, 0.4, -0.3])
            reward = 1.0
            done = False
            return next_state, reward, done, {}
    return MockEnvironment()

@pytest.fixture
def q_learning(mock_env):
    return QLearning(state_dim=4, action_dim=2)

@pytest.fixture
def policy_gradient(mock_env):
    return PolicyGradient(state_dim=4, action_dim=2)

@pytest.fixture
def actor_critic(mock_env):
    return ActorCritic(state_dim=4, action_dim=2)

def test_q_learning(q_learning, mock_env):
    state = mock_env.reset()
    action = q_learning.get_action(q_learning.params, state, epsilon=0.1)
    assert action is not None
    assert isinstance(action, int)

def test_policy_gradient(policy_gradient, mock_env):
    state = mock_env.reset()
    action = policy_gradient.get_action(policy_gradient.params, state)
    assert action is not None
    assert isinstance(action, int)

def test_actor_critic(actor_critic, mock_env):
    state = mock_env.reset()
    action = actor_critic.get_action(actor_critic.actor_params, state)
    assert action is not None
    assert isinstance(action, int)

if __name__ == "__main__":
    pytest.main()
