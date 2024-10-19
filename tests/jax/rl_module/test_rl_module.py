import pytest
import jax
import jax.numpy as jnp
from RL_Developments.Jax.rl_module import (
    PPOBuffer,
    discount_cumsum,
    Actor,
    Critic,
    RLAgent,
    select_action,
    update_ppo,
    get_minibatches,
    train_rl_agent
)
import gym

@pytest.fixture
def ppo_buffer():
    size = 100
    obs_dim = (4,)
    act_dim = (2,)
    return PPOBuffer(size, obs_dim, act_dim)

def test_ppo_buffer_add(ppo_buffer):
    obs = jnp.array([0.1, -0.2, 0.3, -0.4])
    act = jnp.array([0.5, -0.6])
    rew = 1.0
    val = 0.9
    logp = -0.1
    ppo_buffer.add(obs, act, rew, val, logp)
    assert ppo_buffer.ptr == 1
    assert jnp.array_equal(ppo_buffer.obs_buf[0], obs)
    assert jnp.array_equal(ppo_buffer.act_buf[0], act)
    assert ppo_buffer.rew_buf[0] == rew
    assert ppo_buffer.val_buf[0] == val
    assert ppo_buffer.logp_buf[0] == logp

def test_ppo_buffer_finish_path(ppo_buffer):
    obs = jnp.array([0.1, -0.2, 0.3, -0.4])
    act = jnp.array([0.5, -0.6])
    rew = 1.0
    val = 0.9
    logp = -0.1
    ppo_buffer.add(obs, act, rew, val, logp)
    ppo_buffer.finish_path(last_val=0.5)
    assert ppo_buffer.ptr == 1
    assert ppo_buffer.path_start_idx == 1
    assert len(ppo_buffer.adv_buf) == 100

def test_discount_cumsum():
    x = jnp.array([1, 2, 3])
    discount = 0.9
    result = discount_cumsum(x, discount)
    expected = jnp.array([1 + 2 * 0.9 + 3 * 0.9**2, 2 + 3 * 0.9, 3])
    assert jnp.allclose(result, expected)

def test_actor():
    actor = Actor(action_dim=2, features=[64, 64])
    x = jnp.array([[0.1, -0.2, 0.3, -0.4]])
    params = actor.init(jax.random.PRNGKey(0), x)
    action_logits = actor.apply(params, x)
    assert action_logits.shape == (1, 2)

def test_critic():
    critic = Critic(features=[64, 64])
    x = jnp.array([[0.1, -0.2, 0.3, -0.4]])
    params = critic.init(jax.random.PRNGKey(0), x)
    value = critic.apply(params, x)
    assert value.shape == (1, 1)

def test_rl_agent():
    agent = RLAgent(observation_dim=4, action_dim=2)
    x = jnp.array([[0.1, -0.2, 0.3, -0.4]])
    params = agent.init(jax.random.PRNGKey(0), x)
    action_logits, value = agent.apply(params, x)
    assert action_logits.shape == (1, 2)
    assert value.shape == (1, 1)

def test_select_action():
    agent = RLAgent(observation_dim=4, action_dim=2)
    x = jnp.array([[0.1, -0.2, 0.3, -0.4]])
    params = agent.init(jax.random.PRNGKey(0), x)
    key = jax.random.PRNGKey(0)
    action, log_prob = select_action(params, x, key)
    assert isinstance(action, jnp.ndarray)
    assert isinstance(log_prob, jnp.ndarray)

def test_update_ppo():
    agent = RLAgent(observation_dim=4, action_dim=2)
    x = jnp.array([[0.1, -0.2, 0.3, -0.4]])
    params = agent.init(jax.random.PRNGKey(0), x)
    optimizer_state = optax.adam(learning_rate=3e-4).init(params)
    batch = {
        'obs': x,
        'act': jnp.array([[0, 1]]),
        'ret': jnp.array([1.0]),
        'adv': jnp.array([0.5]),
        'logp': jnp.array([-0.1])
    }
    clip_ratio = 0.2
    params, optimizer_state, loss, aux = update_ppo(params, batch, optimizer_state, clip_ratio)
    assert isinstance(loss, jnp.ndarray)
    assert isinstance(aux, tuple)

def test_get_minibatches():
    data = {
        'obs': jnp.array([[0.1, -0.2, 0.3, -0.4]] * 10),
        'act': jnp.array([[0, 1]] * 10),
        'ret': jnp.array([1.0] * 10),
        'adv': jnp.array([0.5] * 10),
        'logp': jnp.array([-0.1] * 10)
    }
    batch_size = 2
    minibatches = list(get_minibatches(data, batch_size))
    assert len(minibatches) == 5
    for minibatch in minibatches:
        assert len(minibatch['obs']) == batch_size

def test_train_rl_agent():
    env = gym.make('CartPole-v1')
    agent = RLAgent(observation_dim=4, action_dim=2)
    trained_agent, params = train_rl_agent(agent, env, num_episodes=1, max_steps=10)
    assert isinstance(trained_agent, RLAgent)
    assert isinstance(params, dict)

if __name__ == "__main__":
    pytest.main()
