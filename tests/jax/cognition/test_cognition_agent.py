import pytest
import jax.numpy as jnp
from RL_Developments.Jax.cognition import CognitionAgent, CognitionNetwork

@pytest.fixture
def cognition_agent():
    input_dim = 4
    output_dim = 2
    return CognitionAgent(input_dim, output_dim)

def test_cognition_agent_initialization(cognition_agent):
    assert isinstance(cognition_agent, CognitionAgent)
    assert cognition_agent.input_dim == 4
    assert cognition_agent.output_dim == 2

def test_cognition_agent_forward(cognition_agent):
    state = jnp.array([[0.1, -0.2, 0.3, -0.4]])
    memory_state = jnp.zeros((1, 128))
    output, updated_memory = cognition_agent.forward(cognition_agent.params, state, memory_state)
    assert output.shape == (1, 2)
    assert updated_memory.shape == (1, 128)

def test_cognition_agent_update(cognition_agent):
    batch_size = 32
    input_dim = 4
    output_dim = 2
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
    targets = jax.random.normal(jax.random.PRNGKey(1), (batch_size, output_dim))

    initial_params = cognition_agent.params
    cognition_agent.update(cognition_agent.params, cognition_agent.opt_state, states, targets)
    updated_params = cognition_agent.params

    assert not jnp.array_equal(initial_params, updated_params)

def test_cognition_agent_predict(cognition_agent):
    state = jnp.array([[0.1, -0.2, 0.3, -0.4]])
    output = cognition_agent.predict(state)
    assert output.shape == (1, 2)

def test_cognition_agent_save_load_model(cognition_agent, tmp_path):
    model_path = tmp_path / "cognition_agent_model"
    cognition_agent.save_model(model_path)
    cognition_agent.load_model(model_path)
    assert cognition_agent.params is not None

if __name__ == "__main__":
    pytest.main()
