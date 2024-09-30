import gym

def create_environment(env_name):
    """
    Create and return a gym environment.

    Args:
    env_name (str): The name of the gym environment to create.

    Returns:
    gym.Env: The created gym environment.
    """
    env = gym.make(env_name)
    return env

def process_action(action):
    """
    Process the action to ensure it's in the correct format for the environment.

    Args:
    action (numpy.ndarray): The action from the agent.

    Returns:
    numpy.ndarray: The processed action.
    """
    # Ensure the action is a 1D numpy array
    return action.flatten()

def process_state(state):
    """
    Process the state from the environment to ensure it's in the correct format for the agent.

    Args:
    state (numpy.ndarray): The state from the environment.

    Returns:
    numpy.ndarray: The processed state.
    """
    # Ensure the state is a 1D numpy array
    return state.flatten()

# Example usage:
# env = create_environment("Pendulum-v1")
# state = env.reset()
# processed_state = process_state(state)
# action = agent.get_action(processed_state)
# processed_action = process_action(action)
# next_state, reward, done, _ = env.step(processed_action)
