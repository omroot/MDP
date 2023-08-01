import numpy as np
import gym

def generate_arbitrary_policy(env: gym.Env)->dict:
    """
    Generates an arbitrary policy for the FrozenLake environment.

    Args:
        env (gym.Env): The FrozenLake environment.

    Returns:
        dict: A dictionary representing the policy, where policy[state] is a dictionary of action-probabilities.
    """

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Generate a random policy
    policy = {}
    for state in range(n_states):
        action_probabilities = np.random.rand(n_actions)
        action_probabilities /= np.sum(action_probabilities)  # Normalize to make it a probability distribution
        policy[state] = {action: prob for action, prob in enumerate(action_probabilities)}

    return policy