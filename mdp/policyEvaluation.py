import gym
import numpy as np


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

def synchronous_state_sweep_policy_evaluation(env: gym.Env,
                                              policy: dict,
                                              gamma: float=1.0,
                                              theta: float=1e-10,
                                              max_iterations: int=1000,
                                              verbose:bool = False)->dict:
    """
    Performs synchronous state sweep policy evaluation to estimate the state-values of a given policy.

    Args:
        env (gym.Env): The environment.
        policy (dict): A dictionary representing the policy, where policy[state] is a dictionary of action-probabilities.
        gamma (float): Discount factor for future rewards (default: 1.0).
        theta (float): Convergence threshold. The algorithm stops when the change in state-values is less than theta (default: 1e-6).
        max_iterations (int): Maximum number of iterations for the algorithm (default: 1000).

    Returns:
        dict: A dictionary containing the estimated state-values for each state.
    """

    # Initialize state-values arbitrarily (e.g., all zeros)
    state_values = {state: 0.0 for state in list(range(env.nS))}

    for i in range(max_iterations):
        delta = 0.0

        # Make a copy of state_values to perform synchronous updates
        new_state_values = state_values.copy()

        # Iterate over all states
        for state in env.P:
            v_outerSum = 0.0
            # Calculate the value of the state based on the current policy
            for action, action_prob in policy[state].items():
                v_innerSum = 0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    v_innerSum += probability * (reward + gamma * state_values[nextState])
                v_outerSum = v_outerSum + action_prob * v_innerSum
            # Update the new state value and calculate the change in value
            new_state_values[state] = v_outerSum
            delta = max(delta, abs(new_state_values[state] - state_values[state]))
            if verbose:
                print(f"iteration: {i}, state {state}, action {action} , delta {delta} ")

        # Update the state-values with the new values
        state_values = new_state_values

        # Check for convergence
        if delta < theta:
            break

    return state_values, i

def asynchronous_inplace_policy_evaluation(env: gym.Env,
                                           policy: dict,
                                           gamma: float=1.0,
                                           theta: float=1e-10,
                                           max_iterations: int=1000,
                                           verbose: bool = False)->np.ndarray:
    """
    Performs in-place synchronous state sweep policy evaluation to estimate the state-values of a given policy.

    Args:
        env (gym.Env): The environment.
        policy (dict): A dictionary representing the policy, where policy[state] is a dictionary of action-probabilities.
        gamma (float): Discount factor for future rewards (default: 1.0).
        theta (float): Convergence threshold. The algorithm stops when the change in state-values is less than theta (default: 1e-6).
        max_iterations (int): Maximum number of iterations for the algorithm (default: 1000).

    Returns:
        dict: A dictionary containing the estimated state-values for each state.
    """

    # Initialize state-values arbitrarily (e.g., all zeros)
    state_values = {state: 0.0 for state in list(range(env.nS))}

    for i in range(max_iterations):
        delta = 0.0

        # Iterate over all states
        for state in env.P:
            v_outerSum = 0.0
            # Calculate the value of the state based on the current policy
            for action, action_prob in policy[state].items():
                v_innerSum = 0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    v_innerSum += probability * (reward + gamma * state_values[nextState])
                v_outerSum = v_outerSum + action_prob * v_innerSum
            # Update the new state value and calculate the change in value

            delta = max(delta, abs(v_outerSum - state_values[state]))
            state_values[state] = v_outerSum
            if verbose:
                print(f"iteration: {i}, state {state}, action {action} , delta {delta} ")


        # Check for convergence
        if delta < theta:
            break

    return state_values, i
