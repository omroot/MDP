import gym
import numpy as np
import attr
from mdp.policyEvaluation import PolicyEvaluator

def compare_policies(policy_1: dict, policy_2: dict, number_actions: int, number_states: int) -> bool:
    """
    Compare two policies to check if they are equal.

    Args:
        policy_1 (dict): The first policy represented as a dictionary.
        policy_2 (dict): The second policy represented as a dictionary.
        number_actions (int): The number of actions in the policy.
        number_states (int): The number of states in the policy.

    Returns:
        bool: True if the policies are equal, False otherwise.
    """
    policy_array_1 = []
    policy_array_2 = []

    for s in range(number_states):
        for a in range(number_actions):
            policy_array_1.append(policy_1[s][a])
            policy_array_2.append(policy_2[s][a])

    return policy_array_1 == policy_array_2

def improve_policy(env, value_function_vector: np.ndarray, number_actions: int, number_states: int, discount_rate: float):
    """
    Improves a given policy based on the current value function estimates.

    Args:
        env (gym.Env): The environment.
        value_function_vector (np.ndarray): The value function estimates for each state.
        number_actions (int): The number of actions in the policy.
        number_states (int): The number of states in the policy.
        discount_rate (float): Discount factor for future rewards.

    Returns:
        tuple: A tuple containing the improved policy (dictionary) and the Q-values matrix (2D numpy array).
    """
    q_values_matrix = np.zeros((number_states, number_actions))
    improved_policy = {}

    for state in range(number_states):
        improved_policy[state] = {i: 0 for i in range(number_actions)}

    for state_index in range(number_states):
        for action_index in range(number_actions):
            for probability, next_state, reward, is_terminal_state in env.P[state_index][action_index]:
                q_values_matrix[state_index, action_index] += probability * (reward + discount_rate * value_function_vector[next_state])

        best_action_index = np.where(q_values_matrix[state_index, :] == np.max(q_values_matrix[state_index, :]))[0][0]
        improved_policy[state_index][best_action_index] = 1

    return improved_policy, q_values_matrix

@attr.s
class PolicyIterator:
    """
    Class for performing Policy Iteration on an MDP.

    Attributes:
        env (gym.Env): The environment to evaluate the policy on.
        discount_rate (float): Discount factor for future rewards (default: 0.9).
        state_number (int): The number of states in the policy (default: 16).
        action_number (int): The number of actions in the policy (default: 4).
        max_iterations_policy_iteration (int): Maximum number of iterations for the policy iteration algorithm (default: 1000).
        max_iterations_iterative_policy_evaluation (int): Maximum number of iterations for iterative policy evaluation (default: 1000).
        convergence_tolerance_iterative_policy_evaluation (float): Convergence threshold for iterative policy evaluation (default: 10**(-6)).
        synchronous_evaluation (bool): If True, use synchronous evaluation in policy iteration (default: True).
    """

    env = attr.ib()
    discount_rate = attr.ib(default=0.9)
    state_number = attr.ib(default=16)
    action_number = attr.ib(default=4)
    max_iterations_policy_iteration = attr.ib(default=1000)
    max_iterations_iterative_policy_evaluation = attr.ib(default=1000)
    convergence_tolerance_iterative_policy_evaluation = attr.ib(default=10**(-6))
    synchronous_evaluation = attr.ib(default=True)

    def policy_iteration(self) -> dict:
        """
        Perform Policy Iteration to find the optimal policy.

        Returns:
            dict: The final optimal policy represented as a dictionary.
        """
        initial_policy = {}
        for state in range(self.state_number):
            initial_policy[state] = {i: 1/self.action_number for i in range(self.action_number)}

        current_policy = initial_policy

        for iteration in range(self.max_iterations_policy_iteration):
            print("Iteration - {} - of policy iteration algorithm".format(iteration))
            if iteration == 0:
                current_policy = initial_policy

            policy_evaluator = PolicyEvaluator(env=self.env,
                                               policy=current_policy,
                                               gamma=self.discount_rate,
                                               theta=self.convergence_tolerance_iterative_policy_evaluation,
                                               max_iterations=self.max_iterations_iterative_policy_evaluation)

            if self.synchronous_evaluation:
                value_function_vector_computed, _ = policy_evaluator.synchronous_state_sweep_policy_evaluation()
            else:
                value_function_vector_computed, _ = policy_evaluator.asynchronous_inplace_policy_evaluation()

            improved_policy, q_values_matrix = improve_policy(self.env,
                                                              value_function_vector_computed,
                                                              self.action_number,
                                                              self.state_number,
                                                              self.discount_rate)

            if compare_policies(current_policy, improved_policy, self.action_number, self.state_number):
                current_policy = improved_policy
                print("Policy iteration algorithm converged!")
                break

            current_policy = improved_policy

        return current_policy

# Example usage
if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    policy_iterator = PolicyIterator(env=env)
    final_policy = policy_iterator.policy_iteration()

    print("Final Policy:")
    print(final_policy)
