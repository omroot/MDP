import gym
import numpy as np
import attr

@attr.s
class PolicyEvaluator:
    """
    Class for evaluating policies in Markov Decision Processes (MDPs).

    Attributes:
        env (gym.Env): The environment to evaluate the policy on.
        policy (dict): A dictionary representing the policy, where policy[state] is a dictionary of action-probabilities.
        gamma (float): Discount factor for future rewards (default: 1.0).
        theta (float): Convergence threshold. The algorithm stops when the change in state-values is less than theta (default: 1e-10).
        max_iterations (int): Maximum number of iterations for the algorithm (default: 1000).
        verbose (bool): If True, prints detailed information about each iteration (default: False).
    """

    env = attr.ib()
    policy = attr.ib()
    gamma = attr.ib(default=1.0)
    theta = attr.ib(default=1e-10)
    max_iterations = attr.ib(default=1000)
    verbose = attr.ib(default=False)

    def synchronous_state_sweep_policy_evaluation(self) -> dict:
        """
        Performs synchronous state sweep policy evaluation to estimate the state-values of the given policy.

        Returns:
            dict: A dictionary containing the estimated state-values for each state.
            int: The number of iterations taken to converge.
        """
        state_values = {state: 0.0 for state in range(self.env.nS)}

        for i in range(self.max_iterations):
            delta = 0.0
            new_state_values = state_values.copy()

            for state in self.env.P:
                v_outerSum = 0.0

                for action, action_prob in self.policy[state].items():
                    v_innerSum = 0
                    for probability, next_state, reward, is_terminal_state in self.env.P[state][action]:
                        v_innerSum += probability * (reward + self.gamma * state_values[next_state])
                    v_outerSum += action_prob * v_innerSum

                new_state_values[state] = v_outerSum
                delta = max(delta, abs(new_state_values[state] - state_values[state]))

                if self.verbose:
                    print(f"iteration: {i}, state {state}, action {action}, delta {delta}")

            state_values = new_state_values

            if delta < self.theta:
                break

        return state_values, i

    def asynchronous_inplace_policy_evaluation(self) -> dict:
        """
        Performs in-place synchronous state sweep policy evaluation to estimate the state-values of the given policy.

        Returns:
            dict: A dictionary containing the estimated state-values for each state.
            int: The number of iterations taken to converge.
        """
        state_values = {state: 0.0 for state in range(self.env.nS)}

        for i in range(self.max_iterations):
            delta = 0.0

            for state in self.env.P:
                v_outerSum = 0.0

                for action, action_prob in self.policy[state].items():
                    v_innerSum = 0
                    for probability, next_state, reward, is_terminal_state in self.env.P[state][action]:
                        v_innerSum += probability * (reward + self.gamma * state_values[next_state])
                    v_outerSum += action_prob * v_innerSum

                delta = max(delta, abs(v_outerSum - state_values[state]))
                state_values[state] = v_outerSum

                if self.verbose:
                    print(f"iteration: {i}, state {state}, action {action}, delta {delta}")

            if delta < self.theta:
                break

        return state_values, i

# Example usage
if __name__ == "__main__":
    # Assuming 'env' is the environment object and 'policy' is the policy to evaluate
    env = gym.make('Taxi-v3')
    policy = {state: {action: 1.0 / env.action_space.n for action in range(env.action_space.n)} for state in range(env.nS)}

    policy_evaluator = PolicyEvaluator(env=env, policy=policy)
    sync_state_values, sync_iterations = policy_evaluator.synchronous_state_sweep_policy_evaluation()
    async_state_values, async_iterations = policy_evaluator.asynchronous_inplace_policy_evaluation()

    print("Synchronous State Sweep Policy Evaluation:")
    print(sync_state_values)
    print("Number of iterations:", sync_iterations)

    print("Asynchronous In-Place Policy Evaluation:")
    print(async_state_values)
    print("Number of iterations:", async_iterations)
