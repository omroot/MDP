import numpy as np
import attr

@attr.s
class ValueIteration:
    """
    Value iteration algorithm for solving a Markov Decision Process (MDP).

    Parameters:
        env (gym.Env): The OpenAI Gym environment representing the MDP.
        gamma (float, optional): The discount factor for future rewards. Default is 1.0.
        threshold (float, optional): The convergence threshold for the algorithm. Default is 1e-6.
        max_iterations (int, optional): The maximum number of iterations for the algorithm. Default is 10000.
        inplace (bool, optional): If True, the algorithm performs in-place value iteration. Default is False.

    Attributes:
        optimal_state_values (np.ndarray): The optimal state values computed by the algorithm.
        optimal_policy (np.ndarray): The optimal policy derived from the optimal state values.
    """

    env = attr.ib()
    gamma = attr.ib(default=1.0)
    threshold = attr.ib(default=1e-6)
    max_iterations = attr.ib(default=10000)
    inplace = attr.ib(default=False)  # New attribute for algorithm version

    optimal_state_values = attr.ib(default=None)
    optimal_policy = attr.ib(default=None)

    def fit(self) -> None:
        """
        Fit the Value Iteration algorithm to the MDP environment.

        Performs value iteration using either in-place value iteration or state sweep value iteration.
        Computes the optimal state values and extracts the optimal policy.

        Returns:
            None
        """
        if self.inplace:
            self.optimal_state_values = self._inplace_value_iteration()
        else:
            self.optimal_state_values = self._state_sweep_value_iteration()

        self.optimal_policy = self.extract_policy(self.optimal_state_values)

    def extract_policy(self, value_table: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Extract the optimal policy from the computed state values.

        Parameters:
            value_table (np.ndarray): The computed optimal state values.
            gamma (float, optional): The discount factor for future rewards. Default is 1.0.

        Returns:
            np.ndarray: The optimal policy represented as an array of action indices for each state.
        """
        policy = np.zeros(self.env.observation_space.n, dtype=int)

        for state in range(self.env.observation_space.n):
            Q_table = np.zeros(self.env.action_space.n)

            for action in range(self.env.action_space.n):
                for next_sr in self.env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

            policy[state] = np.argmax(Q_table)

        return policy

    def _inplace_value_iteration(self):
        """
        Perform in-place value iteration to compute the optimal state values.

        Returns:
            np.ndarray: The computed optimal state values.
        """
        value_table = np.zeros(self.env.observation_space.n)

        for i in range(self.max_iterations):
            max_delta = 0.0

            for state in range(self.env.observation_space.n):
                old_state_value = value_table[state]
                value_table[state] = self._calculate_state_value(state, value_table)
                max_delta = max(max_delta, np.abs(value_table[state] - old_state_value))

            if max_delta < self.threshold:
                print(f'Value-iteration (in-place) converged at iteration #{i + 1}.')
                break

        return value_table

    def _state_sweep_value_iteration(self):
        """
        Perform state sweep value iteration to compute the optimal state values.

        Returns:
            np.ndarray: The computed optimal state values.
        """
        value_table = np.zeros(self.env.observation_space.n)
        updated_value_table = np.zeros(self.env.observation_space.n)

        for i in range(self.max_iterations):
            max_delta = 0.0

            for state in range(self.env.observation_space.n):
                updated_value_table[state] = self._calculate_state_value(state, value_table)
                max_delta = max(max_delta, np.abs(updated_value_table[state] - value_table[state]))

            value_table = np.copy(updated_value_table)

            if max_delta < self.threshold:
                print(f'Value-iteration (state sweep) converged at iteration #{i + 1}.')
                break

        return value_table

    def _calculate_state_value(self, state, value_table):
        """
        Calculate the state value for a given state using the Bellman equation.

        Parameters:
            state (int): The state index for which the value is calculated.
            value_table (np.ndarray): The current state values.

        Returns:
            float: The calculated state value.
        """
        Q_value = []

        for action in range(self.env.action_space.n):
            next_states_rewards = []
            for next_sr in self.env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                next_states_rewards.append(trans_prob * (reward_prob + self.gamma * value_table[next_state]))

            Q_value.append(np.sum(next_states_rewards))

        return max(Q_value)
