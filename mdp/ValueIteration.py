import numpy as np
import attr

@attr.s
class ValueIteration:
    env = attr.ib()
    gamma = attr.ib(default=1.0)
    threshold = attr.ib(default=1e-6)
    max_iterations = attr.ib(default=10000)
    inplace = attr.ib(default=False)  # New attribute for algorithm version

    optimal_state_values = attr.ib(default=None)
    optimal_policy = attr.ib(default=None)

    def fit(self) -> None:
        if self.inplace:
            self.optimal_state_values = self._inplace_value_iteration()
        else:
            self.optimal_state_values = self._state_sweep_value_iteration()

        self.optimal_policy = self.extract_policy(self.optimal_state_values)


    def extract_policy(self, value_table: np.ndarray, gamma: float = 1.0) -> np.ndarray:
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
        Q_value = []

        for action in range(self.env.action_space.n):
            next_states_rewards = []
            for next_sr in self.env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                next_states_rewards.append(trans_prob * (reward_prob + self.gamma * value_table[next_state]))

            Q_value.append(np.sum(next_states_rewards))

        return max(Q_value)



