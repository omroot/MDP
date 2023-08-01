import numpy as np
import attr

@attr.s
class PolicySimulator:
    """
    A policy simulator for evaluating the performance of a given policy in a given environment.

    Parameters:
        env (gym.Env): The OpenAI Gym environment to run the simulation on.
        policy (np.ndarray): The policy to evaluate, represented as an array of action indices for each state.
        episodes (int, optional): The number of episodes to run for evaluation. Default is 1000.
        steps_list (list, optional): A list to store the number of steps taken in each episode. Default is an empty list.
        misses (int, optional): The number of episodes where the agent did not reach the goal state. Default is 0.

    Methods:
        run_single_episode(): Run a single episode using the given policy and return the number of steps taken.
        run_episodes(): Run multiple episodes using the given policy and collect performance statistics.
        print_performance_metrics(): Print the performance metrics, including average steps to reach the goal and the miss rate.
        evaluate_policy(): Evaluate the given policy using the specified number of episodes and print the results.
    """

    env = attr.ib()
    policy = attr.ib()
    episodes = attr.ib(default=1000)
    steps_list = attr.ib(factory=list)
    misses = attr.ib(default=0)

    def run_single_episode(self):
        """
        Run a single episode using the given policy and return the number of steps taken.

        Returns:
            int: The number of steps taken in the episode.
                  If the agent reaches the goal state, returns the number of steps taken.
                  If the agent does not reach the goal state, returns positive infinity (float('inf')).
        """
        observation = self.env.reset()
        steps_taken = 0

        while True:
            action = self.policy[observation]
            observation, reward, done, _ = self.env.step(action)
            steps_taken += 1

            if done and reward == 1:
                return steps_taken
            elif done and reward == 0:
                return float('inf')

    def run_episodes(self):
        """
        Run multiple episodes using the given policy and collect performance statistics.

        Returns:
            None
        """
        for episode in range(self.episodes):
            steps_taken = self.run_single_episode()
            self.steps_list.append(steps_taken)
            if steps_taken == float('inf'):
                self.misses += 1

    def print_performance_metrics(self):
        """
        Print the performance metrics, including average steps to reach the goal and the miss rate.

        Returns:
            None
        """
        print('----------------------------------------------')
        print(f'You took an average of {np.mean(self.steps_list):.0f} steps to reach the goal.')
        print(f'And you fell in the hole {(self.misses / len(self.steps_list)) * 100:.2f}% of the times.')
        print('----------------------------------------------')

    def evaluate_policy(self):
        """
        Evaluate the given policy using the specified number of episodes and print the results.

        Returns:
            None
        """
        self.run_episodes()
        self.print_performance_metrics()
