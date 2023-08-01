import attr
import numpy as np
import gym

@attr.s
class MDP:
    """
    Markov Decision Process (MDP) class representing a finite MDP.

    Parameters:
        num_states (int): The number of states in the MDP.
        num_actions (int): The number of actions in the MDP.
        transition_matrix (np.ndarray): The transition matrix of shape (num_states, num_actions, num_states)
                                       representing the probabilities of transitioning to the next state given a state-action pair.
        reward_matrix (np.ndarray): The reward matrix of shape (num_states, num_actions) representing the rewards for each state-action pair.
        discount_factor (float): The discount factor for future rewards.

    Methods:
        get_transition_probabilities(state, action): Get the transition probabilities from a given state-action pair to all possible next states.
        get_reward(state, action): Get the reward for a given state-action pair.
    """

    num_states: int = attr.ib()
    num_actions: int = attr.ib()
    transition_matrix: np.ndarray = attr.ib()
    reward_matrix: np.ndarray = attr.ib()
    discount_factor: float = attr.ib()

    def get_transition_probabilities(self, state, action):
        """
        Get the transition probabilities from a given state-action pair to all possible next states.

        Parameters:
            state (int): The current state.
            action (int): The action to be taken.

        Returns:
            np.ndarray: A 1D array of shape (num_states,) containing the probabilities of transitioning to each next state.
        """
        return self.transition_matrix[state, action]

    def get_reward(self, state, action):
        """
        Get the reward for a given state-action pair.

        Parameters:
            state (int): The current state.
            action (int): The action to be taken.

        Returns:
            float: The reward for the given state-action pair.
        """
        return self.reward_matrix[state, action]

class GymToMDPWrapper(MDP):
    """
    Wrapper class to convert an OpenAI Gym environment into an MDP format.

    Parameters:
        env (gym.Env): The OpenAI Gym environment.

    Attributes:
        num_states (int): The number of states in the Gym environment.
        num_actions (int): The number of actions in the Gym environment.
        transition_matrix (np.ndarray): The transition matrix of shape (num_states, num_actions, num_states)
                                       representing the probabilities of transitioning to the next state given a state-action pair.
        reward_matrix (np.ndarray): The reward matrix of shape (num_states, num_actions) representing the rewards for each state-action pair.
        discount_factor (float): The discount factor for future rewards.
    """

    def __init__(self, env):
        self.env = env
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        transition_matrix, reward_matrix = self._create_transition_and_reward_matrices(env)
        discount_factor = 0.99  # You can set the discount factor based on your requirements.

        super().__init__(num_states=num_states,
                         num_actions=num_actions,
                         transition_matrix=transition_matrix,
                         reward_matrix=reward_matrix,
                         discount_factor=discount_factor)

    def _create_transition_and_reward_matrices(self, env):
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        transition_matrix = np.zeros((num_states, num_actions, num_states))
        reward_matrix = np.zeros((num_states, num_actions))

        for state in range(num_states):
            for action in range(num_actions):
                transitions = env.P[state][action]

                for trans_prob, next_state, reward, _ in transitions:
                    transition_matrix[state, action, next_state] = trans_prob
                    reward_matrix[state, action] = reward

        return transition_matrix, reward_matrix
