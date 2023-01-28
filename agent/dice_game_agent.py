from abc import ABC, abstractmethod
import random

from dice_game import DiceGame
import numpy as np


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game
        self.local_cache = {}

    @abstractmethod
    def play(self, state):
        pass

    def get_next_state(self, action, state):
        if (action, state) not in self.local_cache:
            self.local_cache[(action, state)] = self.game.get_next_states(action, state)
        return self.local_cache[(action, state)]


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return 0, 1, 2


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return 0, 1, 2
        else:
            return ()


class MyAgent(DiceGameAgent):
    """An AI agent for playing a dice game.
    This agent uses the value iteration algorithm to find the optimal policy.
    """
    def __init__(self, game, gamma=0.96, theta=0.1):
        """
        Parameters:
            game (DiceGame): The game that the agent will play.
            gamma (float): The discount factor for future rewards.
            theta (float): The threshold for the convergence of the value iteration algorithm.
        """
        super().__init__(game)
        self.__gamma = gamma
        self.__theta = theta
        self.__perform_value_iteration()

    def __initialize_state_value_array_and_policy(self):
        """Initialize the state-value array and policy dictionary for the value iteration algorithm.
        Returns:
            (dict, dict): A tuple with a zero-state-value array and an empty policy for each initial state.
        """
        state_value_array = {}
        policy = {}
        for state in self.game.states:
            state_value_array[state] = 0
            policy[state] = ()
        return state_value_array, policy

    def __calculate_state_value_sum(self, state_value_array, current_state, states, game_over, reward, probabilities):
        """Calculate the expected state-value for the next state given the current state,
        action and the transition probabilities.

        Args:
            state_value_array (dict): The state-value array.
            current_state (tuple): The current state of the game.
            states (list): The list of next states.
            game_over (bool): Whether the game is over.
            reward (float): The reward for the current state and action.
            probabilities (list): The transition probabilities for the next states.

        Returns:
            (float): The expected state-value for the next state.
        """
        state_value_sum = 0
        for state, probability in zip(states, probabilities):
            if not game_over:
                state_value_sum += probability * (reward + self.__gamma * state_value_array[state])
            else:
                state_value_sum += probability * (reward + self.__gamma * self.game.final_score(current_state))
        return state_value_sum

    def __perform_value_iteration(self):
        """Perform the value iteration algorithm to find the optimal policy for the current game.

        The value iteration algorithm is an iterative method to find the optimal policy for a given MDP (Markov Decision Process).
        The algorithm starts with initializing the state value array and policy to default values.
        Then, it iteratively updates the state value array and policy until the maximum change in the state value array is less than a given threshold (theta).
        The algorithm terminates when the maximum change is less than the threshold.
        """
        state_value_array, policy = self.__initialize_state_value_array_and_policy()
        delta_max = self.__theta + 1
        while delta_max >= self.__theta:
            delta_max = 0
            for current_state in self.game.states:
                current_state_value = state_value_array[current_state]
                max_action = 0
                for action in self.game.actions:
                    states, game_over, reward, probabilities = self.get_next_state(action, current_state)
                    state_value_sum = self.__calculate_state_value_sum(
                        state_value_array, current_state, states, game_over, reward, probabilities
                    )
                    if state_value_sum > max_action:
                        max_action = state_value_sum
                        policy[current_state] = action
                    state_value_array[current_state] = max_action
                delta_max = max(delta_max, abs(current_state_value - state_value_array[current_state]))

        self.__policy = policy

    def play(self, state):
        return self.__policy[state]


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    if verbose:
        print(f"Testing agent: \n\t{type(agent).__name__}")
    if verbose:
        print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if verbose:
            print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if verbose and not game_over:
            print(f"Dice: \t\t{state}")

    if verbose:
        print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    #  or delete the line to make it change each time it is run
    np.random.seed(1)

    game = DiceGame()

    agent1 = AlwaysHoldAgent(game)
    play_game_with_agent(agent1, game, verbose=True)

    print("\n")

    agent2 = PerfectionistAgent(game)
    play_game_with_agent(agent2, game, verbose=True)


if __name__ == "__main__":
    main()
