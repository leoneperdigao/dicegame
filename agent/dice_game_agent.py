from abc import ABC, abstractmethod
import random

from dice_game import DiceGame
import numpy as np


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass

    def get_next_states_cached(self, cache, action, state):
        if (action, state) not in cache:
            cache[(action, state)] = self.game.get_next_states(action, state)
        return cache[(action, state)]


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
    def __init__(self, game, gamma=0.99, theta=0.0001):
        """Initializes the agent by performing a value iteration
        After the value iteration is run an optimal policy is returned. This
        policy instructs agent on what action to take in any possible state.
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)
        self.__gamma = gamma
        self.__theta = theta
        self.__policy = {}
        self.__v_arr = {}
        self.__local_cache = {}

        self.__init_policy()

    def __init_policy(self):
        """Initialize the policy by setting all actions to a random action"""
        for state in self.game.states:
            self.__v_arr[state] = 0
            self.__policy[state] = random.choice(self.game.actions)



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
