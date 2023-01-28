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
    def __init__(self, game, gamma=0.95, theta=0.001):
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
        self.__policy, self.__v_arr = self.__policy_iteration()

    def __init_policy(self):
        """Initialize the policy by setting all actions to a random action"""
        for state in self.game.states:
            self.__v_arr[state] = 0
            self.__policy[state] = random.choice(self.game.actions)

    def __policy_iteration(self):
        """
        Perform policy iteration algorithm to find optimal policy
        """
        policy = {state: self.game.actions[0] for state in self.game.states}
        v_arr = {state: 0 for state in self.game.states}
        while True:
            v_arr = self.__policy_evaluation(policy, v_arr)
            policy_stable = True
            for state in self.game.states:
                old_action = policy[state]
                policy[state] = self.__policy_improvement(policy, v_arr, state)
                if old_action != policy[state]:
                    policy_stable = False
            if policy_stable:
                break
        return policy, v_arr

    def __policy_evaluation(self, policy, v_arr):
        """
        Evaluate the current policy
        """
        while True:
            delta_max = 0
            for state in self.game.states:
                v = v_arr[state]
                action = policy[state]
                states, game_over, reward, probabilities = self.get_next_states_cached(
                    self.__local_cache, action, state
                )
                v_arr[state] = sum(
                    prob * (reward + self.__gamma * v_arr[s1]) for s1, prob in zip(states, probabilities) if not game_over
                )
                delta_max = max(delta_max, abs(v - v_arr[state]))
            if delta_max < self.__theta:
                break
        return v_arr

    def __policy_improvement(self, policy, v_arr, state):
        """
        Improve current policy
        """
        max_action_val = float('-inf')
        best_action = policy[state]
        for action in self.game.actions:
            states, game_over, reward, probabilities = self.get_next_states_cached(
                self.__local_cache, action, state
            )
            action_val = sum(
                prob * (reward + self.__gamma * v_arr[s1]) for s1, prob in zip(states, probabilities) if not game_over
            )
            if action_val > max_action_val:
                max_action_val = action_val
                best_action = action
        return best_action

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
