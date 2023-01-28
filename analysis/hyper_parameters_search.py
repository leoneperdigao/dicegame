import time
import random

import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from agent.dice_game_agent import MyAgent, play_game_with_agent
from dice_game import DiceGame


def play_with_agent(game, theta, gamma):
    test_agent = MyAgent(game, theta=theta, gamma=gamma)
    start_time = time.process_time()
    score = play_game_with_agent(test_agent, game)
    execution_time = time.process_time() - start_time
    return score, execution_time


def random_search(theta_range=(0.1, 50), gamma_range=(0.001, 0.999), num_iterations=1000):
    """
    Perform a random search for the optimal values of theta and gamma
    :param num_iterations: Number of iterations to perform the search
    :param theta_range: Tuple of range for theta (min, max)
    :param gamma_range: Tuple of range for gamma (min, max)
    :return: Tuple of optimal theta and gamma values
    """
    best_score = float('-inf')
    best_theta = None
    best_gamma = None

    theta_values = []
    gamma_values = []
    scores = []
    execution_times = []
    game = DiceGame()

    for i in range(num_iterations):
        theta = random.uniform(theta_range[0], theta_range[1])
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        score, execution_time = play_with_agent(game, theta, gamma)

        theta_values.append(theta)
        gamma_values.append(gamma)
        scores.append(score)
        execution_times.append(execution_time)

        if score > best_score:
            best_theta, best_gamma, best_score = theta, gamma, score

    print(f"best_theta={best_theta}, best_gamma={best_gamma}, best_score={best_score}")
    return theta_values, gamma_values, scores, execution_times


def grid_search(theta_range=np.linspace(0.1, 50), gamma_range=np.linspace(0.001, 0.99), num_iterations=1000):
    """Perform a grid search to find the optimal values of theta and gamma.
    game: an instance of the DiceGame class
    theta_range: a list of possible values for theta to search over
    gamma_range: a list of possible values for gamma to search over
    num_games: the number of games to play for each combination of theta and gamma
    """
    best_score = float('-inf')
    best_theta = None
    best_gamma = None

    theta_values = []
    gamma_values = []
    scores = []
    execution_times = []
    game = DiceGame()

    for _ in range(num_iterations):
        for theta in theta_range:
            for gamma in gamma_range:
                score, execution_time = play_with_agent(game, theta, gamma)

                theta_values.append(theta)
                gamma_values.append(gamma)
                scores.append(score)
                execution_times.append(execution_time)

                if score > best_score:
                    best_score = score
                    best_theta = theta
                    best_gamma = gamma

    print(f"best_theta={best_theta}, best_gamma={best_gamma}, best_score={best_score}")
    return theta_values, gamma_values, scores, execution_times


def plot_results(theta_values, gamma_values, scores, execution_times):
    """Plot the results of the game in a graph and table format"""
    # create table
    # table = PrettyTable()
    # table.field_names = ["Theta", "Gamma", "Score", "Execution Time (s)"]
    #
    # for theta, gamma, score, execution_time in zip(theta_values, gamma_values, scores, execution_times):
    #     table.add_row([theta, gamma, score, execution_time])

    # calculate average
    average_score = np.mean(scores)
    average_theta = np.mean(theta_values)
    average_gamma = np.mean(gamma_values)
    average_execution_time = np.mean(execution_times)
    # table.add_row(["Average Theta", "Average Gamma", "Average Score", "Average Execution Time (s)"])
    # table.add_row([average_theta, average_gamma, average_score, average_execution_time])

    # normalize execution times
    max_execution_time = max(execution_times)
    normalized_execution_times = [time / max_execution_time for time in execution_times]
    scaling_factor = 100

    # create scatter plot
    plt.scatter(theta_values, gamma_values, c=scores, s=[time * scaling_factor for time in normalized_execution_times])
    plt.xlabel('Theta')
    plt.ylabel('Gamma')
    plt.title('Hyperparameter tuning results')
    plt.colorbar(label='Score')
    plt.show()

    # print table
    # print(table)


if __name__ == "__main__":
    theta_values, gamma_values, scores, execution_times = random_search(num_iterations=10000)
    plot_results(theta_values, gamma_values, scores, execution_times)

