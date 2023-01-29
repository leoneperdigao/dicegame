import time
import random

import numpy as np
import matplotlib.pyplot as plt

from agent.dice_game_agent import MyAgent, play_game_with_agent
from dice_game import DiceGame


def play_with_agent(game, theta, gamma):
    test_agent = MyAgent(game, theta=theta, gamma=gamma)
    start_time = time.process_time()
    score = play_game_with_agent(test_agent, game)
    execution_time = time.process_time() - start_time
    return score, execution_time


def run_simulation(game, theta=None, gamma=None, theta_range=(0.1, 50), gamma_range=(0.001, 1.0), num_iterations=1000):
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
    best_execution_time = float('inf')

    theta_values = []
    gamma_values = []
    scores = []
    execution_times = []

    for i in range(num_iterations):
        if not theta and not gamma:
            theta = random.uniform(theta_range[0], theta_range[1])
            gamma = random.uniform(gamma_range[0], gamma_range[1])

        score, execution_time = play_with_agent(game, theta, gamma)

        theta_values.append(theta)
        gamma_values.append(gamma)
        scores.append(score)
        execution_times.append(execution_time)

        if score > best_score and execution_time < best_execution_time:
            best_execution_time = execution_time
            best_theta, best_gamma, best_score = theta, gamma, score

    print(f"best_theta={best_theta}, best_gamma={best_gamma}, best_score={best_score}, best_execution_time={best_execution_time}")
    return theta_values, gamma_values, scores, execution_times


def plot_results(theta_values, gamma_values, scores, execution_times):
    """Plot the results of the game in a graph and table format"""
    # calculate average
    average_score = np.mean(scores)
    average_theta = np.mean(theta_values)
    average_gamma = np.mean(gamma_values)
    average_execution_time = np.mean(execution_times)

    print(
        f"average_theta={average_theta}, average_gamma={average_gamma}, average_score={average_score}, "
        f"average_execution_time={average_execution_time}"
    )

    # normalize execution times
    max_execution_time = max(execution_times)
    normalized_execution_times = [time / max_execution_time for time in execution_times]
    scaling_factor = 100

    # create scatter plot
    fig, axs = plt.subplots(1, 4, figsize=(36, 12))
    cax0 = axs[0].scatter(theta_values, gamma_values, c=normalized_execution_times, s=scaling_factor)
    axs[0].set_xlabel('Theta')
    axs[0].set_ylabel('Gamma')
    axs[0].set_title('Theta vs Gamma')
    fig.colorbar(cax0, ax=axs[0], label='Execution(s) time')

    cax1 = axs[1].scatter(theta_values, gamma_values, c=scores)
    axs[1].set_xlabel('Theta')
    axs[1].set_ylabel('Gamma')
    axs[1].set_title('Theta vs Gamma')
    fig.colorbar(cax1, ax=axs[1], label='Score')

    cax2 = axs[2].scatter(theta_values, execution_times, c=scores)
    axs[2].set_xlabel('Theta')
    axs[2].set_ylabel('Execution(s) time')
    axs[2].set_title('Theta vs Execution Time')
    fig.colorbar(cax2, ax=axs[2], label='Score')

    cax3 = axs[3].scatter(gamma_values, execution_times, c=scores)
    axs[3].set_xlabel('Gamma')
    axs[3].set_ylabel('Execution(s) time')
    axs[3].set_title('Gamma vs Execution Time')
    fig.colorbar(cax3, ax=axs[3], label='Score')

    axs[0].plot(average_theta, average_gamma, 'ro', markersize=12)
    axs[1].plot(average_theta, average_gamma, 'ro', markersize=12)
    axs[2].plot(average_theta, average_execution_time, 'ro', markersize=12)
    axs[3].plot(average_gamma, average_execution_time, 'ro', markersize=12)

    plt.show()


def plot_linear_results_for_theta(theta_values, scores, execution_times):
    plt.plot(theta_values, scores)
    plt.xlabel('Theta')
    plt.ylabel('Score')
    plt.title('Average Score of 1000 games')
    plt.grid(True)
    plt.show()

    plt.plot(theta_values, execution_times)
    plt.xlabel('Theta')
    plt.ylabel('Execution time')
    plt.title('Average time of 1000 games')
    plt.grid(True)
    plt.show()


def plot_linear_results_for_gamma(gamma_values, scores, execution_times):
    plt.plot(gamma_values, scores)
    plt.xlabel('Gamma')
    plt.ylabel('Score')
    plt.title('Average Score of 1000 games')
    plt.grid(True)
    plt.show()

    plt.plot(gamma_values, execution_times)
    plt.xlabel('Gamma')
    plt.ylabel('Execution time')
    plt.title('Average time of 1000 games')
    plt.grid(True)
    plt.show()


def run_simulations():
    np.random.seed()
    game = DiceGame()
    theta_values, gamma_values, scores, execution_times = run_simulation(game=game, num_iterations=1000)
    plot_results(theta_values, gamma_values, scores, execution_times)

    game = DiceGame(dice=2, sides=3)
    theta_values, gamma_values, scores, execution_times = run_simulation(game=game, num_iterations=1000)
    plot_results(theta_values, gamma_values, scores, execution_times)


def run_fine_tune_theta(game, theta_candidates, gamma_candidates):
    scores = []
    execution_times = []
    games_sample = 1000

    average_gamma = [np.mean(gamma_candidates)]

    for gamma in average_gamma:
        for theta in theta_candidates:
            total_score = 0
            total_time = 0

            game.reset()

            start_time = time.process_time()
            test_agent = MyAgent(game, theta=theta, gamma=gamma)
            total_time += time.process_time() - start_time

            for _ in range(games_sample):
                start_time = time.process_time()
                score = play_game_with_agent(test_agent, game)
                total_time += time.process_time() - start_time
                total_score += score

            scores.append(total_score / games_sample)
            execution_times.append(total_time)

    plot_linear_results_for_theta(theta_candidates, scores, execution_times)


def run_fine_tune_gamma(game, theta_candidates, gamma_candidates):
    scores = []
    execution_times = []
    games_sample = 1000

    average_theta = [np.mean(theta_candidates)]

    for gamma in gamma_candidates:
        for theta in average_theta:
            total_score = 0
            total_time = 0

            game.reset()

            start_time = time.process_time()
            test_agent = MyAgent(game, theta=theta, gamma=gamma)
            total_time += time.process_time() - start_time

            for _ in range(games_sample):
                start_time = time.process_time()
                score = play_game_with_agent(test_agent, game)
                total_time += time.process_time() - start_time
                total_score += score

            scores.append(total_score / games_sample)
            execution_times.append(total_time)

    plot_linear_results_for_gamma(gamma_candidates, scores, execution_times)


def fine_tune_hyperparameters():
    theta_candidates = np.arange(0.1, 2, 0.1)
    gamma_candidates = np.arange(0.850, 0.999, 0.001)

    np.random.seed()
    game = DiceGame()
    run_fine_tune_theta(game, theta_candidates, gamma_candidates)
    run_fine_tune_gamma(game, theta_candidates, gamma_candidates)

    np.random.seed()
    game = DiceGame(dice=2, sides=3)
    run_fine_tune_theta(game, theta_candidates, gamma_candidates)
    run_fine_tune_gamma(game, theta_candidates, gamma_candidates)


if __name__ == "__main__":
    fine_tune_hyperparameters()

