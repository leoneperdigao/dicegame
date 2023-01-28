# The Dice Game

AI written in Python that can play a simple dice game. 

## Game rules:

The player starts with 0 points
Roll three fair six-sided dice

Now choose one of the following:
* Stick, accept the values shown. If two or more dice show the same values, then all of them are flipped upside down: 1 becomes 6, 2 becomes 5, 3 becomes 4, and vice versa. The total is then added to your points and this is your final score.
* OR reroll the dice. You may choose to hold any combination of the dice on the current value shown. Rerolling costs you 1 point – so during the game and perhaps even at the end your score may be negative. You then make this same choice again.

The best possible score for this game is 18 and is achieved by rolling three 1s on the first roll.
The reroll penalty prevents you from rolling forever to get this score. If the value of the current dice is greater 
than the expected value of rerolling them (accounting for the penalty), then you should stick.

## Methodology

The methodology used in this solution is the value iteration algorithm for Markov Decision Processes (MDPs). 
This algorithm is an iterative method to find the optimal policy for a given MDP. 
The algorithm starts by initializing the state value array and policy to default values. 
Then, it iteratively updates the state value array and policy until the maximum change in the state value array is less than a given threshold (theta). 
The algorithm terminates when the maximum change is less than the threshold.

Value iteration is similar to another algorithm called policy iteration, but it has some key differences. 
Policy iteration starts by initializing the policy to random actions and then evaluates and improves the policy until it converges to the optimal policy. 
On the other hand, value iteration evaluates the state value function and improves it until it converges to the optimal state value function. 
Both algorithms are guaranteed to converge to the optimal solution for any MDP, but value iteration generally requires fewer iterations to converge.

The math behind the `perform_value_iteration` method is the Bellman equation. 
The Bellman equation is a fundamental concept in the field of dynamic programming and is used to calculate the value of a state in a Markov Decision Process (MDP), which is the core of the value iteration algorithm. 
The equation can be represented as:

```math
V(s) = max(R(s,a) + γ * Σ(T(s,a,s') * V(s')))
```

Where:

- `V(s)` is the value of the current state `s`.
- `R(s,a)` is the immediate reward for taking action `a` in state `s`.
- `γ` is the discount factor for future rewards.
- `T(s,a,s')` is the transition probability of reaching state `s'` from state s when taking action `a`.
- `V(s')` is the value of the next state `s'`.

This formula is used to calculate the expected state-value for the next state given the current state, action and the transition probabilities. 
The algorithm then updates the state-value array and the policy based on this expected value.

### Code walk-trough

This solution provides an AI agent for playing a dice game, which uses the value iteration algorithm to find the optimal policy for the game. 
The value iteration algorithm is a method to find the optimal policy for a given Markov Decision Process (MDP). 
The agent extends the `DiceGameAgent` class, which is likely an abstraction for a game that can be played using dice.

The `__init__` method of the agent takes in the game that the agent will play, as well as two parameters: gamma and theta. 
The gamma parameter is a discount factor for future rewards, which is used to weigh the importance of future rewards relative to immediate rewards. 
The theta parameter is a threshold for the convergence of the value iteration algorithm. 
The `__init__` method calls the `__perform_value_iteration method`, which performs the value iteration algorithm to find the optimal policy for the game.

The `__initialize_state_value_array_and_policy` method initializes the state-value array and policy dictionary for the value iteration algorithm. 
It creates an empty state-value array and an empty policy dictionary, then sets the value of each key in the array to 0 and the value of each key in the dictionary to an empty tuple.

The `__calculate_state_value_sum` method calculates the expected state-value for the next state, given the current state, action and the transition probabilities. 
It takes in the current state-value array, the current state, the list of next states, whether the game is over, the reward for the current state and action and the transition probabilities for the next states. 
It iterates over the list of next states and the corresponding transition probabilities, and calculates the expected state-value for the next state.

The `__perform_value_iteration` method performs the value iteration algorithm to find the optimal policy for the current game. 
It starts by initializing the state-value array and policy to default values by calling the `__initialize_state_value_array_and_policy` method. 
Then, it iteratively updates the state-value array and policy until the maximum change in the state-value array is less than the given threshold (theta). 
The algorithm terminates when the maximum change is less than the threshold.

The `play` method takes in the current state, it returns the action stored in the `__policy` dictionary corresponding to the given state.

The key part of the code is the `__perform_value_iteration` method, which is where the value iteration algorithm is implemented. 
The method uses the get_next_state method to get the next states, rewards, probability and game over status for the current action and state. 
It uses these values to update the state-value array and the policy dictionary.

## Results

Results of 1000 executions with random gamma and theta values:

TODO show table

### Graph

![image info](./analysis/results/graph-1000.png)


## Installation

1. Install Poetry by running `pip install poetry` in your command line.
2. Clone this repository by running `git clone https://github.com/leoneperdigao/dicegame.git` in your command line.
3. Navigate to the root directory of the cloned repository by running `cd dicegame`.
4. Use Poetry to install the dependencies by running poetry install. This will create a virtual environment for the project and install all the required packages.
5. Run the game by executing `poetry run python dice_game.py`.
6. Enjoy the game!