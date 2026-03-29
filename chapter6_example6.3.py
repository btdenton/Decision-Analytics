#Chapter 6, Example 6.3: Simulation of a non-Markovian policy for stock option
import numpy as np

# Input data
T = 30
strike_price = 1000
initial_price = 980
initial_state = "I"
nsamples = 10000
P = np.array([[0.80, 0.20],[0.25, 0.75]]) # Transition probability matrix
states = ["I", "D"]
state_effects = {"I": 1.10, "D": 0.90} #effects of state on price
rewards = []

# Markov chain simulation to find expected reward
np.random.seed(42) #Fix seed for repeatability
for _ in range(nsamples):
    current_state = initial_state # Initialize the starting state and price
    current_price = initial_price
    prices = [] # daily closing stock prices
    reward = 0
    for t in range(1, T + 1):
        prices.append(current_price) # Record the current price
        mean_price = np.mean(prices)  # Calculate mean price up to the current day
        # Decision rule to execute the option
        if mean_price > 1.2 * strike_price or (t == T and mean_price > strike_price):
            reward = max(mean_price - strike_price, 0)
            break 
        # Transition to the next state
        if current_state == "I":
            next_state = np.random.choice(states, p=P[0])
        else:
            next_state = np.random.choice(states, p=P[1])
        current_state = next_state  # Update the state and stock price
        current_price = current_price * state_effects[current_state]
    rewards.append(reward)

mean = np.mean(rewards)

# Calculate the 95% confidence interval
std_error = np.std(rewards) / np.sqrt(nsamples)
z_score = 1.96 # Z-score for 95% confidence
CI = (mean - z_score * std_error, mean + z_score * std_error)

#Print results
print(f"Expected Reward: ${mean:.2f}")
print(f"95% Confidence Interval: (${CI[0]:.2f}, ${CI[1]:.2f})")
