#Chapter 8, Example 8.2: Epsilon-greedy approach to multi-armed bandit
import numpy as np
import matplotlib.pyplot as plt

# Input data
reward_probs = [  # Reward probabilities for each arm's outcome (+1,-1)
    [0.55, 0.45],
    [0.4, 0.6],
    [0.6, 0.4],
    [0.5, 0.5],
    [0.3, 0.7]
]
ndecisions = len(reward_probs) # Number of decisions
nsamples = 10000
epsilon_values = [0, 0.01, 0.05, 0.1, 0.5]
results = {}

for epsilon in epsilon_values:
    rewards = []
    decision_values = np.zeros(ndecisions) # Estimated values for each decision
    decision_counts = np.zeros(ndecisions) # Selection counts for each decision
    for t in range(nsamples):
        # Epsilon-greedy decision selection
        if np.random.rand() < epsilon:
            d = np.random.randint(ndecisions) # Explore
        else:
            d = np.argmax(decision_values) # Exploit
        # Generate reward based on the selected decision's probabilities
        reward = np.random.choice([1, -1], p=reward_probs[d])
        rewards.append(reward)
        # Update estimated value for the selected decision
        decision_counts[d] += 1
        decision_values[d] += (reward - decision_values[d]) / decision_counts[d]
    # Compute average reward over time
    avg_reward = np.cumsum(rewards) / np.arange(1, nsamples + 1)
    results[epsilon] = avg_reward

# Plot results
plt.figure(figsize=(10, 6))
for epsilon, avg_rewards in results.items():
    plt.plot(avg_rewards, label=f"epsilon = {epsilon}")
    
plt.title("Average Reward vs. Iterations for epsilon-Greedy Alg (5-Armed Bandit)")
plt.xlabel("Number of Iterations")
plt.ylabel("Average Reward")
plt.grid()
plt.show()
