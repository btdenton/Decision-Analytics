#Chapter 3, Example 3.4: Resource allocation
import numpy as np

# Input data
costs = [4,3,5] # per unit costs of the items
rewards = [11,7,12] # per unit reward for projects 
capacity = 10 # Total capacity for projects (budget)
T = len(costs) # Number of items

# Optimal value function initialization
v = np.zeros(capacity + 1)

# Compute the maximum benefit for each capacity
for w in range(capacity + 1):
  for i in range(T):
    if costs[i] <= w:
      v[w] = max(v[w], v[w - costs[i]] + rewards[i])

# Reconstruct the optimal solution
solution = np.zeros(T)
w = capacity
while w > 0:
  for i in range(T):
    if w >= costs[i] and v[w] == v[w - costs[i]] + rewards[i]:
      solution[i] += 1 # Include this item in the solution
      w -= costs[i] # Reduce the remaining capacity
      break # Break to restart the item selection

# Print the maximum value and the optimal solution
print(f"Maximum rewards: {v[capacity]}")
print(f"Optimal solution: {solution}")