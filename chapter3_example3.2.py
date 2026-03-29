#Chapter 3, Example 3.2: Production Lot-sizing
import numpy as np

# Input data
demand = [1, 3, 2, 4] # Monthly demand
c_s = 3 # Setup cost for each production run
c_v = 1 # Variable cost per unit produced
c_h = 0.5 # Holding cost per unit inventory per month

# Initialize variables
T = len(demand)
cost = np.zeros((T, T))
decision = np.zeros((T))
v = np.zeros(T + 1) # Optimal value function

# Precompute cost to meet demand from i to j periods ahead
for i in range(0, T):
  for j in range(i, T):
    cost[i, j] = c_s + c_v * sum(demand[i:j + 1])
    inventory_cost = 0
    for s in range(i + 1, j + 1):
      inventory_cost = inventory_cost + c_h * (s - i) * demand[s]
    cost[i, j] = cost[i, j] + inventory_cost

# Solve the optimality equations to optimal decision each period
for i in range(T-1, -1, -1):
  min_value = float('inf') 
  for j in range(i, T):
    v[i] = cost[i, j] + v[j + 1]
    if v[i] < min_value:
      min_value = v[i]
      optimal_decision = j
  v[i] = min_value
  decision[i] = optimal_decision

print("\nOptimal Production Schedule:")
print(f"\nTotal Cost: {min_value}")
period = 0
while period < T:
  periods_ahead = int(decision[period] - period)
  produce = sum(demand[period:period + periods_ahead + 1])
  if (periods_ahead == 0):
    print(f"Month {period}: Produce {produce} unit(s) for Month {period} demand")
  else:
    print(f"Month {period}: Produce {produce} unit(s) for Months {period} \
          to {period + periods_ahead} demand")
  if period < T: 
      period = period + periods_ahead + 1