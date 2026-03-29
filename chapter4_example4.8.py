#Chapter 4, Example 4.8: Convergence from finite to infinite horizon MDP
import numpy as np

# Input data
nstates = 2
ndecisions = 2
states = ['F', 'E']     #States: Full (F) and Empty (E)
decisions = ['C', 'D']  #Decisions: Charge (C) and Defer (D)
T = 300                 #Time horizon
Lambda = 0.9            #Discount factor

# Transition probability matrix (2, 2, 2):  T[s, d, s'] = P(s' | s, d)
P = np.array([
    [[1.0, 0.0], [0.8, 0.2]],  # From F: [C, D]
    [[1.0, 0.0], [0.0, 1.0]],  # From E: [C, D]
])

# Reward function (2, 2): rewards[s, d] = R(s, d)
rewards = np.array([
    [-1, 0],   # F: [C, D]
    [-2, -3],  # E: [C, D]
])

# Initialize
value = np.zeros((T + 1, nstates, ndecisions))  #Value-to-go: value[t, s, d]
v = np.zeros((T + 1, nstates)) #Optimal value function
optimal_policy = np.zeros((T + 1, nstates), dtype=int) # Optimal policy

# Solve MDP to find optimal policy
for t in range(T - 1, -1, -1):
    for s in range(nstates):
        for d in range(ndecisions):
            value[t, s, d] = rewards[s, d] + Lambda * np.dot(P[s, d], v[t + 1])
        optimal_policy[t, s] = np.argmax(value[t, s])
        v[t, s] = value[t, s, optimal_policy[t, s]]

# Print the results for first epoch
print("Optimal Value Function at epoch 1:")
for s in range(nstates):
    best_d = optimal_policy[1, s]
    print(f"  State {states[s]}:  {value[1, s, optimal_policy[1, s]]}")

print("\nOptimal Policy:")
for s in range(nstates):
    print(f"  State {states[s]}: optimal policy = {decisions[optimal_policy[1, s]]}")