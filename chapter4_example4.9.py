#Chapter 4, Example 4.9: Value iteration
import numpy as np

# Input data
nstates = 2
ndecisions = 2
TPM = np.array([
    [[0.8, 0.2], [1.0, 0.0]],
    [[0.0, 1.0], [1.0, 0.0]]
])
rewards = np.array([
    [0, -1],
    [-3, -2]
])
Lambda = 0.9
epsilon = 0.001

#Initialize variables
v = np.zeros(nstates) #optimal value function
n = 0 # Iteration index for value iteration

# Perform value iteration for an infinite horizon MDP.
while True:
    new_v = np.zeros(nstates)
    value = np.zeros((nstates, ndecisions)) 
    for s in range(nstates):
        for d in range(ndecisions):
            value[s, d] = rewards[s, d] + Lambda * np.dot(TPM[s, d, :], \
                                                          v)
        new_v[s] = np.max(value[s, :]) #Choose optimal decision
    if np.max(np.abs(new_v-v)) \
        < epsilon*(1-Lambda)/(2*Lambda):
        break
    v = new_v
    n += 1

# Get epsilon optimal policy from the value function
optimal_policy = np.zeros(nstates, dtype=int) 

for s in range(nstates):
    for d in range(ndecisions):
      value[s, d] = rewards[s, d] + Lambda * np.dot(TPM[s, d, :], v)
    optimal_policy[s] = np.argmax(value[s, :])

print("Number of iterations:", n)
print("Optimal Policy:", optimal_policy)
print("Optimal Value Function:", v)
print("Error Tolerance:", epsilon)
