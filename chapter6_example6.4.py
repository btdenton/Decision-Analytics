#Chapter 6, Example 6.4: Probabilistic sensitivity analysis of an MDP for machine maintenance
import numpy as np
import matplotlib.pyplot as plt

# Input data
T = 3
nstates = 4
ndecisions = 2
states = ['Excellent', 'Good', 'Damaged', 'Failed']
decisions = ['R', 'W']
q = 0.9
rho = 1
c = 0.5
mu = [20, 10, 5, 1] # Terminal rewards for Excellent, Good, Damaged, Failed
nsamples = 1000
# Transition probability matrix (4, 2, 4):  T[s, d, s'] = P(s' | s, d)
P = np.zeros((nstates, ndecisions, nstates))
P[:, 0, :] = np.array([ # d = 'R'
    [1.0, 0.0, 0.0, 0.0],
    [q, 1 - q, 0.0, 0.0],
    [q, 0.0, 1 - q, 0.0],
    [q, 0.0, 0.0, 1 - q]
])
P[:, 1, :] = np.array([ # d = 'W'
    [0.7, 0.3, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.4, 0.6],
    [0.0, 0.0, 0.0, 1.0]
])

rewards = np.zeros((nstates, ndecisions))
for s in range(nstates):
    rewards[s, 0] = -c # r(s, 'R')
    rewards[s, 1] = -np.exp(- rho * (s + 1)) # r(s, 'W')
sample = np.random.beta(1,1,size=nsamples) # Generate samles for sensitivity analysis
v_sample = np.zeros((nsamples, nstates), dtype = float)

# Perform Monte Carlo simulations for sensitivity analysis
np.random.seed(42) #Fix seed for repeatability
for k in range(nsamples):
    value = np.zeros((T, nstates, ndecisions))  #Value-to-go function
    v = np.zeros((T + 1, nstates))  #Optimal value function
    optimal_policy = np.zeros((T, nstates), dtype=int)  # Policy
    v[T, :] = mu    # Boundary condition at epoch T
    P_W_sample = np.copy(P[:, 1, :]) #Sample rows of TPM[:, 1, :] matrix
    for i in range(nstates - 1):      # Generate k^th sample of P_W
        a = 0.8 * P[i, 1, i]
        b = 1.2 * P[i, 1, i]
        P_W_sample[i, i] = a + (b - a) * sample[k]
        P_W_sample[i, i + 1] = 1 - P_W_sample[i, i]
    P_W_sample[nstates - 1, nstates - 1] = 1 # State 'Failed' absorbing state
    for t in range(T - 1, -1, -1):  # Backward induction for current sample
        for s in range(nstates):
            value[t,s,0] = rewards[s,0] + np.dot(P[s,0,:],v[t+1,:])
            value[t,s,1] = rewards[s,1] + np.dot(P_W_sample[s],v[t+1,:])
            optimal_policy[t, s] = np.argmax(value[t, s])
            v[t, s] = value[t, s, optimal_policy[t, s]]
    v_sample[k, :] = v[0, :]

# Show a histogram of the distribution of v_1(’Excellent’)
v_sample_0 = v_sample[:, 0] #Extract all ’Excellent’ state samples
plt.hist(v_sample_0, bins=10, edgecolor='black')
# Add titles and labels
plt.title("Histogram of Values for 'Excellent' State")
plt.xlabel("Value range")
plt.ylabel("Frequency")
plt.show()
