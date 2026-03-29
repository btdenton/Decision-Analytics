#Chapter 4, Example 4.4: Optimal time to treat
import numpy as np

# Input data
T = 25 # Time horizon (age 50 through 75 years, T=25 boundary condition
nstates = 5
ndecisions = 2
states = ['L', 'M', 'H', 'V', 'E'] # Health states and adverse event (E)
decisions = ['C', 'Q']            # Decisions: Continue ('C'), Quit/Treat ('Q')
c_R = [0.1, 0.4, 2, 10, 0]        # Risk multipliers by state L, M, H, V, E
d_T = 0.05                      # Annual disutility of treatment

# Transition probabilities for decision 'C', excluding E state for now
P_C = np.array([
    [0.74, 0.20, 0.05, 0.01],  # From L
    [0.10, 0.70, 0.17, 0.03],  # From M
    [0.01, 0.24, 0.62, 0.13],  # From H
    [0.01, 0.06, 0.30, 0.63]   # From V
])

# Transition probability matrix: TPM[t, s, d, s'] = P(s' | s, d) at time t
P = np.zeros((T + 1, nstates, ndecisions, nstates))
for t in range(T + 1):
    for s in range(nstates - 1):  # Exclude E
        p_e = c_R[s] * 0.0003 * np.exp(0.0891 * (t + 50))
        P[t, s, 0, :-1] = P_C[s] * (1 - p_e)
        P[t, s, 0, -1] = p_e
    P[t, -1, :, -1] = 1.0  # E is absorbing
    P[t, :, 1, :] = 0.0

# Reward function: rewards[t, s, d] 
rewards = np.zeros((T + 1, nstates, ndecisions))
for t in range(T + 1):
    rewards[t, :, 0] = 1  # Decision 'C'
    rewards[t, :, 1] = -0.721 * (t + 50) + 64.36 - (25 - t) * d_T #Decision 'Q'
    rewards[t, -1, :] = 0  # E state reward is 0

# Initialize
value = np.zeros((T + 1, nstates, ndecisions))  # Value-to-go function
v = np.zeros((T + 1, nstates))      # Optimal value function
optimal_policy = np.zeros((T + 1, nstates), dtype=int)  # Policy

# Boundary condition
v[T] = rewards[T, :, 1]
optimal_policy[T, :-1] = 1  # decision 'Q'
optimal_policy[T, -1] = -1  # E: no decision

# Backward induction
for t in range(T - 1, -1, -1):
    for s in range(nstates):
        if s == nstates - 1:  # E state
            v[t, s] = 0
            optimal_policy[t, s] = -1
            continue
        for d in range(ndecisions):
            value[t, s, d] = rewards[t, s, d] + np.dot(P[t, s, d], v[t + 1])
        optimal_policy[t, s] = np.argmax(value[t, s, :])
        v[t, s] = value[t, s, optimal_policy[t, s]]

# Print results
print("Optimal Policy:")
for t in range(T + 1):
    print(f"Optimal policy at age {t + 50}:")
    for s in range(nstates):
        if optimal_policy[t, s] == -1:
            decision = '-' # For E state
        else:
            decision = decisions[optimal_policy[t, s]]
        print(f"  State {states[s]}: {decision}")