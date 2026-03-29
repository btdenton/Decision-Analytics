#Chapter 7, Example 7.3: POMDP for Search and rescue
import numpy as np
from itertools import product

# Input data
T = 4
nstates = 4
ndecisions = 4
nobservations = 4
states = ['S1', 'S2', 'S3', 'S4']
decisions = ['Up', 'Down', 'Left', 'Right']
observations = ['Obs_S1', 'Obs_S2', 'Obs_S3', 'Obs_S4']
N = 10 # Number of discrete points for each dimension of the belief vector
P = np.zeros((nstates,ndecisions,nstates)) # Core state transition probabilities 
P[:, 0, :] = np.array([[0.8, 0.1, 0.1, 0.0], # d = 'Up'
                         [0.1, 0.8, 0.0, 0.1],
                         [0.8, 0.0, 0.1, 0.1],
                         [0.0, 0.0, 0.0, 1.0]])
P[:, 1, :] = np.array([[0.1, 0.1, 0.8, 0.0], # d = 'Down'
                      [0.1, 0.1, 0.0, 0.8],
                      [0.1, 0.0, 0.8, 0.1],
                      [0.0, 0.0, 0.0, 1.0]])
P[:, 2, :] = np.array([[0.8, 0.1, 0.1, 0.0], # d = 'Left'
                      [0.8, 0.1, 0.0, 0.1],
                      [0.1, 0.0, 0.8, 0.1],
                      [0.0, 0.0, 0.0, 1.0]])
P[:, 3, :] = np.array([[0.1, 0.8, 0.1, 0.0], # d = 'Right'
                       [0.1, 0.8, 0.0, 0.1],
                       [0.1, 0.0, 0.1, 0.8],
                       [0.0, 0.0, 0.0, 1.0]])
# Noisy observation probabilities (rows: core state, columns: observation)
Q = np.array([[0.7, 0.1, 0.1, 0.1],
              [0.1, 0.7, 0.1, 0.1],
              [0.1, 0.1, 0.7, 0.1],
              [0.1, 0.1, 0.1, 0.7]])
# Rewards for each state transition ("-9999" denotes infeasible transitions)
rewards = np.array([[0, -1, -1, -9999],
                    [-1, 0, -9999, 10],
                    [-1, -9999, 0, 10],
                    [0, 0, 0, 10]])

# Function for belief update given decision and observation
def update_belief(belief, d, o):
    belief = np.dot(P[:, d, :].T, belief)
    belief = belief * Q[:, o]
    belief = belief / np.sum(belief) # Normalize
    return belief

# Function to find the closest grid point to current belief
def find_closest_belief(belief):
    distances = np.linalg.norm(belief_points - belief, axis=1)
    return belief_points[np.argmin(distances)]

# Function to compute the expected reward given belief
def compute_expected_reward(belief, d):
    total_reward = 0
    for s in range(nstates):
        for s_next in range(nstates):
            total_reward += belief[s] * P[s, d, s_next] * rewards[s, s_next]
    return total_reward

# Function to compute expected value to go given belief
def expected_value_to_go(belief, d, v, t):
    expected_value_to_go = 0
    for o in range(nobservations):
        prob_o = 0
        for s in range(nstates):
            for s_next in range(nstates):
                prob_o += belief[s] * P[s, d, s_next] * Q[s_next, o]
        if prob_o == 0:
            continue
        new_belief = update_belief(belief, d, o)
        closest_belief = find_closest_belief(new_belief)
        closest_belief_idx = np.where((belief_points == closest_belief).all(axis=1))[0][0]
        expected_value_to_go += prob_o * v[closest_belief_idx, t + 1]
    return expected_value_to_go

# Create a finite grid, removing combinations deviating from one by > 10^-8
belief_points = np.array([p for p in product(np.linspace(0,1,N),
                                   repeat=nstates) if np.isclose(sum(p), 1.0)])
v = np.zeros((len(belief_points), T + 2))
optimal_policy = [[None] * (T + 2) for _ in range(len(belief_points))]

# Backward induction to compute optimal value function
for t in reversed(range(T + 1)):
    for i, belief in enumerate(belief_points):
        v_values = []
        expected_value = 0
        for d in range(ndecisions):
            expected_value = compute_expected_reward(belief, d)
            expected_value += expected_value_to_go(belief, d, v, t)
            v_values.append(expected_value)
        v[i, t] = max(v_values)
        optimal_policy[i][t] = decisions[np.argmax(v_values)]

# Print results
print("Optimal Value Function:")
print(v)
print("Approximate Optimal Policy:")
for t in range(T + 1):
    print(f"Time {t}: {[optimal_policy[i][t] for i in range(len(belief_points))]}")