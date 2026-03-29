#Chapter 9, Example 9.1: Robust medical treatment decisions
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Input data
T = 25 # Time horizon (50 (t-0) through 75 (t=25) years
nstates = 5
ndecisions = 2
states = ['L', 'M', 'H', 'V', 'E']  # Health states and adverse event (E)
decisions = ['C', 'Q']              # Decisions: Continue ('C'), Quit/Treat ('Q')
C_R = [0.1, 0.4, 2, 10, 0]          # Risk multipliers by state L, M, H, V, E
D_trt = 0.05                        # Annual disutility of treatment
rho = 0.2                           # Relative deviation for E probability
P = np.array([ # Base transition probabilities for 'C' decision, excluding E 
    [0.74, 0.20, 0.05, 0.01],  # From L
    [0.10, 0.70, 0.17, 0.03],  # From M
    [0.01, 0.24, 0.62, 0.13],  # From H
    [0.01, 0.06, 0.30, 0.63]   # From V
])

rewards = np.zeros((T + 1, nstates, ndecisions)) # Reward function: rewards[t,s,d] 
for t in range(T + 1):
    rewards[t, :, 0] = 1  # Decision 'C'
    rewards[t, :, 1] = -0.721*(t + 50) + 64.36 - (25 - t) * D_trt #Decision 'Q'
    rewards[t, -1, :] = 0  # E state reward is 0

# Midpoint of the uncertain P_t(E|s_t) interval
def p_e_mid(t, s): return C_R[s] * 0.0003 * np.exp(0.0891 * (t + 50))

# Uncertainty interval
def p_interval(t, s):
    mid = p_e_mid(t, s)
    lb = float(np.clip(mid * (1 - rho), 0.0, 1.0))
    ub = float(np.clip(mid * (1 + rho), 0.0, 1.0))
    return lb, ub

value = np.zeros((T + 1, nstates, ndecisions))  # Value-to-go function
v = np.zeros((T + 1, nstates))      # Optimal value function
optimal_policy = np.zeros((T + 1, nstates), dtype=int)  # Policy
v[T] = rewards[T, :, 1] # Boundary condition
optimal_policy[T, :-1] = 1  # decision 'Q'
optimal_policy[T, -1] = -1  # E: no decision

# Backward induction
for t in range(T - 1, -1, -1):
    for s in range(nstates):
        if s == nstates - 1:  #E state
            v[t, s] = 0
            optimal_policy[t, s] = -1
            continue
        lb, ub = p_interval(t, s)
        model = gp.Model() # Create a new Gurobi model
        model.Params.OutputFlag = 0
        p = model.addVar(lb=lb, ub=ub, name="p_e") # Adversary decision variables 
        obj = (1 - p) * P[s].dot(v[t + 1, :-1]) + \
            p * v[t + 1, -1] # Set objective function
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize() # Optimize model
        if model.status == GRB.OPTIMAL: min_value = model.ObjVal
        value[t, s, 0] = rewards[t, s, 0] + min_value   # Robust 'C'
        value[t, s, 1] = rewards[t, s, 1] # 'Q'
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