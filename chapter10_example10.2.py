#Chapter 10, Example 10.2: Linear programming based approximation
import math
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

# Input Data
M = 60      # inventory capacity and maximum demand fulfilled
lam = 10    # mean daily demand for Poisson distribution
r = 5.0     # revenue per unit sold
c_S = 20.0  # fixed setup cost (still unused, as in your original code)
c_V = 1.0   # cost per unit ordered
c_H = 0.50  # holding cost per unit leftover inventory
Lambda = 0.97 # discount factor

# Basis function choice:"constant","linear","quadratic","cubic","quartic", "quintic"
BASIS = "quintic"

# Basis functions
def phi(s: int, M: int):
    if BASIS == "constant":
        return np.array([1.0])
    if BASIS == "linear":
        return np.array([1.0, s])
    if BASIS == "quadratic":
        return np.array([1.0, s, s * s])
    if BASIS == "cubic":
        return np.array([1.0, s, s * s, pow(s,3)])
    if BASIS == "quartic":
        return np.array([1.0, s, s * s, pow(s,3), pow(s,4)])
    if BASIS == "quintic":
        return np.array([1.0, s, s * s, pow(s,3), pow(s,4), pow(s,5)])
    raise ValueError(f"Unknown BASIS: {BASIS}")

# Return value function approximation
def vhat_from_w(w: np.ndarray, M: int): return lambda s: float(phi(int(s), M).dot(w))

# Demand model (truncated Poisson): returns probs for d=0,...,M-1 and P(D>=M)
def truncated_poisson_probs(lam: float, M: int):
    probs = np.zeros(M + 1, dtype=float)
    for d in range(M):
        probs[d] = math.exp(-lam) * (lam ** d) / math.factorial(d)
    probs[M] = max(0.0, 1.0 - probs[:M].sum())
    return probs

# Expected reward and next-state under demand
def expected_future_rewards(s: int, a: int, v_fn, demand_probs: np.ndarray,
    M: int, r: float, c_V: float, c_H: float, Lambda: float):
    sa = s + a  # inventory after order
    total = 0.0
    def contrib(d: int, prob: float):
        sales = min(sa, d)
        leftover = max(0, sa - d)
        s_next = leftover  # lost sales => leftover inventory
        reward = r * sales - (c_V * a + c_H * leftover)
        return prob * (reward + Lambda * v_fn(s_next))
    for d in range(M):
        total += contrib(d, demand_probs[d])
    total += contrib(M, demand_probs[M])
    return total

# Compute max value and argmax decision, for state s 
def bellman_iteration(s: int, v_fn, demand_probs: np.ndarray, M: int,
    r: float, c_V: float, c_H: float, Lambda: float):
    best_val = -float("inf")
    best_decision = 0
    for a in range(0, M - s + 1):
        val = expected_future_rewards(s,a,v_fn,demand_probs,M,r,c_V,c_H,Lambda)
        if val > best_val: best_val, best_decision = val, a
    return best_val, best_decision

#Compute linear program constraint coefficients for the approximate LP
def expected_future_reward_features(s: int, a: int, demand_probs: np.ndarray, 
                                      M: int,r: float, c_V: float, c_H: float):
    sa = s + a
    E_reward = 0.0
    E_phi_next = np.zeros(len(phi(0, M)))
    def contrib(d: int, prob: float):
        sales = min(sa, d)
        leftover = max(0, sa - d)
        s_next = leftover
        reward = r * sales - (c_V * a + c_H * leftover)
        return prob, reward, s_next
    for d in range(M):
        prob, reward, s_next = contrib(d, demand_probs[d])
        E_reward += prob * reward
        E_phi_next += prob * phi(s_next, M)
    prob, reward, s_next = contrib(M, demand_probs[M])
    E_reward += prob * reward
    E_phi_next += prob * phi(s_next, M)
    return E_reward, E_phi_next

#Compute linear program constraint coefficients for exact LP
def expected_reward_and_next_state_probs(s: int, a: int, demand_probs: np.ndarray,
    M: int, r: float, c_V: float, c_H: float):
    sa = s + a
    E_reward = 0.0
    p_next = {}
    def add_mass(d: int, prob: float):
        nonlocal E_reward, p_next
        sales = min(sa, d)
        leftover = max(0, sa - d)
        s_next = leftover
        reward = r * sales - (c_V * a + c_H * leftover)
        E_reward += prob * reward
        p_next[s_next] = p_next.get(s_next, 0.0) + prob
    for d in range(M):
        add_mass(d, demand_probs[d])
    add_mass(M, demand_probs[M])  # truncation mass
    return E_reward, p_next

# Approximate Linear Program solve
def solve_approx_alp(M: int, lam: float, r: float, c_V: float, c_H: float, Lambda: float):
    demand_probs = truncated_poisson_probs(lam, M)
    m = len(phi(0, M))
    mu = np.ones(M + 1) / (M + 1) #Objective: minimize sum_s mu(s) * vhat(s)
    Phi = np.vstack([phi(s, M) for s in range(M + 1)])
    c = Phi.T @ mu  # c in R^m
    model = gp.Model("ALP") #Gurobi linear program formulation
    w = model.addMVar(shape=m, lb=-GRB.INFINITY, name="w") # Decision vars, w
    model.setObjective(c @ w, GRB.MINIMIZE)  # Objective: min c^T w

    # Constraints: for all (s,a),phi(s)^T w >= E[r] + Lambda * E[phi(s')]^T w
    for s in range(M + 1):
        phi_s = phi(s, M)
        for a in range(0, M - s + 1):
            E_reward, E_phi_next = expected_future_reward_features(
                s, a, demand_probs, M, r, c_V, c_H)
            model.addConstr((phi_s - Lambda * E_phi_next) @ w >= E_reward,
                            name=f"bellman_{s}_{a}")
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"ALP (Gurobi) failed. Status={model.Status}")
    w_sol = w.X  # numpy array
    v_fn = vhat_from_w(w_sol, M) # Greedy policy induced by vhat
    pi = np.zeros(M + 1, dtype=int)
    for s in range(M + 1):
        _, best_decision = bellman_iteration(s, v_fn, demand_probs, M, r, c_V, c_H, Lambda)
        pi[s] = best_decision
    return w_sol, pi

def solve_exact_lp(M: int, lam: float, r: float, c_V: float, c_H: float, Lambda: float):
    demand_probs = truncated_poisson_probs(lam, M)
    mu = np.ones(M + 1) / (M + 1)     # Weights for LP objective
    model = gp.Model("Exact_Discounted_MDP_LP")
    model.Params.OutputFlag = 0
    V = model.addVars(M + 1, lb=-GRB.INFINITY, name="V") # Variables: V[s]
    # Objective: minimize sum_s mu[s] * V[s]
    model.setObjective(gp.quicksum(mu[s] * V[s] for s in range(M + 1)), GRB.MINIMIZE)

    # Bellman inequalities as constraints: v(s) >= E[r(s,a)] + Lambda * E[v(s')]
    for s in range(M + 1):
        for a in range(0, M - s + 1):
            E_reward, p_next = expected_reward_and_next_state_probs(
                s, a, demand_probs, M, r, c_V, c_H)
            rhs = E_reward + Lambda*gp.quicksum(prob*V[sp] for sp,prob in p_next.items())
            model.addConstr(V[s] >= rhs, name=f"bellman_{s}_{a}")
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Exact LP (Gurobi) failed. Status={model.Status}")
    v = np.array([V[s].X for s in range(M + 1)], dtype=float)
    pi = np.zeros(M + 1, dtype=int)
    for s in range(M + 1):
        _, best_decision = bellman_iteration(s, v_fn=lambda sp: v[sp],demand_probs=
                        demand_probs, M=M, r=r,c_V=c_V, c_H=c_H, Lambda=Lambda)
        pi[s] = best_decision
    return v, pi

# Solve approximate infinite horizon ALP
w_hat, pi_hat = solve_approx_alp(M, lam, r, c_V, c_H, Lambda)
v_hat_fn = vhat_from_w(w_hat, M)
s_grid = np.arange(M + 1)
v_hat = np.array([v_hat_fn(s) for s in s_grid])

# Solve exact infinite-horizon LP for comparison
v_exact, pi_exact = solve_exact_lp(M, lam, r, c_V, c_H, Lambda)
rmse = float(np.sqrt(np.mean((v_hat - v_exact) ** 2)))
max_err = float(np.max(np.abs(v_hat - v_exact)))

print(f"BASIS = {BASIS}   Lambda = {Lambda}")
print(f"RMSE(v): {rmse:.4f}   max|err|: {max_err:.4f}")

plt.figure()
plt.plot(s_grid, v_exact, label="Exact $v(s)$")
plt.plot(s_grid, v_hat, label=f"Approx $\\hat v(s)$ (ALP, {BASIS} basis)")
plt.xlabel("State $s$ (inventory)")
plt.ylabel("Value $v(s)$")
plt.title("Infinite-Horizon Discounted: Exact vs Approx (ALP)")
plt.legend()
plt.tight_layout()
plt.show()
