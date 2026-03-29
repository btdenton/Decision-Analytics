#Chapter 10, Example 10.1: Stochastic production lotsizing
import math
import numpy as np
import matplotlib.pyplot as plt

#Input Data
T = 10      # decision epochs
M = 60      # inventory capacity and maximum demand fulfilled
lam = 10    # mean daily demand for Poissone distribution
r = 5.0     # revenue per unit sold
c_S = 20.0  # fixed setup cost
c_V = 1.0   # cost per unit ordered
c_H = 0.50  # holding cost per unit leftover inventory

# Basis function choice:"constant","linear","quadratic","cubic","quartic", "quintic"
BASIS = "cubic" 
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

#Return value function approximation
def vhat_from_w(w: np.ndarray, M: int): return lambda s: float(phi(int(s), M).dot(w))

# Demand model (truncated Poisson): returns probs for d=0,...,M-1 and P(D>=M)
def truncated_poisson_probs(lam: float, M: int):
    probs = np.zeros(M + 1, dtype=float)
    for d in range(M):
        probs[d] = math.exp(-lam) * (lam ** d) / math.factorial(d)
    probs[M] = max(0.0, 1.0 - probs[:M].sum())
    return probs

# Immediate reward: E_D[ r_t(s,a,D) + v_next(s')] using demand truncated at D=M
def expected_future_rewards(s: int,a: int,v_next_fn,demand_probs: np.ndarray,
    M: int,r: float,c_V: float,c_H: float):
    sa = s + a  # inventory after order
    total = 0.0
    def contrib(d: int, prob: float):
        sales = min(sa, d)
        leftover = max(0, sa - d)
        s_next = max(0, sa - d)  # lost sales => same as leftover
        reward = r * sales - (c_V * a + c_H * leftover)
        return prob * (reward + v_next_fn(s_next))
    for d in range(M): total += contrib(d,demand_probs[d]) # Exact mass for d=0..M-1
    total += contrib(M, demand_probs[M]) # Truncation correction for D=M
    return total

#Compute max value and argmax decision, for state s at epoch t
def bellman_iteration(t: int,s: int,v_next_fn,demand_probs: np.ndarray, M: int,
    r: float,c_V: float,c_H: float):
    best_val = -float("inf")
    best_decision = 0
    for a in range(0, M - s + 1):
        val = expected_future_rewards(s,a,v_next_fn,demand_probs,M,r,c_V,c_H)
        if val > best_val: best_val, best_decision = val, a
    return best_val, best_decision

# Value Function Approximation: compute value function samples, then least squares 
def value_function_approx(T: int,M: int,lam: float,r: float,c_V: float,c_H: float):
    demand_probs = truncated_poisson_probs(lam, M)
    m = len(phi(0, M))
    w = [np.zeros(m) for _ in range(T+1)] # Weights at t (w_T zeros because v_T(s)=0)
    pi = np.zeros((T, M + 1), dtype=int)
    for t in range(T - 1, -1, -1):
        v_next = vhat_from_w(w[t + 1], M)
        Phi = np.zeros((M + 1, m)) # Create regression data for all states 
        y = np.zeros(M + 1)
        for s in range(M + 1):
            Phi[s, :] = phi(s, M)
            y[s], pi[t,s] = bellman_iteration(t,s,v_next,demand_probs,M,r,c_V,c_H)
        w[t], *_ = np.linalg.lstsq(Phi, y, rcond=None) # Least squares fit
    return w, pi

# Exact DP for comparison to approximation
def solve_exact_dp(T: int,M: int,lam: float,r: float,c_V: float,c_H: float):
    demand_probs = truncated_poisson_probs(lam, M)
    v = np.zeros((T + 1, M + 1))
    pi = np.zeros((T, M + 1), dtype=int)
    v[T, :] = 0.0
    for t in range(T - 1, -1, -1):
        for s in range(M + 1):
            v[t, s], pi[t, s] = bellman_iteration(t,s,v_next_fn=lambda sp: v[t + 1, sp],
                demand_probs=demand_probs,M=M,r=r,c_V=c_V,c_H=c_H)
    return v, pi

# Solve approximate MDP
w_hat, pi_hat = value_function_approx(T, M, lam, r, c_V, c_H)
v0_hat_fn = vhat_from_w(w_hat[0], M)
s_grid = np.arange(M + 1)
v0_hat = np.array([v0_hat_fn(s) for s in s_grid])

# Solve exact MDP for comparison
v_exact, pi_exact = solve_exact_dp(T, M, lam, r, c_V, c_H)
v0_exact = v_exact[0, :]
rmse = float(np.sqrt(np.mean((v0_hat - v0_exact) ** 2))) #root mean square error
max_err = float(np.max(np.abs(v0_hat - v0_exact)))
print(f"BASIS = {BASIS}")
print(f"RMSE(v_0): {rmse:.4f}   max|err|: {max_err:.4f}")

plt.figure()
plt.plot(s_grid, v0_exact, label="Exact $v_0(s)$")
plt.plot(s_grid, v0_hat, label=f"Approx $\\hat v_0(s)$ ({BASIS} basis)")
plt.xlabel("State $s$ (inventory)")
plt.ylabel("Value $v_0(s)$")
plt.title("Exact vs Approximate Value Function")
plt.legend()
plt.tight_layout()
plt.show()


