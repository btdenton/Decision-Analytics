#Chapter 4, Example 4.11: Machine maintenance
import numpy as np

#Input data
states = ["Excellent", "Good", "Fair", "Poor", "Broken"]
decisions = ["Run", "Fix", "New"]
Lambda = 0.99  # discount factor 
profit = 5     # reward in non-Broken states
loss = -10     # reward in Broken state
cost = {"Run": 0, "Fix": 3, "New": 9}
MAX_ITS = 100 #max allowable iterations of policy iteration
# Transition probability matrices P(a) 
P = {
    "Run": np.array([
        [0.7, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]),
    "New": np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]),
    "Fix": np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.6, 0.3, 0.1, 0.0, 0.0],
        [0.0, 0.6, 0.3, 0.1, 0.0],
        [0.0, 0.0, 0.6, 0.3, 0.1],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]),
}
# Initial policy 
policy_0 = ["New", "New", "New", "New", "New"]

def build_reward_matrix(states, decisions, profit, loss, cost): 
    r = np.zeros((len(states), len(decisions)))
    for i, s in enumerate(states):
        base = profit if s != "Broken" else loss
        for j, a in enumerate(decisions):
            r[i, j] = base - cost[a]
    return r

def policy_evaluation(policy, P, r, Lambda, decisions):
    nS = len(policy)
    P_pi = np.zeros((nS, nS))
    r_pi = np.zeros(nS)
    decision_to_index = {a: j for j, a in enumerate(decisions)}
    for i, a in enumerate(policy):
        P_pi[i, :] = P[a][i, :]
        r_pi[i] = r[i, decision_to_index[a]]
    A = np.eye(nS) - Lambda * P_pi
    v = np.linalg.solve(A, r_pi)
    return v

def policy_improvement(policy, v, P, r, Lambda, states, decisions, tol=1e-12):
    decision_to_index = {a: j for j, a in enumerate(decisions)}
    policy_new = []
    stop = True  # assume no change unless one is found
    for i, s in enumerate(states):
        value = []
        for a in decisions:
            q = r[i, decision_to_index[a]] + Lambda * P[a][i, :].dot(v)
            value.append(q)
        value = np.array(value)
        best_policy = decisions[int(np.argmax(value))]
        policy_new.append(best_policy)
        if best_policy != policy[i]:   # if no decision changes stop
            stop = False
    return policy_new, stop

def policy_iteration(states,decisions,P,r,Lambda,policy_0,max_iters=MAX_ITS):
    policy = list(policy_0)
    for n in range(max_iters):
        v = policy_evaluation(policy, P, r, Lambda, decisions)
        policy_new,Stop = policy_improvement(policy,v,P,r,Lambda,states,decisions)
        if Stop:
            print(f"Optimal Policy: {policy}")
            return policy, v
        else:
            print(f"Iteration {n}: policy = {policy}")
        policy = policy_new
    print("Policy iteration did not converge within max_iters.")

#Solve infinite horizon MDP with policy iteration
r = build_reward_matrix(states, decisions, profit, loss, cost)
policy_optimal,v=policy_iteration(states,decisions,P,r,Lambda,policy_0)

