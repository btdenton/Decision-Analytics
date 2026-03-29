#Chapter 9, Example 9.4: Factory robot coordination problem
import numpy as np
import itertools
from functools import lru_cache

# Input data
T = 3
states = [0, 1]
nstates = 2
decisions = ['wait', 'process']
ndecisions = 2
nobservations = 2
Lambda = 0.95 # Discount factor

# Transition probability matrix: TPM[s, d1, d2, s_next]
P = np.zeros((nstates, ndecisions, ndecisions, nstates), dtype=float)
for s in range(nstates):
    for d1 in range(ndecisions):
        for d2 in range(ndecisions):

            # Higher probability of next task if both process at s=1, or wait at s=0
            if (s == 1 and d1 == 1 and d2 == 1) or (s == 0 and d1 == 0 and d2 == 0):
                p_next_task = 0.9

            # Otherwise lower probability
            else: p_next_task = 0.5
            P[s, d1, d2, 1] = p_next_task
            P[s, d1, d2, 0] = 1 - p_next_task

O = np.array([ # Observation probabilities: O[s, o] = P(o|s)
    [0.9, 0.1],  # P(o=0|s=0)=0.9, P(o=1|s=0)=0.1
    [0.1, 0.9]   # P(o=0|s=1)=0.1, P(o=1|s=1)=0.9
], dtype=float)

rewards = np.zeros((nstates,ndecisions,ndecisions),dtype=float) # rewards[s,d1,d2]

# s = 1
rewards[1, 1, 1] = 15.0    # Both process when a task exists
rewards[1, 1, 0] = -12.0   # Exactly one processes
rewards[1, 0, 1] = -12.0   # Exactly one processes
rewards[1, 0, 0] = -8.0    # Both wait
# s = 0
rewards[0, 1, 1] = -10.0   # Both process unnecessarily
rewards[0, 1, 0] = -8.0    # One processes unnecessarily
rewards[0, 0, 1] = -8.0    # One processes unnecessarily
rewards[0, 0, 0] = 6.0     # Both wait

# Initial belief over states
initial_belief = [0.5, 0.5]

# Policy-tree representation
def leaf(d): return ('leaf', d) # take decision d and stop
def node(d, c0, c1): 
    return ('node', d, c0, c1) # take decision d then follow subtree c0 or c1

# Generate depth-tau trees from depth-(tau-1) trees
def exhaustive_backup(Q_prev):
    result = []
    for d in range(ndecisions):
        for child_o0, child_o1 in itertools.product(Q_prev, repeat=2):
            result.append(node(d, child_o0, child_o1))
    return result

# Compute the expected value of joint policy starting state s with  depth_h
@lru_cache(maxsize=None) #Forces caching for faster performance
def value_state(s, pi1, pi2, depth_h):
    if depth_h == 0: return 0
    d1 = pi1[1]
    d2 = pi2[1]
    if depth_h == 1: return rewards[s, d1, d2]
    # Future value
    value = 0
    for s_next in range(nstates):
        for o1 in range(nobservations):
            for o2 in range(nobservations):
                c1 = pi1 if pi1[0] == 'leaf' else (pi1[2] if o1 == 0 else pi1[3])
                c2 = pi2 if pi2[0] == 'leaf' else (pi2[2] if o2 == 0 else pi2[3])
                value += (P[s, d1, d2, s_next] * O[s_next, o1] * O[s_next, o2]
                           * value_state(s_next, c1, c2, depth_h - 1))
    return rewards[s, d1, d2] + Lambda * value

# Compute expected value of a joint policy (pi1, pi2) under initial belief
def value_belief(belief, pi1, pi2, depth_h):
    return float(belief[0] * value_state(0, pi1, pi2, depth_h) +
                 belief[1] * value_state(1, pi1, pi2, depth_h))

# Evaluate pairs (pi1, pi2) on belief points, prune policies never near-optimal
def prune(Q1, Q2, depth_h, grid_pts=41, eps=1e-6, topk=None):
    beliefs = np.linspace(0.0, 1.0, grid_pts)
    keep_pairs = set()
    for p in beliefs:
        b = [1 - p, p]
        vals = [(value_belief(b, pi1, pi2, depth_h), i, j) # Evaluate policies
                for i, pi1 in enumerate(Q1) for j, pi2 in enumerate(Q2)]
        vals.sort(key=lambda x: x[0], reverse=True)
        if topk is None:
            best = vals[0][0]
            winners = [(i, j) for v, i, j in vals if v >= best - eps]
        else:
            cutoff = vals[min(topk - 1, len(vals) - 1)][0]
            winners = [(i, j) for v, i, j in vals if v >= cutoff - eps]
        keep_pairs.update(winners)
    keep_i = {i for (i, _) in keep_pairs}
    keep_j = {j for (_, j) in keep_pairs}
    Q1_new = [pi for i, pi in enumerate(Q1) if i in keep_i]
    Q2_new = [pj for j, pj in enumerate(Q2) if j in keep_j]
    return Q1_new, Q2_new

# Show policy tree
def show_policy(pi, indent=0):
    pad = '  ' * indent
    if pi[0] == 'leaf':
        return f"{pad} d = {decisions[pi[1]]}"
    s = f"{pad}d = {decisions[pi[1]]}\n"
    s += f"{pad} if o = 0:\n{show_policy(pi[2], indent+2)}\n"
    s += f"{pad} if o = 1:\n{show_policy(pi[3], indent+2)}"
    return s

# Solve Dec-POMDPs
Q1 = [leaf(d) for d in range(ndecisions)] #Initialize policies
Q2 = [leaf(d) for d in range(ndecisions)]

for tau in range(2, T + 2): # Grow tree to depth T + 1 with pruning
    Q1 = exhaustive_backup(Q1)
    Q2 = exhaustive_backup(Q2)
    changed = True  #Iterative pruning until no change
    while changed:
        before = (len(Q1), len(Q2))
        Q1, Q2 = prune(Q1, Q2, tau)
        changed = before != (len(Q1), len(Q2))

# Loop over policy trees, get best value and policy tree
best_pair = None
best_value = -np.inf
for pi1, pi2 in itertools.product(Q1, Q2):
    v = value_belief(initial_belief, pi1, pi2, T + 1)
    if v > best_value:
        best_value = v
        best_pair = (pi1, pi2)

# Print results
print(f"Optimal value (T = {T}, initial belief = {initial_belief}): {best_value:.4f}\n")
print('Robot 1 policy tree:')
print(show_policy(best_pair[0]))
print('Robot 2 policy tree:')
print(show_policy(best_pair[1]))
print(f"\nAfter pruning at depth T + 1 = {T + 1}: |Q1^(T+1)| = {len(Q1)}, \
|Q2^(T+1)| = {len(Q2)}")