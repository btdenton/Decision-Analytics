#Chapter 9, Example 9.3: Searching for a hiker in a Canadian national park
from collections import defaultdict
import numpy as np

# Input Data
N = 3  # dimension of grid with N x N sectors
Lambda = 0.9
FIND_REWARD = 10.0
START_P1 = (0, 0)       #Ranger 1 location
START_P2 = (N-1, N-1)   #Ranger 2 location
TERMINAL = ("T",)       # Terminal state marker
ALL_SECTORS = set(range(N * N))
TOLERANCE, MAX_ITERATIONS =1e-6,1000 #Value iteration tolerance, max iterations
MAX_STEPS = 200
MAX_REACHABLE_STATES = 10**6
np.random.seed(42) #Fix seed for repeatability

#Utility functions
def rc_to_idx(r, c): return r * N + c #convert 2D to 1D index
def idx_to_rc(i):    return (i // N, i % N)# convert 1D to 2D index

#Check feasible decisions given moving off grid is not permitted
def neighbors(idx):
    r, c = idx_to_rc(idx)
    options = []
    if r > 0:        options.append(('U', rc_to_idx(r-1, c)))
    if r < N-1:      options.append(('D', rc_to_idx(r+1, c)))
    if c > 0:        options.append(('L', rc_to_idx(r, c-1)))
    if c < N-1:      options.append(('R', rc_to_idx(r, c+1)))
    options.append(('S', rc_to_idx(r, c)))
    return options  # list of (decision, next_idx)

#Cartesian product of feasible single agent decisions
def joint_decisions(p1, p2):
    for a1, np1 in neighbors(p1):
        for a2, np2 in neighbors(p2):
            yield (a1, a2, np1, np2)

#Rangers search their initial sectors without movement cost
def initial_search():
    p1 = rc_to_idx(*START_P1)
    p2 = rc_to_idx(*START_P2)
    visited = frozenset({p1, p2}) 
    return (p1, p2, visited)

#State transition dynamics given current state and join decisions
def step_transition(state, joint_dec):
    p1, p2, visited = state
    a1, a2, np1, np2 = joint_dec
    before_unvisited = set(ALL_SECTORS) - set(visited)
    if not before_unvisited:
        return [(1.0, TERMINAL, 0.0)]
    move_cost = int(np1 != p1) + int(np2 != p2)
    new_sectors = {np1, np2}
    newly_new = [c for c in new_sectors if c in before_unvisited]
    p_success = len(newly_new) / len(before_unvisited)
    r_succ = FIND_REWARD - move_cost
    next_visited = frozenset(set(visited) | new_sectors)
    r_fail = - move_cost
    next_state_fail = (np1, np2, next_visited)
    out = []
    if p_success > 0:
        out.append((p_success, TERMINAL, r_succ))
    out.append((1.0 - p_success, next_state_fail, r_fail))
    return out

#Identify all reachable states
def enumerate_reachable():
    start = initial_search()
    frontier = [start]
    seen = {start}
    while frontier:
        s = frontier.pop()
        p1, p2, visited = s
        for jd in joint_decisions(p1, p2):
            for prob, ns, _ in step_transition(s, jd):
                if ns != TERMINAL and ns not in seen:
                    seen.add(ns)
                    frontier.append(ns)
        if len(seen) == MAX_REACHABLE_STATES: 
            print("Hit max reachable states limit")
            break
    return seen

def simulate(policy, MAX_STEPS): #Simulate policy
    start1 = rc_to_idx(*START_P1) #Ranger 1 start sector
    start2 = rc_to_idx(*START_P2) #Ranger 2 start sector
    start_sectors = {start1, start2}
    P0 = len(start_sectors) / (N*N) 
    print("\n--- Simulation of policy ---")
    if np.random.random() <= P0:
        print(f"Found at rangers' start points: reward = {FIND_REWARD}")
        return
    s = initial_search() # Otherwise, continue search
    t, G = 0, 0.0
    while s != TERMINAL and t < MAX_STEPS:
        p1, p2, _ = s
        jd = policy[s]
        a1, a2, _, _ = jd
        print(f"t={t:2d}  pos1={idx_to_rc(p1)} pos2={idx_to_rc(p2)} \
              decision=(R1 {a1}, R2 {a2})")
        outcomes = step_transition(s, jd) # Sample next outcome
        u, acc = np.random.random(), 0.0
        prob, ns, r = outcomes[-1]  # fallback
        for p, nst, rew in outcomes:
            acc += p
            if u <= acc:
                ns, r = nst, rew
                break
        print(f"    reward={r:+.1f}\n")
        G += (Lambda**t) * r
        s, t = ns, t + 1
        if s == TERMINAL:
            print(f"t={t:2d}  FOUND!  discounted return = {G:.3f}")
            break
    if s != TERMINAL:
        print(f"Stopped at t={MAX_STEPS}. discounted return = {G:.3f}")

# Value iteration to solve for near-optimal policy
states = enumerate_reachable()  #Enumerate reachable states
print(f"Reachable non-terminal states: {len(states)}")
v = defaultdict(float) #Optimal value function estimate
policy = {}
for it in range(MAX_ITERATIONS):
    delta = 0.0
    for s in states:
        p1, p2, _ = s
        best_q, best_jd = float('-inf'), None
        for jd in joint_decisions(p1, p2):
            q = 0.0
            for prob, ns, r in step_transition(s, jd):
                q += prob * (r + Lambda * (0.0 if ns == TERMINAL else v[ns]))
            if q > best_q:
                best_q, best_jd  = q, jd
        delta = max(delta, abs(best_q - v[s]))
        v[s], policy[s] = best_q, best_jd
    if delta < TOLERANCE:
        print(f"Converged in {it} iterations (delta={delta:.3e}).")
        break
#Simulate to illustrate application of the optimal policy
simulate(policy, MAX_STEPS)
