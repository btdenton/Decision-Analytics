#Chapter 9, Example 9.2: Centralized planning for agents in automated warehouse
import numpy as np

#Input data
GRID_SIZE = 5
DECISIONS = ['L', 'R', 'U', 'D', 'N']
Lambda = 0.9
X_rc = (2, 2) #target X location (row, columns)
Y_rc = (3, 1) #target Y location (row, column)
def rc_to_idx(r, c): #Helper function: convert (row, column) to index
    r = max(1, min(GRID_SIZE, r))
    c = max(1, min(GRID_SIZE, c))
    return (r - 1) * GRID_SIZE + c
def idx_to_rc(idx): #Helper function: convet index to (row, column)
    idx = max(1, min(GRID_SIZE * GRID_SIZE, idx))
    r = (idx - 1) // GRID_SIZE + 1
    c = (idx - 1) % GRID_SIZE + 1
    return r, c
X_idx = rc_to_idx(X_rc[0], X_rc[1])
Y_idx = rc_to_idx(Y_rc[0], Y_rc[1])
S_single = list(range(1, GRID_SIZE*GRID_SIZE + 1)) # Joint states and decisions
STATES = [(s1, s2, hX, hY) for s1 in S_single for s2 \
          in S_single for hX in (0,1) for hY in (0,1)]
JOINT_DECISIONS = [(a1, a2) for a1 in DECISIONS for a2 in DECISIONS]
def is_terminal(s): return s[2] == 1 and s[3] == 1 #Check if target reached
NONTERMINAL_STATES = [s for s in STATES if not is_terminal(s)]
STATE_TO_INDEX = {s: i for i, s in enumerate(NONTERMINAL_STATES)}
MAX_STEPS=1000 #Upper bound on state transitions

#Robot movement and reward functions
def step(cell_idx, a):
    r, c = idx_to_rc(cell_idx)
    if a == 'L': c -= 1
    elif a == 'R': c += 1
    elif a == 'U': r -= 1
    elif a == 'D': r += 1
    elif a == 'N': pass
    else: raise ValueError(a)
    return rc_to_idx(r, c)

def move_cost(a): return 0 if a == 'N' else -1

def reward_on_arrival(hX, hY, s1p, s2p):
    hitX = int((s1p == X_idx) or (s2p == X_idx))
    hitY = int((s1p == Y_idx) or (s2p == Y_idx))
    return 10 * (1 - hX) * hitX + 10 * (1 - hY) * hitY

def successor(state, joint_decision): #Get next joint state
    s1, s2, hX, hY = state
    d1, d2 = joint_decision
    s1p = step(s1, d1)
    s2p = step(s2, d2)
    r_move = move_cost(d1) + move_cost(d2)
    r_arrival = reward_on_arrival(hX, hY, s1p, s2p)
    r = r_move + r_arrival
    hXp = 1 if (hX or (s1p == X_idx) or (s2p == X_idx)) else 0
    hYp = 1 if (hY or (s1p == Y_idx) or (s2p == Y_idx)) else 0
    return (s1p, s2p, hXp, hYp), r

def policy_evaluation(policy): 
    n = len(NONTERMINAL_STATES)
    A = np.eye(n)
    b = np.zeros(n)
    for s in NONTERMINAL_STATES:
        i = STATE_TO_INDEX[s]
        a = policy[s]
        sp, r = successor(s, a)
        b[i] = r
        if not is_terminal(sp): #Deterministic transition under policy
            j = STATE_TO_INDEX[sp]
            A[i, j] -= Lambda  #Move Lambda*v(sp) to LHS
    v_vec = np.linalg.solve(A, b)
    values = {s: 0.0 for s in STATES} #Initialize value to go
    for s in NONTERMINAL_STATES:
        values[s] = float(v_vec[STATE_TO_INDEX[s]])
    return values

#Policy iteration
def policy_iteration():
    policy = {s: ('N','N') for s in STATES}
    stable = False #Continue iterations until policy does not change
    while not stable:
        v = policy_evaluation(policy)
        stable = True   #Policy improvement steps
        for s in STATES:
            if is_terminal(s): continue
            a_curr = policy[s]   #Current policy value
            sp_curr, r_curr = successor(s, a_curr)
            best_val = r_curr + Lambda * v[sp_curr]
            best_a = a_curr
            for a in JOINT_DECISIONS: #Find the best decision
                sp, r = successor(s, a)
                q = r + Lambda * v[sp]
                if q > best_val + 1e-12: best_val, best_a = q, a
            if best_a != a_curr:
                policy[s] = best_a
                stable = False
    return v, policy

#Output:'.' empty;debris location:'x'/'y';debris cleared:X/Y;'1,2' robots
def print_grid(s1, s2, hX, hY):
    grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] 
    rX, cX = X_rc
    rY, cY = Y_rc
    grid[rX-1][cX-1] = 'X' if hX else 'x'
    grid[rY-1][cY-1] = 'Y' if hY else 'y'
    r1, c1 = idx_to_rc(s1)
    r2, c2 = idx_to_rc(s2)
    grid[r1-1][c1-1] = '1'
    grid[r2-1][c2-1] = '2' if (r2, c2) != (r1, c1) else 'B'  #'B' if robots stacked
    header = "    " + " ".join(f"{c:>2}" for c in range(1, GRID_SIZE+1))
    print(header)
    for r in range(GRID_SIZE):
        row_str = " ".join(f"{cell:>2}" for cell in grid[r])
        print(f"{r+1:>2} | {row_str}")

def simulate(policy, start_state, max_steps=MAX_STEPS):
    s = start_state
    sample_path = []
    print_grid(*s)
    for t in range(max_steps):
        if is_terminal(s): break
        a = policy[s]
        sp, r = successor(s, a)
        sample_path.append((t, s, a, r, sp))
        print(f"t={t}, DECISION={a}, reward={r}, next={sp}")
        print_grid(*sp)
        s = sp
    return sample_path

# Execute policy iteration
print("Running policy iteration...")
print("Legend: '.' empty, x/y sites (uncleared), X/Y sites (cleared),\n \
1/2 robots, B robots in same cell.\n")
print("Start State:")
v, policy = policy_iteration()
start = (rc_to_idx(1, 1), rc_to_idx(5, 5), 0, 0) #Robot start locations
sample_path = simulate(policy, start_state=start, max_steps=MAX_STEPS)
