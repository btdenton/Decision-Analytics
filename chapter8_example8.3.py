#Chapter 8, Example 8.3: Monte Carlo policy iteration for grid world
import numpy as np
import random
from collections import defaultdict

# Input data
grid_size = 4  # 4x4 grid
nstates = 16
states = range(nstates)
ndecisions = 4
decisions = ['up', 'down', 'left', 'right']
Lambda = 0.95       # Discount factor
start_state = 0     #top left corner; states indexed left to right, top down
goal_state = 15     # index from 0; 16 states total
penalty_state = 9   # treated as barrier/wall
rewards = {goal_state: 10, penalty_state: -5}  # Goal reward and barrier penalty
default_reward = -0.5    # Reward for non-terminal, non-barrier states
max_iterations = 100    # Max iterations of Monte Carlo Policy Iteration.
NUM_EPISODES = 1000
movements = {   #Define decisions
    0: -grid_size,  # 'up'    (row - 1)
    1: grid_size,   # 'down'  (row + 1)
    2: -1,          # 'left'  (col - 1)
    3: 1            # 'right' (col + 1)
}
#Boundary conditions for 4x4 grid
top_row    = {0, 1, 2, 3}
bottom_row = {12, 13, 14, 15}
left_col   = {0, 4, 8, 12}
right_col  = {3, 7, 11, 15}
boundaries = {
    0: top_row,     # 'up'
    1: bottom_row,  # 'down'
    2: left_col,    # 'left'
    3: right_col    # 'right'
}

#State transition and reward
def next_state_and_reward(s, chosen_decision):
    # Check for boundary
    if s in boundaries[chosen_decision]:
        tentative_next = s  # hit edge, stay put
    else:
        tentative_next = s + movements[chosen_decision]
    # Check for barrier
    if tentative_next == penalty_state:
        next_s = s
        reward = rewards.get(penalty_state, default_reward)
    else:
        next_s = tentative_next
        reward = rewards.get(next_s, default_reward)
    return next_s, reward

#Sample movement direction
def step(s, d):
    decision_probs = {d: 0.8}  # move in intended direction with prob 0.8
    if d == 0 or d == 1:       # 'up' or 'down'
        decision_probs[2] = 0.1  # 'left'
        decision_probs[3] = 0.1  # 'right'
    elif d == 2 or d == 3:     # 'left' or 'right'
        decision_probs[0] = 0.1  # 'up'
        decision_probs[1] = 0.1  # 'down'
    chosen_decision = random.choices(
        list(decision_probs.keys()),
        list(decision_probs.values())
    )[0]
    next_s, reward = next_state_and_reward(s, chosen_decision)
    return next_s, reward

#Generate an episode based on current policy
def generate_episode(policy):
    episode = []
    s = start_state
    steps = 0
    while s != goal_state:
        d = policy[s]
        next_s, reward = step(s, d)
        episode.append((s, d, reward))
        s = next_s
        steps += 1
    return episode

#Evaluate current policy
def policy_evaluation(policy, optimal_value, episodes=NUM_EPISODES):
    returns = defaultdict(list)
    for _ in range(episodes):
        episode = generate_episode(policy)
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, _, reward = episode[t]
            G = reward + Lambda * G
            if s not in visited:
                returns[s].append(G)
                optimal_value[s] = np.mean(returns[s])
                visited.add(s)

#Estimate Q-function
def Q_value(s, d, V):
    if d == 0 or d == 1:  # up/down --> perpendicular: left, right
        outcomes = [(d, 0.8), (2, 0.1), (3, 0.1)]
    else:                 # left/right --> perpendicular: up, down
        outcomes = [(d, 0.8), (0, 0.1), (1, 0.1)]
    q = 0.0
    for chosen_decision, p in outcomes:
        next_s, r = next_state_and_reward(s, chosen_decision)
        q += p * (r + Lambda * V[next_s])
    return q

#Greedy policy improvement
def policy_improvement(policy, optimal_value):
    policy_stable = True
    for s in states:
        # skip if If goal or barrier state
        if s in [goal_state, penalty_state]:
            continue
        old_decision = policy[s]
        #select a decision based on Q_value
        decision_values = {
            d: Q_value(s, d, optimal_value) for d in range(ndecisions)
        }
        best_decision = max(decision_values, key=decision_values.get)
        if old_decision != best_decision:
            policy_stable = False
            policy[s] = best_decision
    return policy_stable

#Solve using Monte Carlo Policy Iteration
np.random.seed(42) #Fix seed for repeatability
policy = { # Initialize policy randomly (ignoring goal and barrier states)
    s: random.choice(range(ndecisions))
    for s in states
    if s not in [goal_state, penalty_state]
}
v = np.zeros(nstates)  #initialize value function

for i in range(max_iterations):
    print(f"Iteration {i + 1}")
    policy_evaluation(policy, v, NUM_EPISODES)
    for s in range(nstates):
        print(f"State {s}: {v[s]:.4f}")
    policy_stable = policy_improvement(policy, v)
    if policy_stable:
        print("Policy converged!")
        break

#Print the final policy
print("\nFinal Policy:")
for s in range(nstates):
    if s == goal_state:
        print(f"State {s}: Goal")
    elif s == penalty_state:
        print(f"State {s}: Barrier")
    else:
        print(f"State {s}: {decisions[policy[s]]}")
