#Chapter 8, Example 8.5: Dynamic pricing
import numpy as np
import random

# Input data
states = range(10, 51, 5) # Possible competitor prices
decisions = range(10, 51, 5) # Possible prices ($10 to $50 in $5 increments)
demand_sensitivity = 0.2 # Sensitivity of demand to price differences
cost_per_unit = 5 # Fixed cost per product
episodes = 1000 # Number of training episodes
max_steps = 12 # Maximum steps per episode 
epsilon = 0.2 # Initial exploration rate
epsilon_decay = 0.99 # Rate of decrease in epsilon per epoch
alpha = 0.05 #recency weight
Lambda = 0.97 # Economic discount factor representing time value of money
base_demand = 10 # Minimum demand independent of price
nstates = len(states)
ndecisions = len(decisions)
np.random.seed(42) #Fix seed for repeatability

# Simulate customer demand as a function of the agent and competitor price
def simulate_demand(agent_price, competitor_price):
    price_difference = competitor_price - agent_price
    demand = max(0,base_demand+demand_sensitivity*price_difference*base_demand)
    return int(demand)

# Compute reward (profit) based on the price decision and the demand generated
def rewards(decision, demand):
    return (decision - cost_per_unit) * demand

# Choose a decision (price) using the epsilon-greedy policy
def choose_decision(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(ndecisions)) # Explore
    else:
        return np.argmax(Q_table[state]) # Exploit decision with highest Q-value
        
# Test the learned policy based on a test scenario
def test_policy():
    s = random.choice(range(nstates))
    total_profit = 0
    for step in range(max_steps):
        d = np.argmax(Q_table[s])
        # Simulate the environment
        demand = simulate_demand(decisions[d], states[s])
        profit = rewards(decisions[d], demand)
        total_profit += profit
        print(f"Step {step + 1}: Competitor Price = ${states[s]}, \
              Agent Price = ${decisions[d]}")
        s = random.choice(range(nstates)) #new competitor price
    print(f"Total Profit: ${total_profit}")

# Initialize Q-table with small random values
Q_table = np.random.uniform(low=1, high=2, size=(nstates, ndecisions))

# Q-Learning Training Loop with epsilon decaying over time
for episode in range(episodes):
    decaying_epsilon = max(0.01,epsilon*(epsilon_decay ** episode)) 
    for step in range(max_steps):
        s = random.choice(range(nstates))  # Competitor's random price decision
        d = choose_decision(s, decaying_epsilon)
        demand = simulate_demand(decisions[d], states[s])
        reward = rewards(decisions[d], demand)
        next_s = random.choice(range(nstates))
        max_future_q = np.max(Q_table[next_s])
        # Update Q-value
        current_Q = Q_table[s, d]
        new_q = current_Q + alpha *(reward + Lambda * max_future_q - current_Q)
        Q_table[s, d] = new_q

# Optimal policy
optimal_policy = [decisions[np.argmax(Q_table[s])] for s in range(nstates)]
print("\n Policy:")
for state, decision in zip(states, optimal_policy):
    print(f"Competitor Price: ${state} -> Optimal Agent Price: ${decision}")



