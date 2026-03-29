#Chapter 5, Example 5.8:  Multi-stage financial planning
import gurobipy as gp
from gurobipy import GRB

# Input data
b = 15000 # Initial investment
G = 25000 # Target goal
q = 1.12 # Surplus rate
r = 1.15 # Shortage rate
prob = 0.125 # Probability of each scenario
scenarios = ['L', 'H']
z = {('H',1): 1.25, ('L',1): 1.06, ('H',2): 1.14, ('L',2): 1.12} # Returns 

# Create a new Gurobi model
model = gp.Model("Financial Planning")

# Decision variables
decisions = [('x', i, 1) for i in [1, 2]] # Stage 1, x: stocks (i=1),bonds (i=2)
decisions += [('x',i,2,s1) for s1 in scenarios for i in [1,2]] # Stage 2 variables
decisions += [('x', i, 3, s1, s2) for s1 in scenarios for s2 \
              in scenarios for i in [1, 2]] # Stage 3 variables
decisions += [(var, s1, s2, s3) # Stage 4,w: shortage, y: surplus
            for s1 in scenarios
            for s2 in scenarios
            for s3 in scenarios
            for var in ['w', 'y']] 
decisions = model.addVars(decisions, name="decisions")

# Objective function
obj = gp.quicksum(prob * (- r * decisions['w', s1, s2, s3] + q * \
                          decisions['y', s1, s2, s3]) 
                    for s1 in scenarios for s2 in scenarios for s3 in scenarios)
model.setObjective(obj, GRB.MAXIMIZE)

# Constraints
model.addConstr(gp.quicksum( decisions['x', i, 1] for i in \
                      [1, 2]) == b,"Initial_Investment") # Investment constraint

for s1 in scenarios:  # Portfolio rebalancing constraints for stage 2
    model.addConstr(gp.quicksum(- z[(s1, i)] *  decisions['x', i, 1] +\
                                decisions['x', i, 2, s1] 
                    for i in [1, 2]) == 0, f"Rebalance_2_{s1}")

for s1 in scenarios: # Portfolio rebalancing constraints for stage 3
    for s2 in scenarios:
        model.addConstr(gp.quicksum(- z[(s2, i)] *  decisions['x', i, 2, s1] +\
                                    decisions['x', i, 3, s1, s2] 
                        for i in [1, 2]) == 0, f"Rebalance_3_{s1}{s2}")
# Final stage constraints
for s1 in scenarios:
   for s2 in scenarios:
      for s3 in scenarios:
        model.addConstr(gp.quicksum(z[(s3, i)]*decisions['x',i,3,s1,s2] \
                for i in [1, 2]) - decisions['y', s1, s2, s3] + \
                        decisions['w', s1, s2, s3] == G, f"Goal_{s1}{s2}{s3}")

model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    for v in model.getVars():
        if v.X > 1e-6:
            name = v.VarName.replace("decisions[", "").rstrip("]")
            parts = [p.strip().strip("'") for p in name.split(",")]
            print(f"{parts[0]}({','.join(parts[1:])}): {v.X}")
else:
    print("Optimal solution not found.")

