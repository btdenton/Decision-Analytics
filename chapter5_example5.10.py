#Chapter 5, Example 5.10:  EVPI for the stochastic unit commitment problem
import gurobipy as gp
from gurobipy import GRB

# Input data
I = [1, 2] # Set of power-generating units
T = [1, 2, 3] # Set of time periods
Omega = ['High', 'Low'] # Scenarios
p = {'High': 0.6, 'Low': 0.4} # Scenario probabilities
c_start = {(1, 1): 10, (1, 2): 10, (1, 3): 10,
           (2, 1): 12, (2, 2): 12, (2, 3): 12} # Start-up costs
c_var = {(1, 1): 20, (1, 2): 20, (1, 3): 20,
         (2, 1): 22, (2, 2): 22, (2, 3): 22} # Variable generation costs
c_short = {1: 50, 2: 50, 3: 50} # Short-term generation costs
M = {1: 100, 2: 80} # Maximum generation capacity
d = {(1, 'High'): 150, (2, 'High'): 100, (3, 'High'): 200,
     (1, 'Low'): 100, (2, 'Low'): 80, (3, 'Low'): 120} # Demand scenarios

# Create a new Gurobi model
model = gp.Model("SUCP")

# Decision variables
decisions = {}
decisions.update(model.addVars((('u', i, t) for i in I for t in T), \
                               vtype=GRB.BINARY, name="u"))
decisions.update(model.addVars((('x', i, t, omega) for i in I for t in T for \
                         omega in Omega), vtype=GRB.CONTINUOUS, name="x",lb=0))
decisions.update(model.addVars((('e', t, omega) for t in T for omega in Omega),\
                               vtype=GRB.CONTINUOUS, name="e", lb=0))

# Objective function
obj = (gp.quicksum(c_start[i, t]*decisions['u',i,t] for i in I for t in T) +\
      gp.quicksum(p[omega]*(gp.quicksum(c_var[i,t]*decisions['x',i,t,omega] for\
                       i in I for t in T) + gp.quicksum(c_short[t] * \
                       decisions['e',t,omega] for t in T)) for omega in Omega))
model.setObjective(obj, GRB.MINIMIZE)

# Power balance constraints
model.addConstrs((gp.quicksum(decisions['x', i, t, omega] for i in I) + \
                  decisions['e', t, omega] == d[t, omega]
                  for t in T for omega in Omega), "PowerBalance")

# Generation limits
model.addConstrs((decisions['x', i, t, omega] <= decisions['u', i, t] * \
             M[i] for i in I for t in T for omega in Omega), "GenerationLimits")

# Solve model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    for i in I:
        for t in T:
            if decisions['u', i, t].X > 0.5: # Operational units
                print(f"Unit {i} is operational at time {t}.")
                for omega in Omega:
                    print(f" Generation under scenario {omega}: \
                          {decisions['x', i, t, omega].X:.2f}")
    for t in T:
        for omega in Omega:
            if decisions['e', t, omega].X > 0.01: # Short-term generation
                print(f"Short-term generation at time {t} under scenario \
                      {omega}: {decisions['e', t, omega].X:.2f}")