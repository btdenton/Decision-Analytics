#Chapter 5, Example 5.6: Value of the stochastic solution for deliveries
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import itertools

# Input data
n=5 # number of deliveries
c_w = 0.7 # cost coefficient for truck (customer) waiting
c_s = 0.3 # cost coefficient forloading dock (server) waiting
p=[0.25, 0.25, 0.25, 0.25] #probability of service times
z=[1.5, 2.0, 2.5, 3.0] #service times
# Construct all len(z)^n scenarios by generating all combinations of indices
comb_indices = list(itertools.product(range(len(z)), repeat=n)) 
scenarios = np.zeros((len(comb_indices), n + 1)) #Initiate scenarios array
for i, comb in enumerate(comb_indices): # Populate array with scenarios
    prob = np.prod([p[idx] for idx in comb]) #Calculate scenario probabilities
    # Set the elements and their probability in the result array
    scenarios[i, :-1] = [z[idx] for idx in comb] # Set elements of z by indices
    scenarios[i, -1] = prob # Set the probability
Omega = np.zeros((len(scenarios), n)) # 2D array of scenarios of length n

model = gp.Model("Arrival Scheduling Model")# Create a new Gurobi model

# Second stage decision variables for customer waiting (w), server idling (s)
decisions = [('y_w',i,omega) for i in range(n) for omega in range(len(Omega))] + \
            [('y_s',i,omega) for i in range(n) for omega in range(len(Omega))] + \
            [('r',i) for i in range(n - 1)]
decisions = model.addVars(decisions, vtype=GRB.CONTINUOUS, name="decisions")

# Set the objective function
obj = gp.quicksum(scenarios[omega, n] * c_w * decisions['y_w', i, omega] + \
                  scenarios[omega,n] * c_s*decisions['y_s', i, omega] for i \
                      in range(n) for omega in range(len(Omega)))
model.setObjective(obj, GRB.MINIMIZE) # Set to minimize objective

# Create constraints
for omega in range(len(Omega)):
    model.addConstr(decisions['y_w', 1, omega] - decisions['y_s', 1, omega] == \
                    scenarios[omega, 0] - decisions['r', 0], f"balance_{0}_{omega}")
    for i in range(1, n - 1):
        model.addConstr(- decisions['y_w', i, omega] + \
           decisions['y_w', i + 1, omega] - decisions['y_s', i + 1, omega] \
               == scenarios[omega, i] - decisions['r', i], f"balance_{i}_{omega}")

# Solve the 2-SLP appointment scheduling model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    solution={'Objective Value': model.objVal,'r':[decisions[('r', i)].X 
                                                   for i in range(n - 1)],}
    print("Optimal Objective Value:", solution['Objective Value'])
    print("Optimal r:", solution['r'])
else:
    print("No solution found.")