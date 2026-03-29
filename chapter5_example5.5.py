#Chapter 5, Example 5.5: Vaccine purchase planning
import gurobipy as gp
from gurobipy import GRB

# Input data
c_u = 0.8 # cost coefficient for underage, y_u
c_o = 0.2 # cost coefficient for overage, y_o
p = [0.4, 0.5, 0.1] # probability vector for demand
z = [1500, 2200, 2800] # z values for demand for each omega
Omega = range(len(p)) # Index set for all demand outcomes omega

# Create a new Gurobi model
model = gp.Model("Newsvendor Model")

# Decision variables
decisions = [('x',)] + \
            [('y_u', omega) for omega in Omega] + \
            [('y_o', omega) for omega in Omega]
decisions = model.addVars(decisions, vtype=GRB.CONTINUOUS, name="decisions")

# Objective function
obj = c_u*gp.quicksum(p[omega]*decisions[('y_u', omega)] for omega in Omega) + \
    c_o*gp.quicksum(p[omega] * decisions[('y_o', omega)] for omega in Omega)
model.setObjective(obj, GRB.MINIMIZE)

# Constraints
for omega in Omega:
    model.addConstr(decisions[('y_u', omega)] - decisions[('y_o', omega)] == \
                    z[omega] - decisions[('x',)], f"balance_{omega}")

# Optimize model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    solution = {
        'Objective Value': model.objVal,
        'x': decisions[('x',)].x,
        'y_u': {omega: decisions[('y_u', omega)].x for omega in Omega},
        'y_o': {omega: decisions[('y_o', omega)].x for omega in Omega}
    }

    print("Optimal Objective Value:", solution['Objective Value'])
    print("Optimal x:", solution['x'])
    print("Optimal y_u:", solution['y_u'])
    print("Optimal y_o:", solution['y_o'])
    
else:
    print("No solution found.")