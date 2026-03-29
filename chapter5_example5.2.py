#Chapter 5, Example 5.2: Shortest path problem
import gurobipy as gp
from gurobipy import GRB

# Input data
vertices = ['A', 'B', 'C', 'D'] # Start vertex A and end vertex D 
edges = {('A','B'): 2, ('A','C'): 4, ('B','C'): 1, ('B','D'): 7, ('C','D'): 3}

# Create a Gurobi linear programming model
model = gp.Model("shortest_path")

# Vector of decision variables: Add variables for edges 
decisions = model.addVars(edges, vtype=GRB.BINARY, name="edge")

# Create objective function to minimize the total cost of selected edges
model.setObjective(gp.quicksum(decisions[i, j]*edges[i, j] 
                               for (i, j) in edges), GRB.MINIMIZE)

# Add constraint for start vertex A (ensure sum of outgoing edges is 1)
model.addConstr(gp.quicksum(decisions['A',j] 
                            for j in vertices if ('A',j) in edges) == 1,"start")

# Add constraint for end vertex D (ensure sum of incoming edges is 1)
model.addConstr(gp.quicksum(decisions[i,'D'] 
                            for i in vertices if (i,'D') in edges) == 1, "end")

# Add flow balance constraints for all vertices except start and end
for vertex in vertices[1:-1]: # Vertices B and C in this case
    model.addConstr(
    gp.quicksum(decisions[j, vertex] for j in vertices if (j, vertex) in edges) ==
    gp.quicksum(decisions[vertex, k] for k in vertices if (vertex, k) in edges), 
    f"flow_balance_{vertex}")

# Optimize the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Total Cost of Shortest Path:", model.objVal)
    for (i, j), var in decisions.items():
        if var.x > 0.999: # Interpret values close to 1 as 1
            print(f"Path includes edge {i} to {j} with weight {edges[(i, j)]}")
