#Chapter 3, Example 3.7: Shortest path problem with cycles
from math import inf

# Input data
network = {  # directed, weighted adjacency list
    '1': {'2': 5, '3': 1},
    '2': {'3': 4, '4': 6, '5': 7},
    '3': {'4': 1, '5': 2},
    '4': {'6': 2},
    '5': {'4': 3, '6': 8},
    '6': {}
}
source = '1' #start node

def shortestpath_dp(adj, source, target=None):
    v = {u: inf for u in adj}      #initialize value function
    parent = {u: None for u in adj}
    v[source] = 0.0
    N = set() #initialize to empty set
    stage = 0
    while len(N) < len(adj):
        unvisited = [u for u in adj if u not in N] 
        k = min(unvisited, key=lambda u: v[u]) #Find nearest node not in N
        if v[k] == inf: break  # break if remaining nodes unreachable
        N.add(k) #add nearest node to set of visited nodes 
        stage += 1
        if target is not None and k == target: break
        for j, w_kj in adj[k].items():  #Update path length from k to neighbors

            if j in N:
                continue
            if v[k] + w_kj < v[j]:
                v[j] = v[k] + w_kj
                parent[j] = k
    return v, parent

def reconstruct_path(parent, target):
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))

#Find shortest path from start node to all nodes in network
v, parent = shortestpath_dp(network, source=source)

print(f"\nShortest paths from {source}:")
for node in sorted(network):
    if v[node] < inf:
        path = reconstruct_path(parent, node)
        length = int(v[node]) if float(v[node]).is_integer() else v[node]
        print(f"{source} → {node}: {'-->'.join(path)} (length={length})")
    else: print(f"  {source} → {node}: unreachable")

