import os
from dataset import AMLtoGraph
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from pyvis.network import Network
from collections import deque
import numpy as np  

dataset = AMLtoGraph('/Users/liuyueyi/Desktop/aml/AntiMoneyLaunderingDetectionWithGNN/data')
data = dataset[0]
output_dir = 'visualization'
os.makedirs(output_dir, exist_ok=True)

def sample_suspicious_subgraph(data, sample_size=3000, depth=50):
    G = to_networkx(data, node_attrs=['y'])
    illicit_nodes = [n for n in G.nodes if G.nodes[n]['y'] == 1]
    start_node = np.random.choice(illicit_nodes)
    visited = set([start_node])
    queue = deque([(start_node, 0)]) 
    
    while queue and len(visited) < sample_size:
        node, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_depth + 1))

    return G.subgraph(visited)

subgraph = sample_suspicious_subgraph(data)

net = Network(notebook=True, cdn_resources='remote', height="750px")

for n in subgraph.nodes():
    node = subgraph.nodes[n]
    net.add_node(
        str(n), 
        color='red' if node['y'] == 1 else 'blue',
        size=20 if node['y'] == 1 else 10,
        title=f"""
        Node {n}
        Type: {'Illicit' if node['y'] == 1 else 'Licit'}
        Degree: {subgraph.degree(n)}
        """
    )

for u, v in subgraph.edges():
    net.add_edge(
        str(u), str(v), 
        color='orange' if (subgraph.nodes[u]['y'] == 1 or subgraph.nodes[v]['y'] == 1) else 'lightgray',
        title=f"From {u} to {v}",
        width=2 if (subgraph.nodes[u]['y'] == 1 or subgraph.nodes[v]['y'] == 1) else 1
    )

net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 100
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based"
  }
}
""")
output_file = os.path.join(output_dir, "suspicious_subgraph.html")
net.show(output_file)
