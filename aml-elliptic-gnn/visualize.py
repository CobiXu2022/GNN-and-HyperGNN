import pandas as pd
import torch
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from loader import load_data, data_to_pyg  
import utils as u
import matplotlib.colors as mcolors
from pyvis.network import Network
import os

def visualize_graph(data_path, noAgg=False, interactive=True, subgraph=False):
    # Load data
    df_class_feature, df_edges = load_data(data_path, noAgg)
    data = data_to_pyg(df_class_feature, df_edges)

    # Create full graph
    graph = to_networkx(data, to_undirected=True)
    print(f"Total nodes: {len(graph.nodes())}, Total edges: {len(graph.edges())}")

    cmap = plt.get_cmap('viridis')
    licit_color = mcolors.rgb2hex(cmap(1.0))  # Purple (class 1)
    illicit_color = mcolors.rgb2hex(cmap(0.0))  # Yellow (class 0)
    edge_color = illicit_color 

    # Create subgraph via BFS if requested
    if subgraph:
        start_node = np.random.choice(np.where(data.y.numpy() == 1)[0])  # Assuming label '1' is illicit
        print(f"Start-node BFS: {start_node} (Class: {data.y[start_node].item()})")
        graph_edges = list(nx.bfs_edges(graph, start_node, depth_limit=5))
        graph = graph.edge_subgraph(graph_edges)
        print(f"Sub-graph stats: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    if interactive:
        os.environ['PYVIS_TEMPLATE_PATH'] = '/home/zmxu/miniconda3/envs/aml/share/pyvis/templates'
        output_path = './data/interactive_network.html'
        net = Network(
            height="1200px", 
            width="100%", 
            notebook=False,
            directed=False,
            bgcolor="#222222",  # Dark background for better contrast
            font_color="white"
        )
        # Physics configuration as JSON string
        physics_options = """
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -300,
                    "centralGravity": 0.02,
                    "springLength": 200,
                    "springConstant": 0.05,
                    "damping": 0.3,
                    "avoidOverlap": 0.8
                },
                "maxVelocity": 5,
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": true,
                    "iterations": 500,
                    "updateInterval": 10
                }
            }
        }
        """
        
        # Add nodes
        illicit_nodes = set(np.where(data.y == 0)[0])
        for node in graph.nodes():
            is_illicit = node in illicit_nodes
            net.add_node(
                node,
                label="",
                color=illicit_color if is_illicit else licit_color,
                size=10 if is_illicit else 3,
                title=f"ID: {node} | Type: {'Illicit' if is_illicit else 'Licit'}"
            )

        # Add edges
        for u, v in graph.edges():
            is_suspicious = u in illicit_nodes or v in illicit_nodes
            net.add_edge(
                u, v,
                color=edge_color if is_suspicious else "#666666",
                width=1.5 if is_suspicious else 0.3,
                opacity=0.8 if is_suspicious else 0.2
            )

        # Set options correctly
        net.set_options(physics_options)
        net.save_graph(output_path)
        print(f"Interactive visualization saved to {output_path}")
    else:
        # Prepare plot
        sns.set(style="white")
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, k=0.1, iterations=50) 

        # Get node colors
        node_colors = 1 - data.y.numpy()[list(graph.nodes())]  

        # Classify edges
        illicit_nodes = set(np.where(data.y == 0)[0]) & set(graph.nodes())  
        suspicious_edges = [(u,v) for u,v in graph.edges() 
                            if u in illicit_nodes or v in illicit_nodes]
        normal_edges = list(set(graph.edges()) - set(suspicious_edges))

        # Visualization layers
        nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, width=0.5, alpha=0.3)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap='viridis', node_size=20, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, edgelist=suspicious_edges, edge_color=plt.cm.viridis(1.0), width=0.5, alpha=0.8)
        nx.draw_networkx_nodes(graph, pos, nodelist=list(illicit_nodes), node_color=plt.cm.viridis(1.0), node_size=50, alpha=1.0)

        # Enhanced legend
        class_names = {1: "Illicit", 0: "Licit"}
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=class_names[0],
                        markerfacecolor=plt.cm.viridis(0.0), markersize=20),
            plt.Line2D([0], [0], marker='o', color='w', label=class_names[1],
                        markerfacecolor=plt.cm.viridis(1.0), markersize=20),
            plt.Line2D([0], [0], color=plt.cm.viridis(1.0), lw=4, label='Suspicious Edges')
        ]

        plt.legend(handles=legend_elements, title="Network Components", fontsize=16, title_fontsize=18, bbox_to_anchor=(1.1, 1), loc='upper left')

        plt.title(f"Network of Illicit Node", fontsize=20)
        plt.axis('off')

        # Save output
        plt.tight_layout()
        plt.savefig(
            './data/suspicious_subnetwork.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        print("Visualization saved to suspicious_subnetwork.png")
        plt.close()

if __name__ == "__main__":
    data_path = "./data"  
    visualize_graph(data_path, noAgg=False, interactive=True, subgraph=True)


