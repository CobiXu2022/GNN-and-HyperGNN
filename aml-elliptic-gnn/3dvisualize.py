import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.utils.convert import to_networkx
from loader import load_data, data_to_pyg

def visualize_subgraph(data_path, output_path, noAgg=False):
    # Load data
    df_class_feature, df_edges = load_data(data_path, noAgg)
    data = data_to_pyg(df_class_feature, df_edges)

    # Create full graph
    graph = to_networkx(data, to_undirected=True)
    print(f"Total nodes: {len(graph.nodes())}, Total edges: {len(graph.edges())}")

    illicit_nodes = set(np.where(data.y.numpy() == 0)[0])  
    if not illicit_nodes:
        print("No illicit nodes found!")
        return

    start_node = np.random.choice(list(illicit_nodes)) 
    print(f"Starting BFS from node: {start_node}")

    # Create subgraph via BFS
    subgraph_edges = list(nx.bfs_edges(graph, start_node, depth_limit=10))
    if not subgraph_edges:
        print("No edges found in the subgraph")
        return

    subgraph = graph.edge_subgraph(subgraph_edges)
    print(f"Sub-graph stats: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")

    # Prepare 3D visualization
    pos = nx.spring_layout(subgraph, dim=3)  # 3D layout
    edge_x = []
    edge_y = []
    edge_z = []
    edge_colors = []
    
    normal_edges_x = []
    normal_edges_y = []
    normal_edges_z = []
    
    highlight_edges_x = []
    highlight_edges_y = []
    highlight_edges_z = []

    for u, v in subgraph.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]

        normal_edges_x.extend([x0, x1, None])
        normal_edges_y.extend([y0, y1, None])
        normal_edges_z.extend([z0, z1, None])

        if u in illicit_nodes or v in illicit_nodes:
            highlight_edges_x.extend([x0, x1, None])
            highlight_edges_y.extend([y0, y1, None])
            highlight_edges_z.extend([z0, z1, None])

    # Create edges traces
    normal_edge_trace = go.Scatter3d(
        x=normal_edges_x,
        y=normal_edges_y,
        z=normal_edges_z,
        line=dict(width=0.5, color='#888'),  
        hoverinfo='none',
        mode='lines'
    )
    
    highlight_edge_trace = go.Scatter3d(
        x=highlight_edges_x,
        y=highlight_edges_y,
        z=highlight_edges_z,
        line=dict(width=2, color='yellow'),  
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_z = []
    node_color = []
    node_texts = [] 

    for node in subgraph.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        if data.y[node] == 0:
            node_color.append('yellow')  
            class_label = "Illicit"
        else:
            node_color.append('purple') 
            class_label = "Licit"
        
        hover_text = f'Node ID: {node}<br>Class: {class_label}'
        node_texts.append(hover_text)

    # Create node trace
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            size=5,
            color=node_color,
            opacity=0.8
        ),
        text=node_texts,
        hoverinfo='text'
    )

    # Create figure
    fig = go.Figure(data=[normal_edge_trace, highlight_edge_trace, node_trace],
                    layout=go.Layout(
                        title='3D Subgraph Visualization',
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ),
                        margin=dict(l=0, r=0, b=0, t=0)
                    ))

    # Save the figure as an HTML file
    fig.write_html(output_path)
    print(f"3D visualization saved to {output_path}")

if __name__ == "__main__":
    # Customize this path as needed
    data_path = "./data"  # Replace with the actual path to your data
    output_path = "./data/subgraph_visualization3.html"  # Output HTML file path
    visualize_subgraph(data_path, output_path, noAgg=False)