import pandas as pd
import torch
import os.path as osp
import json
import random
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import networkx as nx

def load_data(data_path, noAgg=False):
    # Read edges, features and classes from csv files
    df_edges = pd.read_csv(osp.join(data_path, "edgelist.csv"))
    df_features = pd.read_csv(osp.join(data_path, "features.csv"), header=None)
    df_classes = pd.read_csv(osp.join(data_path, "classes.csv"))

    # Name columns based on index
    colNames1 = {'0': 'txId', 1: "Time step"}
    colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(94)}
    colNames3 = {str(ii+96): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

    colNames = dict(colNames1, **colNames2, **colNames3)
    colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}

    # Rename feature columns
    df_features = df_features.rename(columns=colNames)
    if noAgg:
        df_features = df_features.drop(df_features.iloc[:, 96:], axis=1)

    # Map unknown class to '3'
    df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '3'

    # Merge classes and features in one DataFrame
    df_class_feature = pd.merge(df_classes, df_features)

    # Exclude records with unknown class transaction
    df_class_feature = df_class_feature[df_class_feature["class"] != '3']

    # Build DataFrame with head and tail of transactions (edges)
    known_txs = df_class_feature["txId"].values
    df_edges = df_edges[(df_edges["txId1"].isin(known_txs)) & (df_edges["txId2"].isin(known_txs))]

    # Create graph
    G = nx.from_pandas_edgelist(df_edges, 'txId1', 'txId2')

    # Find all connected components
    connected_components = list(nx.connected_components(G))

    # Randomly select nodes to reach target count
    selected_node_ids = set()
    for component in connected_components:
        selected_node_ids.update(component)
        if len(selected_node_ids) >= 10000:
            break

    # If not enough nodes, randomly select more
    if len(selected_node_ids) < 10000:
        additional_nodes = random.sample(set(G.nodes()) - selected_node_ids, 10000 - len(selected_node_ids))
        selected_node_ids.update(additional_nodes)

    # Filter edges to those connecting to selected nodes
    df_edges_filtered = df_edges[(df_edges["txId1"].isin(selected_node_ids)) & (df_edges["txId2"].isin(selected_node_ids))]
    df_class_feature = df_class_feature.dropna(subset=['class'])
    # Print class distribution for selected nodes
    print("\nClass distribution:")
    print(df_class_feature[df_class_feature["txId"].isin(selected_node_ids)]['class'].value_counts())

    # Count the number of edges in the filtered graph
    num_edges = df_edges_filtered.shape[0]
    print(f"Number of edges: {num_edges}")

    # Generate JSON format data
    json_data = {
        "nodes": [
            {
                "id": int(row["txId"]),
                "class": 0 if int(row["class"]) == 1 else 1,
                "time_step": int(row["Time step"]),
                "local_features": row.iloc[3:97].astype(float).tolist(),
                "aggregate_features": row.iloc[97:].astype(float).tolist()
            }
            for _, row in df_class_feature[df_class_feature["txId"].isin(selected_node_ids)].iterrows()
        ],
        "links": [
            {
                "source": int(u),
                "target": int(v),
                "weight": 1.0
            }
            for u, v in zip(df_edges_filtered["txId1"].astype(int).values, df_edges_filtered["txId2"].astype(int).values)
        ]
    }

    # Save to JSON file
    with open(osp.join(data_path, 'subgraph_data2.json'), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    print("Graph data saved to subgraph_data2.json")

    return selected_node_ids, df_edges_filtered

load_data('/home/zmxu/Desktop/ly/aml-elliptic-gnn/data/', noAgg=False)