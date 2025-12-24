import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, GCNConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels/4), heads=1, concat=False, dropout=0.6)
        self.lin = Linear(int(hidden_channels/4), out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(x)
        x = self.sigmoid(x)
        
        return x
    
class GATv2Convolution(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, edge_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels // 4, heads=1, concat=False, edge_dim=edge_dim, dropout=0.6)
        self.lin = Linear(hidden_channels // 4, out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)  # Dropout
        x = F.elu(self.conv1(x, edge_index, edge_attr))  # First convolution
        x = F.dropout(x, p=0.6, training=self.training)  # Dropout
        x = F.elu(self.conv2(x, edge_index, edge_attr))  # Second convolution
        x = self.lin(x)  # Final linear layer
        x = self.sigmoid(x)  # Sigmoid activation
        
        return x


class GAS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.scorer = Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, edge_attr):  
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        scores = self.scorer(x)
        
        return torch.sigmoid(scores)