# GCN_model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CustomGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_mlp = torch.nn.Linear(num_edge_features, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        edge_features = self.edge_mlp(edge_attr)

        # 结合边与点的特征
        x_combined = torch.cat([x[edge_index[0]], edge_features], dim=1)

        out = self.linear(x_combined)
        return out
