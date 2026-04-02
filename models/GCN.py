import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


def fc(in_features, out_features, dropout):
    return nn.Sequential(nn.BatchNorm1d(in_features),
                         nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(in_features, out_features))

pooling_functions = {
    "add":  global_add_pool,
    "mean": global_mean_pool,
    "max":  global_max_pool,
}


class GCN(nn.Module):
    def __init__(self, 
                 args,
                 input_dim=90,
                 hidden_dim=256,
                 output_dim=1024,
                 num_classes=2,
                 dropout=0.0,
                 norm="batch",
                 act_fn="relu"):
        super(GCN, self).__init__()
        # self.features = args.in_size
        # self.hidden_dim = args.hidden_dim
        # self.num_layers = args.num_layers
        # self.num_classes = args.num_classes

        self.args = args
        self.features = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.dropout = dropout

        #* [batch, layer]
        if norm == "batch":
            self.norm = nn.BatchNorm1d
        elif norm == "layer":
            self.norm = nn.LayerNorm
        else:
            self.norm = None

        #* [relu, gelu]
        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "gelu":
            self.act_fn = nn.GELU()
        else:
            self.act_fn = None

        self.conv1 = GCNConv(self.features, self.hidden_dim)
        self.norm1 = self.norm(self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(self.args.num_layers - 2):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.norms.append(self.norm(self.hidden_dim))

        self.convs.append(GCNConv(self.hidden_dim, self.output_dim))
        self.norms.append(self.norm(self.output_dim))
        # self.norm = nn.BatchNorm1d(self.hidden_dim)



    def init_fc(self):
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x

    def forward(self, data, name=None):
        """_summary_

        Args:
            data (pyg): 
        Returns:
            x: tensor format with [N * output_dim]
        """
        # device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')
        
        # data.to(device)
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = getattr(data, f"{name}_x") if name is not None else data.x
        edge_index = getattr(data, f"{name}_edge_index") if name is not None else data.edge_index

        x = self.conv1(x, edge_index)
        if self.norm is not None:
            x = self.norm1(x)
        if self.act_fn is not None:
            x = self.act_fn(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x) if self.norm is not None else x
            x = self.act_fn(x) if self.act_fn is not None else x
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.bn(x)

        return x

    def __repr__(self):
        return self.__class__.__name__