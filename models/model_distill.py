import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import unbatch

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


from transformers import BioGptTokenizer, BioGptModel

from .BrainLLaVA import BrainAdapter
from .batch import Batch

norm_fucnction_factory = {
    "batch":    nn.BatchNorm1d,
    "layer":    nn.LayerNorm,
    "none":     lambda x: x,   
}

act_function_factory = {
    "relu":    nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh":    nn.Tanh,
}

readout_factory = {
    "mean":    global_mean_pool,
    "max":     global_max_pool,
    "sum":     global_add_pool,
}


class GCN_d(nn.Module):
    def __init__(self,
                 args, 
                 encoder=None,
                 input_dim=90,
                 hidden_dim=256,
                 output_dim=1024,
                 num_classes=2,
                 dropout=0.0,
                 norm="batch",
                 act_fn="relu",
                 readout="mean"):
        super(GCN_d, self).__init__()

        self.args = args
        self.features = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.encoder = encoder

        #* [batch, layer]
        if norm == "batch":
            self.norm = nn.BatchNorm1d(output_dim)
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
        
        #* [mean, add, max]
        if readout == "mean":
            self.readout = global_mean_pool
        elif readout == "max":
            self.readout = global_max_pool
        elif readout == "add":
            self.readout = global_add_pool
        else:
            self.readout = None

        self.adapter = BrainAdapter(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            norm="batch"
        )
        self.init_fc()


    def init_fc(self):
        # self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.output_dim, self.output_dim // 2)
        self.fc3 = nn.Linear(self.output_dim // 2, self.num_classes)

    def fc_forward(self, x):
        # x = self.act_fn(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act_fn(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x

    def forward(self, data, name=None):
        """_summary_

        Args:
            data (pyg): 
        Returns:
            x (tensor):         [B, num_classes]
            logits (tensor):    [B, hidden_dim]
        """
        device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')
        # data.to(device)
        x = getattr(data, f"{name}_x") if name is not None else data.x
        edge_index = getattr(data, f"{name}_edge_index") if name is not None else data.edge_index
        batch_size  = data.batch.max().item() + 1
        batch   = torch.tensor([[i for j in range(90 * 1)] for i in range(batch_size)], dtype=torch.long).view(-1).to(device)

        x = self.encoder(data, name)
        if self.norm is not None:
            x = self.norm(x)
        # if self.act_fn is not None:
        #     x = self.act_fn(x)
        x = x + self.adapter(x, type="x")

        logits = self.readout(x, batch)
        x = self.fc_forward(logits)

        return x, logits

    def __repr__(self):
        return self.__class__.__name__