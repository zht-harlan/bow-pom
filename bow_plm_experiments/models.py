from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dims: List[int] = [in_channels] + [hidden_dim] * (num_layers - 1) + [out_channels]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return self.net(x)


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = [in_channels] + [hidden_dim] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            GCNConv(channels[idx], channels[idx + 1]) for idx in range(len(channels) - 1)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = [in_channels] + [hidden_dim] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            SAGEConv(channels[idx], channels[idx + 1]) for idx in range(len(channels) - 1)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        heads: int,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
            return

        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(
            GATConv(hidden_dim * heads, out_channels, heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def build_model(
    model_name: str,
    in_channels: int,
    hidden_dim: int,
    out_channels: int,
    num_layers: int,
    dropout: float,
    heads: int,
) -> nn.Module:
    if model_name == "mlp":
        return MLP(in_channels, hidden_dim, out_channels, num_layers, dropout)
    if model_name == "gcn":
        return GCN(in_channels, hidden_dim, out_channels, num_layers, dropout)
    if model_name == "sage":
        return GraphSAGE(in_channels, hidden_dim, out_channels, num_layers, dropout)
    if model_name == "gat":
        return GAT(in_channels, hidden_dim, out_channels, num_layers, dropout, heads)
    raise ValueError(f"Unsupported model: {model_name}")
