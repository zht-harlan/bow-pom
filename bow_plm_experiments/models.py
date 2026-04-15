from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import APPNP, GATConv, GCNConv, JumpingKnowledge, SAGEConv, SGConv


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
        hidden_layers = max(1, num_layers)
        dims: List[int] = [in_channels] + [hidden_dim] * hidden_layers + [out_channels]
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
        hidden_layers = max(1, num_layers)
        channels = [in_channels] + [hidden_dim] * hidden_layers + [out_channels]
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
        hidden_layers = max(1, num_layers)
        channels = [in_channels] + [hidden_dim] * hidden_layers + [out_channels]
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
        hidden_layers = max(1, num_layers)
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(hidden_layers - 1):
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


class SGC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pre = nn.Linear(in_channels, hidden_dim)
        self.conv = SGConv(hidden_dim, out_channels, K=max(1, num_layers), cached=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv(x, edge_index)


class JKNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_layers = max(1, num_layers)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(hidden_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.jump = JumpingKnowledge(mode="cat")
        self.classifier = nn.Linear(hidden_dim * hidden_layers, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.jump(xs) if len(xs) > 1 else xs[0]
        return self.classifier(x)


class APPNPNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_layers = max(1, num_layers)
        dims = [in_channels] + [hidden_dim] * hidden_layers
        self.linears = nn.ModuleList(
            nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)
        )
        self.classifier = nn.Linear(hidden_dim, out_channels)
        self.propagation = APPNP(K=10, alpha=0.1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for linear in self.linears:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = linear(x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return self.propagation(x, edge_index)


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
    if model_name == "sgc":
        return SGC(in_channels, hidden_dim, out_channels, num_layers, dropout)
    if model_name == "jknet":
        return JKNet(in_channels, hidden_dim, out_channels, num_layers, dropout)
    if model_name == "appnp":
        return APPNPNet(in_channels, hidden_dim, out_channels, num_layers, dropout)
    raise ValueError(f"Unsupported model: {model_name}")
