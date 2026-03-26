import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.data import Data


@dataclass
class TrainConfig:
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    num_layers: int
    heads: int
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1).cpu().numpy()
    target = labels.cpu().numpy()
    acc = float((pred == target).mean())
    macro_f1 = float(f1_score(target, pred, average="macro"))
    return {"acc": acc, "f1_macro": macro_f1}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    features: torch.Tensor,
    split_idx: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    model.eval()
    logits = model(features, data.edge_index).cpu()
    labels = data.y.view(-1).cpu()
    results = {}
    for split_name, indices in split_idx.items():
        cpu_indices = indices.cpu()
        results[split_name] = _metrics(logits[cpu_indices], labels[cpu_indices])
    return results


def fit(
    model: nn.Module,
    data: Data,
    features: torch.Tensor,
    split_idx: Dict[str, torch.Tensor],
    config: TrainConfig,
) -> Dict[str, Dict[str, float]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    data = data.to(device)
    features = features.to(device)
    labels = data.y.view(-1).long()
    split_idx = {key: value.to(device) for key, value in split_idx.items()}

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_valid_acc = float("-inf")
    best_valid_f1 = float("-inf")
    stale_epochs = 0

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, data.edge_index)
        loss = F.cross_entropy(logits[split_idx["train"]], labels[split_idx["train"]])
        loss.backward()
        optimizer.step()

        scores = evaluate(model, data, features, split_idx)
        valid_acc = scores["valid"]["acc"]
        valid_f1 = scores["valid"]["f1_macro"]

        improved = (valid_acc > best_valid_acc) or (
            np.isclose(valid_acc, best_valid_acc) and valid_f1 > best_valid_f1
        )
        if improved:
            best_valid_acc = valid_acc
            best_valid_f1 = valid_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    return evaluate(model, data, features, split_idx)
