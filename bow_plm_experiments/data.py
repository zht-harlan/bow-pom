from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid


@dataclass
class GraphBundle:
    name: str
    data: Data
    num_classes: int
    split_idx: Dict[str, torch.Tensor]
    root: Path


def _mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).view(-1)


def _build_split_idx_from_masks(data: Data) -> Dict[str, torch.Tensor]:
    return {
        "train": _mask_to_index(data.train_mask),
        "valid": _mask_to_index(data.val_mask),
        "test": _mask_to_index(data.test_mask),
    }


def _random_split(labels: torch.Tensor, seed: int) -> Dict[str, torch.Tensor]:
    y = labels.view(-1).cpu().numpy()
    indices = np.arange(y.shape[0])

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=y[temp_idx],
    )

    return {
        "train": torch.as_tensor(train_idx, dtype=torch.long),
        "valid": torch.as_tensor(valid_idx, dtype=torch.long),
        "test": torch.as_tensor(test_idx, dtype=torch.long),
    }


def load_dataset(name: str, root: Path, seed: int) -> GraphBundle:
    canonical_name = name.lower()

    if canonical_name in {"cora", "pubmed"}:
        planetoid_name = {"cora": "Cora", "pubmed": "PubMed"}[canonical_name]
        dataset = Planetoid(root=str(root / canonical_name), name=planetoid_name)
        data = dataset[0]
        split_idx = _build_split_idx_from_masks(data)
        return GraphBundle(
            name=canonical_name,
            data=data,
            num_classes=dataset.num_classes,
            split_idx=split_idx,
            root=root / canonical_name,
        )

    if canonical_name == "amazon-photo":
        dataset = Amazon(root=str(root / canonical_name), name="Photo")
        data = dataset[0]
        split_idx = _random_split(data.y, seed)
        return GraphBundle(
            name=canonical_name,
            data=data,
            num_classes=dataset.num_classes,
            split_idx=split_idx,
            root=root / canonical_name,
        )

    if canonical_name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / canonical_name))
        data = dataset[0]
        data.y = data.y.view(-1)
        split_idx = dataset.get_idx_split()
        return GraphBundle(
            name=canonical_name,
            data=data,
            num_classes=dataset.num_classes,
            split_idx={key: value.view(-1) for key, value in split_idx.items()},
            root=root / canonical_name,
        )

    raise ValueError(f"Unsupported dataset: {name}")
