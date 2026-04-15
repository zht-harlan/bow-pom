import csv
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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
    text_path: Optional[Path] = None


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


def _maybe_num_classes(labels: Optional[torch.Tensor]) -> Optional[int]:
    if labels is None:
        return None
    return int(labels.view(-1).max().item()) + 1


def _coerce_to_data(obj: object) -> Data:
    if isinstance(obj, Data):
        return obj

    if isinstance(obj, (list, tuple)):
        if not obj:
            raise TypeError("Empty serialized dataset container.")
        if isinstance(obj[0], Data):
            return obj[0]
        return _coerce_to_data(obj[0])

    if isinstance(obj, dict):
        if "data" in obj:
            return _coerce_to_data(obj["data"])
        data_kwargs = {}
        for key in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask", "num_nodes"):
            if key in obj:
                data_kwargs[key] = obj[key]
        if data_kwargs:
            return Data(**data_kwargs)

    data_kwargs = {}
    for key in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask", "num_nodes"):
        if hasattr(obj, key):
            data_kwargs[key] = getattr(obj, key)
    if data_kwargs:
        return Data(**data_kwargs)

    raise TypeError(f"Unsupported serialized dataset object: {type(obj)!r}")


def _load_local_csv_graph(csv_path: Path) -> Data:
    labels = []
    edges = []
    texts_present = False

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"label", "node_id", "neighbour"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader):
            node_id = int(row["node_id"])
            if node_id != row_idx:
                raise ValueError(
                    f"{csv_path} contains non-contiguous node_id values: expected {row_idx}, got {node_id}."
                )

            labels.append(int(row["label"]))
            texts_present = texts_present or bool(row.get("text"))

            neighbours = literal_eval(row["neighbour"]) if row["neighbour"] else []
            for neighbour in neighbours:
                edges.append((node_id, int(neighbour)))

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = Data(
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        num_nodes=len(labels),
    )
    if not texts_present:
        raise ValueError(f"{csv_path} does not contain usable text rows for BoW generation.")
    return data


def _load_local_dataset(name: str, root: Path, seed: int) -> GraphBundle:
    folder_name = {
        "children": "Children",
        "history": "History",
        "photo": "Photo",
    }[name]
    dataset_root = root / "CSTAG" / folder_name
    pt_path = dataset_root / f"{folder_name}.pt"
    csv_path = dataset_root / f"{folder_name}.csv"

    data: Optional[Data] = None
    if pt_path.exists():
        try:
            data = _coerce_to_data(torch.load(pt_path, map_location="cpu"))
        except Exception:
            data = None

    csv_data: Optional[Data] = None
    if csv_path.exists():
        csv_data = _load_local_csv_graph(csv_path)

    if data is None:
        if csv_data is None:
            raise FileNotFoundError(f"No supported local dataset source found for {name} in {dataset_root}.")
        data = csv_data
    elif csv_data is not None:
        if getattr(data, "edge_index", None) is None:
            data.edge_index = csv_data.edge_index
        if getattr(data, "y", None) is None:
            data.y = csv_data.y
        if getattr(data, "num_nodes", None) is None:
            data.num_nodes = csv_data.num_nodes

    if getattr(data, "y", None) is None:
        raise ValueError(f"{name} is missing labels.")

    if all(getattr(data, mask_name, None) is not None for mask_name in ("train_mask", "val_mask", "test_mask")):
        split_idx = _build_split_idx_from_masks(data)
    else:
        split_idx = _random_split(data.y, seed)

    num_classes = _maybe_num_classes(data.y)
    if num_classes is None:
        raise ValueError(f"Unable to determine number of classes for {name}.")

    return GraphBundle(
        name=name,
        data=data,
        num_classes=num_classes,
        split_idx=split_idx,
        root=dataset_root,
        text_path=csv_path if csv_path.exists() else None,
    )


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

    if canonical_name in {"children", "history", "photo"}:
        return _load_local_dataset(canonical_name, root, seed)

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
