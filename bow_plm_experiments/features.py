import csv
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from bow_plm_experiments.data import GraphBundle


def _feature_cache_path(root: Path, dataset_name: str, plm_model: str) -> Path:
    safe_model_name = plm_model.replace("/", "__").replace(":", "_")
    return root / "feature_cache" / f"{dataset_name}__{safe_model_name}.pt"


def _manual_feature_path(root: Path, dataset_name: str) -> Optional[Path]:
    candidates = [
        root / "manual_features" / f"{dataset_name}_plm.pt",
        root / "manual_features" / f"{dataset_name}_plm.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _manual_text_path(root: Path, dataset_name: str) -> Optional[Path]:
    candidates = [
        root / "texts" / f"{dataset_name}.jsonl",
        root / "texts" / f"{dataset_name}.csv",
        root / "texts" / f"{dataset_name}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_texts(path: Path) -> List[str]:
    if path.suffix == ".jsonl":
        texts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                texts.append(str(record["text"]))
        return texts

    if path.suffix == ".csv":
        texts = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "text" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a 'text' column.")
            for row in reader:
                texts.append(str(row["text"]))
        return texts

    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def _load_manual_feature(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        features = torch.load(path, map_location="cpu")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"{path} must contain a torch.Tensor.")
        return features.float()

    import numpy as np

    return torch.from_numpy(np.load(path)).float()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return masked_hidden.sum(dim=1) / lengths


def _encode_texts(texts: List[str], model_name: str, batch_size: int) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            result = model(**encoded)
            pooled = _mean_pool(result.last_hidden_state, encoded["attention_mask"])
            outputs.append(F.normalize(pooled, p=2, dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def get_features(
    bundle: GraphBundle,
    feature_type: str,
    root: Path,
    plm_model: str,
    batch_size: int,
    force_recompute_plm: bool,
) -> torch.Tensor:
    if feature_type == "bow":
        if bundle.data.x is None:
            raise ValueError(f"{bundle.name} does not provide default node features for BoW.")
        return bundle.data.x.float()

    if feature_type != "plm":
        raise ValueError(f"Unsupported feature type: {feature_type}")

    manual_feature_path = _manual_feature_path(root, bundle.name)
    if manual_feature_path is not None:
        return _load_manual_feature(manual_feature_path)

    cache_path = _feature_cache_path(root, bundle.name, plm_model)
    if cache_path.exists() and not force_recompute_plm:
        return torch.load(cache_path, map_location="cpu").float()

    text_path = _manual_text_path(root, bundle.name)
    if text_path is None:
        raise FileNotFoundError(
            "PLM features require one of the following:\n"
            f"1. Cached embeddings at {cache_path}\n"
            f"2. Manual embeddings at {root / 'manual_features'}\n"
            f"3. Raw node texts at {root / 'texts'}\n"
            f"No PLM source was found for dataset '{bundle.name}'."
        )

    texts = _read_texts(text_path)
    if len(texts) != bundle.data.num_nodes:
        raise ValueError(
            f"Text count mismatch for {bundle.name}: expected {bundle.data.num_nodes}, got {len(texts)}."
        )

    features = _encode_texts(texts, model_name=plm_model, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    return features
