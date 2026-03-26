import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from bow_plm_experiments.data import GraphBundle


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _find_ogbn_arxiv_text_file(dataset_root: Path) -> Optional[Path]:
    candidates = [
        dataset_root / "raw" / "titleabs.tsv.gz",
        dataset_root / "raw" / "titleabs.tsv",
        dataset_root / "mapping" / "titleabs.tsv.gz",
        dataset_root / "mapping" / "titleabs.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_ogbn_arxiv_texts(bundle: GraphBundle) -> Optional[List[str]]:
    text_file = _find_ogbn_arxiv_text_file(bundle.root)
    if text_file is None:
        return None

    texts: List[str] = []
    with _open_maybe_gzip(text_file) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header_skipped = False
        for row in reader:
            if not row:
                continue
            if not header_skipped:
                header_skipped = True
                if any("title" in cell.lower() or "abstract" in cell.lower() for cell in row):
                    continue
            if len(row) >= 3:
                title = row[-2].strip()
                abstract = row[-1].strip()
                text = f"{title}. {abstract}".strip()
            else:
                text = " ".join(cell.strip() for cell in row if cell.strip())
            texts.append(text)

    if len(texts) != bundle.data.num_nodes:
        raise ValueError(
            f"OGBN-Arxiv text count mismatch: expected {bundle.data.num_nodes}, got {len(texts)}."
        )
    return texts


def _feature_to_text(x: torch.Tensor, top_k: int) -> str:
    nonzero_idx = (x != 0).nonzero(as_tuple=False).view(-1)
    if nonzero_idx.numel() == 0:
        return "empty"

    values = x[nonzero_idx].float()
    if nonzero_idx.numel() > top_k:
        _, order = torch.topk(values, k=top_k)
        nonzero_idx = nonzero_idx[order]
        values = values[order]

    tokens = []
    for idx, value in zip(nonzero_idx.tolist(), values.tolist()):
        repeat = 1
        if value >= 2:
            repeat = min(int(round(value)), 3)
        token = f"token{int(idx)}"
        tokens.extend([token] * repeat)
    return " ".join(tokens)


def build_texts_from_features(bundle: GraphBundle, top_k: int = 128) -> List[str]:
    if bundle.data.x is None:
        raise ValueError(f"{bundle.name} does not provide node features for pseudo-text generation.")
    return [_feature_to_text(bundle.data.x[node_idx], top_k=top_k) for node_idx in range(bundle.data.num_nodes)]


def build_dataset_texts(bundle: GraphBundle, top_k: int = 128) -> List[str]:
    if bundle.name == "ogbn-arxiv":
        raw_texts = _load_ogbn_arxiv_texts(bundle)
        if raw_texts is not None:
            return raw_texts
    return build_texts_from_features(bundle, top_k=top_k)


def write_jsonl_texts(texts: Iterable[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
