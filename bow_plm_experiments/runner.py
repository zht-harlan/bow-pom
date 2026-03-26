import csv
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from bow_plm_experiments.data import load_dataset
from bow_plm_experiments.features import get_features
from bow_plm_experiments.models import build_model
from bow_plm_experiments.trainer import TrainConfig, fit


def _write_raw_results(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "feature_type",
                "model",
                "dataset",
                "acc",
                "f1_macro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["feature_type"], row["model"], row["dataset"])
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (feature_type, model, dataset), group in grouped.items():
        acc_values = [float(item["acc"]) for item in group]
        f1_values = [float(item["f1_macro"]) for item in group]
        summary_rows.append(
            {
                "feature_type": feature_type,
                "model": model,
                "dataset": dataset,
                "acc": round(mean(acc_values), 4),
                "acc_std": round(pstdev(acc_values), 4) if len(acc_values) > 1 else 0.0,
                "f1_macro": round(mean(f1_values), 4),
                "f1_macro_std": round(pstdev(f1_values), 4) if len(f1_values) > 1 else 0.0,
                "runs": len(group),
            }
        )
    return summary_rows


def _write_summary_results(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_type",
                "model",
                "dataset",
                "acc",
                "acc_std",
                "f1_macro",
                "f1_macro_std",
                "runs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_wide_results(path: Path, rows: List[Dict[str, object]]) -> None:
    dataset_order = ["ogbn-arxiv", "cora", "pubmed", "amazon-photo"]
    key_order = [
        (dataset, metric)
        for dataset in dataset_order
        for metric in ("acc", "acc_std", "f1_macro", "f1_macro_std")
    ]

    grouped: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        group_key = (row["feature_type"], row["model"])
        if group_key not in grouped:
            grouped[group_key] = {
                "feature_type": row["feature_type"],
                "model": row["model"],
            }
        grouped[group_key][f"{row['dataset']}_acc"] = row["acc"]
        grouped[group_key][f"{row['dataset']}_acc_std"] = row["acc_std"]
        grouped[group_key][f"{row['dataset']}_f1_macro"] = row["f1_macro"]
        grouped[group_key][f"{row['dataset']}_f1_macro_std"] = row["f1_macro_std"]

    headers = ["feature_type", "model"] + [f"{dataset}_{metric}" for dataset, metric in key_order]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for feature_type in ["bow", "plm"]:
            for model_name in ["mlp", "gcn", "sage", "gat"]:
                row = grouped.get((feature_type, model_name), {"feature_type": feature_type, "model": model_name})
                writer.writerow(row)


def run_experiments(
    datasets: List[str],
    feature_types: List[str],
    model_names: List[str],
    root: Path,
    output_dir: Path,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    num_layers: int,
    heads: int,
    batch_size: int,
    seed: int,
    runs: int,
    plm_model: str,
    force_recompute_plm: bool,
) -> None:
    config = TrainConfig(
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        num_layers=num_layers,
        heads=heads,
        seed=seed,
    )
    rows: List[Dict[str, object]] = []

    for dataset_name in datasets:
        for feature_type in feature_types:
            for run_idx in range(runs):
                run_seed = seed + run_idx
                bundle = load_dataset(dataset_name, root=root, seed=run_seed)
                features = get_features(
                    bundle=bundle,
                    feature_type=feature_type,
                    root=root,
                    plm_model=plm_model,
                    batch_size=batch_size,
                    force_recompute_plm=force_recompute_plm,
                )

                for model_name in model_names:
                    model = build_model(
                        model_name=model_name,
                        in_channels=features.size(-1),
                        hidden_dim=hidden_dim,
                        out_channels=bundle.num_classes,
                        num_layers=num_layers,
                        dropout=dropout,
                        heads=heads,
                    )
                    scores = fit(
                        model=model,
                        data=bundle.data.clone(),
                        features=features.clone(),
                        split_idx=bundle.split_idx,
                        config=TrainConfig(**{**asdict(config), "seed": run_seed}),
                    )
                    test_scores = scores["test"]
                    row = {
                        "run": run_idx + 1,
                        "feature_type": feature_type,
                        "model": model_name,
                        "dataset": bundle.name,
                        "acc": round(test_scores["acc"], 4),
                        "f1_macro": round(test_scores["f1_macro"], 4),
                    }
                    rows.append(row)
                    print(row)

    summary_rows = _aggregate_rows(rows)

    _write_raw_results(output_dir / "results_raw.csv", rows)
    _write_summary_results(output_dir / "results_long.csv", summary_rows)
    _write_wide_results(output_dir / "results_wide.csv", summary_rows)

    with (output_dir / "run_config.txt").open("w", encoding="utf-8") as handle:
        for key, value in asdict(config).items():
            handle.write(f"{key}={value}\n")
        handle.write(f"runs={runs}\n")
        handle.write(f"plm_model={plm_model}\n")
        handle.write(f"datasets={','.join(datasets)}\n")
        handle.write(f"feature_types={','.join(feature_types)}\n")
        handle.write(f"models={','.join(model_names)}\n")
