import argparse
from pathlib import Path

from bow_plm_experiments.runner import run_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BoW/PLM feature experiments on graph datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ogbn-arxiv", "pubmed", "children", "history", "photo"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=["bow", "plm"],
        choices=["bow", "plm"],
        help="Feature families to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "gcn", "sage", "gat", "sgc", "jknet", "appnp"],
        choices=["mlp", "gcn", "sage", "gat", "sgc", "jknet", "appnp"],
        help="Models to evaluate.",
    )
    parser.add_argument("--root", default="数据集", help="Dataset/cache root directory.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save CSV summaries.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--bow-max-features",
        type=int,
        default=2048,
        help="Fallback BoW vocabulary size when a dataset does not already provide data.x.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs with different random seeds.",
    )
    parser.add_argument(
        "--plm-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model used to encode raw text when cached PLM features are absent.",
    )
    parser.add_argument(
        "--force-recompute-plm",
        action="store_true",
        help="Ignore cached PLM embeddings and recompute them.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiments(
        datasets=args.datasets,
        feature_types=args.feature_types,
        model_names=args.models,
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        num_layers=args.num_layers,
        heads=args.heads,
        batch_size=args.batch_size,
        bow_max_features=args.bow_max_features,
        seed=args.seed,
        runs=args.runs,
        plm_model=args.plm_model,
        force_recompute_plm=args.force_recompute_plm,
    )
