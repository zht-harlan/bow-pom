import argparse
from pathlib import Path

from bow_plm_experiments.data import load_dataset
from bow_plm_experiments.features import build_and_save_plm_features, manual_text_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate offline PLM node features and save them into data/manual_features."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ogbn-arxiv", "cora", "pubmed", "amazon-photo"],
        help="Datasets whose text files will be encoded.",
    )
    parser.add_argument("--root", default="data", help="Dataset/text root directory.")
    parser.add_argument(
        "--output-dir",
        default="data/manual_features",
        help="Directory to save {dataset}_plm.pt files.",
    )
    parser.add_argument(
        "--plm-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model used for text encoding.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="Encoding device, e.g. cuda, cuda:0, or cpu. Defaults to auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets)
    print(
        f"[Start] generate_plm_features total_datasets={total} "
        f"model={args.plm_model} batch_size={args.batch_size}"
    )

    for index, dataset_name in enumerate(args.datasets, start=1):
        print(f"[Dataset] {index}/{total} dataset={dataset_name}")
        bundle = load_dataset(dataset_name, root=root, seed=args.seed)
        text_path = manual_text_path(root, bundle.name)
        if text_path is None:
            raise FileNotFoundError(
                f"No text file found for dataset '{bundle.name}'. "
                f"Expected one of {root / 'texts' / (bundle.name + '.jsonl')}, "
                f"{root / 'texts' / (bundle.name + '.csv')}, "
                f"or {root / 'texts' / (bundle.name + '.txt')}."
            )

        output_path = output_dir / f"{bundle.name}_plm.pt"
        if output_path.exists() and not args.overwrite:
            print(f"[Skip] dataset={bundle.name} output={output_path} already exists")
            continue

        print(
            f"[Encode] dataset={bundle.name} text={text_path} "
            f"nodes={bundle.data.num_nodes} output={output_path}"
        )
        features = build_and_save_plm_features(
            dataset_name=bundle.name,
            text_path=text_path,
            output_path=output_path,
            model_name=args.plm_model,
            batch_size=args.batch_size,
            expected_count=bundle.data.num_nodes,
            device=args.device,
        )
        print(
            f"[Done] dataset={bundle.name} output={output_path} "
            f"shape={tuple(features.shape)} dtype={features.dtype}"
        )

    print(f"[Finish] output_dir={output_dir}")


if __name__ == "__main__":
    main()
