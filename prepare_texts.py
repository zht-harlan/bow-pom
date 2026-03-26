import argparse
from pathlib import Path

from bow_plm_experiments.data import load_dataset
from bow_plm_experiments.text_prep import build_dataset_texts, write_jsonl_texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare node text files for PLM feature generation."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ogbn-arxiv", "cora", "pubmed", "amazon-photo"],
        help="Datasets to convert into text files.",
    )
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument(
        "--output-dir",
        default="data/texts",
        help="Directory to save {dataset}.jsonl text files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-k",
        type=int,
        default=128,
        help="Max active feature tokens kept when pseudo-text is built from node features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing text files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets)
    print(f"[Start] prepare_texts total_datasets={total} top_k={args.top_k}")

    for index, dataset_name in enumerate(args.datasets, start=1):
        print(f"[Dataset] {index}/{total} dataset={dataset_name}")
        bundle = load_dataset(dataset_name, root=root, seed=args.seed)
        output_path = output_dir / f"{bundle.name}.jsonl"
        if output_path.exists() and not args.overwrite:
            print(f"[Skip] dataset={bundle.name} output={output_path} already exists")
            continue

        texts = build_dataset_texts(bundle, top_k=args.top_k)
        write_jsonl_texts(texts, output_path)
        print(
            f"[Done] dataset={bundle.name} output={output_path} "
            f"num_texts={len(texts)}"
        )

    print(f"[Finish] output_dir={output_dir}")


if __name__ == "__main__":
    main()
