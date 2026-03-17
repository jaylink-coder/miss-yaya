"""Download real training data for Yaya pretraining.

Downloads WikiText-103 (train ~500MB text) and WikiText-2 (train ~2MB text)
from HuggingFace datasets, then saves as plain text files for the data pipeline.

Usage:
    python scripts/download_data.py --dataset wikitext-2 --output_dir data/raw/web
    python scripts/download_data.py --dataset wikitext-103 --output_dir data/raw/web
    python scripts/download_data.py --dataset tinystories --output_dir data/raw/web
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_wikitext(name: str, output_dir: str):
    from datasets import load_dataset

    config = "wikitext-2-raw-v1" if name == "wikitext-2" else "wikitext-103-raw-v1"
    print(f"Downloading {name} ({config})...")
    ds = load_dataset("wikitext", config, trust_remote_code=True)

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        out_path = os.path.join(output_dir, f"{name}_{split}.txt")
        print(f"  Writing {split} -> {out_path}")
        total_chars = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                text = row["text"].strip()
                if text:
                    f.write(text + "\n\n")
                    total_chars += len(text)
        print(f"    {total_chars:,} chars written")


def download_tinystories(output_dir: str):
    from datasets import load_dataset

    print("Downloading TinyStories (train split only)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tinystories_train.txt")

    print(f"  Writing train -> {out_path}")
    total_chars = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if text:
                f.write(text + "\n\n")
                total_chars += len(text)
    print(f"  {total_chars:,} chars written")


def main():
    parser = argparse.ArgumentParser(description="Download training data for Yaya")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2",
        choices=["wikitext-2", "wikitext-103", "tinystories"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/web",
        help="Output directory for raw text files",
    )
    args = parser.parse_args()

    if args.dataset in ("wikitext-2", "wikitext-103"):
        download_wikitext(args.dataset, args.output_dir)
    elif args.dataset == "tinystories":
        download_tinystories(args.output_dir)

    print(f"\nDone. Raw text saved to: {args.output_dir}")
    print("Next step: run scripts/prepare_data.py to tokenize.")


if __name__ == "__main__":
    main()
