"""Build training dataset — full data engine pipeline.

Orchestrates: download → quality filter → tokenize → shard

Usage:
    # Full pipeline (all datasets)
    python scripts/build_dataset.py --config configs/data/sources.yaml

    # Test mode (small sample)
    python scripts/build_dataset.py --config configs/data/sources.yaml --test --max_samples 1000

    # Specific categories only
    python scripts/build_dataset.py --config configs/data/sources.yaml --categories web_text,code

    # Skip download (process already-downloaded data)
    python scripts/build_dataset.py --config configs/data/sources.yaml --skip_download

    # Skip processing (tokenize already-processed data)
    python scripts/build_dataset.py --config configs/data/sources.yaml --skip_download --skip_process
"""

import argparse
import json
import os
import sys
import time
import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def step_header(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_download(config: dict, args):
    """Step 1: Download datasets."""
    step_header("STEP 1/3 — Download Datasets")

    from src.data.downloader import DownloadManager

    storage = config.get("storage", {})
    raw_dir = storage.get("raw_dir", "data/raw")
    cache_dir = storage.get("cache_dir", "data/cache")

    manager = DownloadManager(
        output_dir=raw_dir,
        cache_dir=cache_dir,
        max_workers=4,
    )

    datasets = manager.load_sources_config(args.config)

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    max_samples = args.max_samples if args.test else None

    results = manager.download_all(
        datasets,
        categories=categories,
        max_samples=max_samples,
        force=args.force,
    )

    print(f"\nDownloaded {len(results)} datasets")
    return results


def run_process(config: dict, args):
    """Step 2: Quality filter all downloaded data."""
    step_header("STEP 2/3 — Quality Filtering")

    from src.data.quality import DataQualityPipeline
    from src.data.processing import TextCleaner

    storage = config.get("storage", {})
    raw_dir = storage.get("raw_dir", "data/raw")
    processed_dir = storage.get("processed_dir", "data/processed")
    proc_config = config.get("processing", {})

    pipeline = DataQualityPipeline(
        language=proc_config.get("language", "en"),
        quality_threshold=proc_config.get("quality_threshold", 0.5),
        dedup_threshold=proc_config.get("minhash_threshold", 0.8),
        min_doc_length=proc_config.get("min_doc_length", 100),
        max_doc_length=proc_config.get("max_doc_length", 1000000),
        remove_pii=proc_config.get("remove_pii", True),
        minhash_num_perm=proc_config.get("minhash_num_perm", 128),
    )

    cleaner = TextCleaner(
        remove_urls=proc_config.get("remove_urls", True),
    )

    # Find all downloaded JSONL files
    input_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.jsonl"), recursive=True))
    if not input_files:
        print(f"No JSONL files found in {raw_dir}")
        return

    os.makedirs(processed_dir, exist_ok=True)

    print(f"Processing {len(input_files)} files from {raw_dir}")
    print(f"Output: {processed_dir}")
    print()

    total_input = 0
    total_output = 0

    for file_idx, filepath in enumerate(input_files):
        rel_path = os.path.relpath(filepath, raw_dir)
        # Preserve category subdirectory structure
        output_subdir = os.path.dirname(rel_path)
        output_dir = os.path.join(processed_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, os.path.basename(filepath))

        print(f"  [{file_idx + 1}/{len(input_files)}] {rel_path}", end="", flush=True)

        file_input = 0
        file_output = 0

        with open(filepath, "r", encoding="utf-8", errors="ignore") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                except (json.JSONDecodeError, KeyError):
                    continue

                file_input += 1

                # Clean
                text = cleaner.clean(text)
                if not text:
                    continue

                # Quality pipeline
                filtered_text = pipeline.process_document(text)
                if filtered_text is None:
                    continue

                record["text"] = filtered_text
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                file_output += 1

        total_input += file_input
        total_output += file_output
        rate = file_output / max(file_input, 1) * 100
        print(f" -> {file_output:,}/{file_input:,} docs ({rate:.0f}% pass)")

    # Print stats
    stats = pipeline.get_stats()
    print(f"\nQuality pipeline summary:")
    print(f"  Input docs: {total_input:,}")
    print(f"  Output docs: {total_output:,}")
    print(f"  Pass rate: {total_output / max(total_input, 1) * 100:.1f}%")
    print(f"  Dedup stats: {stats.get('dedup_stats', {})}")
    if "pii_stats" in stats:
        print(f"  PII removed: {stats['pii_stats']}")

    # Save stats
    stats_dir = storage.get("stats_dir", "data/stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "processing_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: {stats_path}")


def run_tokenize(config: dict, args):
    """Step 3: Tokenize processed data into shards."""
    step_header("STEP 3/3 — Tokenize into Training Shards")

    from src.data.tokenize_corpus import CorpusTokenizer

    storage = config.get("storage", {})
    processed_dir = storage.get("processed_dir", "data/processed")
    tok_config = config.get("tokenization", {})

    tokenizer_path = tok_config.get("tokenizer_path", "data/tokenizer/yaya_tokenizer.model")
    shard_size = tok_config.get("shard_size", 100_000_000)
    num_workers = tok_config.get("num_workers", 4)

    # Output dir for shards
    shard_dir = os.path.join(storage.get("processed_dir", "data/processed"), "shards")

    tokenizer = CorpusTokenizer(
        tokenizer_path=tokenizer_path,
        output_dir=shard_dir,
        shard_size=shard_size,
        num_workers=num_workers,
    )

    # Tokenize training data
    train_stats = tokenizer.tokenize_directory(
        input_dir=processed_dir,
        file_pattern="*.jsonl",
        split_name="train",
    )

    # Create eval split
    eval_stats = tokenizer.create_eval_split(
        input_dir=processed_dir,
        file_pattern="*.jsonl",
    )

    print(f"\nTokenization complete:")
    print(f"  Train tokens: {train_stats['total_tokens']:,}")
    print(f"  Eval tokens: {eval_stats['total_tokens']:,}")
    print(f"  Train shards: {train_stats['shards']}")


def main():
    parser = argparse.ArgumentParser(description="Build Yaya training dataset")
    parser.add_argument("--config", type=str, default="configs/data/sources.yaml",
                        help="Path to data sources config")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories to process")
    parser.add_argument("--test", action="store_true",
                        help="Test mode with small samples")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max samples per dataset in test mode")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step")
    parser.add_argument("--skip_process", action="store_true",
                        help="Skip processing step")
    parser.add_argument("--skip_tokenize", action="store_true",
                        help="Skip tokenization step")
    args = parser.parse_args()

    start = time.time()
    config = load_config(args.config)

    print("Yaya AI — Data Engine")
    print(f"Config: {args.config}")
    if args.test:
        print(f"TEST MODE: max {args.max_samples} samples per dataset")
    print()

    # Step 1: Download
    if not args.skip_download:
        run_download(config, args)
    else:
        print("Skipping download step")

    # Step 2: Process
    if not args.skip_process:
        run_process(config, args)
    else:
        print("Skipping processing step")

    # Step 3: Tokenize
    if not args.skip_tokenize:
        run_tokenize(config, args)
    else:
        print("Skipping tokenization step")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  DATA ENGINE COMPLETE — {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
