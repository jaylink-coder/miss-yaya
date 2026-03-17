"""Data preparation pipeline — process raw text into tokenized training data.

Usage:
    python scripts/prepare_data.py --input_dir data/raw/text \
                                   --output_dir data/processed/train \
                                   --tokenizer_path data/tokenizer/yaya_tokenizer.model
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer.tokenizer import YayaTokenizer
from src.data.processing import DataProcessor, TextCleaner, TextFilter, Deduplicator


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with raw text files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokenized data")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to trained tokenizer")
    parser.add_argument("--file_pattern", type=str, default="*.txt", help="File glob pattern")
    parser.add_argument("--shard_size", type=int, default=100_000_000, help="Tokens per shard")
    parser.add_argument("--min_doc_length", type=int, default=100, help="Minimum document length")
    parser.add_argument("--max_doc_length", type=int, default=1_000_000, help="Maximum document length")
    args = parser.parse_args()

    print("=" * 60)
    print("  Yaya Data Preparation Pipeline")
    print("=" * 60)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = YayaTokenizer(args.tokenizer_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Create processing pipeline
    cleaner = TextCleaner(
        min_line_length=10,
        remove_urls=True,
        remove_emails=True,
        normalize_whitespace=True,
    )

    text_filter = TextFilter(
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        max_special_char_ratio=0.3,
        min_alpha_ratio=0.5,
    )

    deduplicator = Deduplicator()

    processor = DataProcessor(
        tokenizer=tokenizer,
        cleaner=cleaner,
        text_filter=text_filter,
        deduplicator=deduplicator,
    )

    # Process data
    print(f"\nProcessing files from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shard size: {args.shard_size:,} tokens")
    print()

    shard_paths = processor.process_directory(
        input_dir=args.input_dir,
        output_path=args.output_dir,
        file_pattern=args.file_pattern,
        shard_size=args.shard_size,
    )

    print(f"\nData preparation complete!")
    print(f"  Output shards: {len(shard_paths)}")
    print(f"  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
