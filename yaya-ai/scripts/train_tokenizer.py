"""Train the Yaya BPE tokenizer on a text corpus.

Usage:
    python scripts/train_tokenizer.py --input_dir data/raw/text \
                                      --vocab_size 64000 \
                                      --output_dir data/tokenizer
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer.trainer import TokenizerTrainer
from src.tokenizer.tokenizer import YayaTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Yaya tokenizer")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with training text files")
    parser.add_argument("--input_files", type=str, nargs="*", default=None, help="Specific input files")
    parser.add_argument("--vocab_size", type=int, default=64000, help="Vocabulary size")
    parser.add_argument("--output_dir", type=str, default="data/tokenizer", help="Output directory")
    parser.add_argument("--model_prefix", type=str, default="yaya_tokenizer", help="Model file prefix")
    parser.add_argument("--character_coverage", type=float, default=0.9995, help="Character coverage")
    args = parser.parse_args()

    print("=" * 60)
    print("  Yaya Tokenizer Training")
    print("=" * 60)

    trainer = TokenizerTrainer(
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        output_dir=args.output_dir,
        character_coverage=args.character_coverage,
    )

    model_path = trainer.train(
        input_files=args.input_files,
        input_dir=args.input_dir,
    )

    # Verify the trained tokenizer
    print("\nVerifying tokenizer...")
    tokenizer = YayaTokenizer(model_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    test_text = "Hello, I am Yaya, a multimodal AI model."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"  Test encode: '{test_text}' -> {len(tokens)} tokens")
    print(f"  Test decode: '{decoded}'")
    print(f"  Token IDs: {tokens[:20]}...")

    print("\nTokenizer training complete!")
    print(f"  Model saved to: {model_path}")


if __name__ == "__main__":
    main()
