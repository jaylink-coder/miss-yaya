"""Evaluate Yaya model on benchmarks and held-out data.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/yaya-1.5b/latest \
                               --model_config configs/model/yaya_1_5b.yaml \
                               --benchmarks mmlu,hellaswag
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer
from src.evaluation.evaluator import Evaluator
from src.data.dataset import TextDataset
from src.data.dataloader import create_dataloader
from src.training.checkpointing import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Evaluate Yaya model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--model_config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/yaya_tokenizer.model")
    parser.add_argument("--eval_data", type=str, default=None, help="Eval data path for perplexity")
    parser.add_argument("--benchmarks", type=str, default="", help="Comma-separated benchmark names")
    parser.add_argument("--benchmark_data_dir", type=str, default=None, help="Benchmark data directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    print("=" * 60)
    print("  Yaya Model Evaluation")
    print("=" * 60)

    # Load model config
    model_config = load_model_config(args.model_config)
    print(f"Model: {model_config.model_name}")

    # Initialize model
    print("Loading model...")
    model = YayaForCausalLM(model_config)

    # Load checkpoint
    ckpt_manager = CheckpointManager(save_dir=os.path.dirname(args.checkpoint))
    ckpt_manager.load(model, checkpoint_path=args.checkpoint)
    model = model.to(args.device)
    model.eval()

    # Load tokenizer
    tokenizer = YayaTokenizer(args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        benchmark_data_dir=args.benchmark_data_dir,
    )

    # Eval data for perplexity
    eval_dataloader = None
    if args.eval_data:
        eval_dataset = TextDataset(
            data_path=args.eval_data,
            max_seq_length=model_config.max_position_embeddings,
            split="eval",
        )
        eval_dataloader = create_dataloader(
            eval_dataset, batch_size=4, shuffle=False, distributed=False
        )

    # Parse benchmark names
    benchmark_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    # Run evaluation
    results = evaluator.full_evaluation(
        eval_dataloader=eval_dataloader,
        benchmark_names=benchmark_names or None,
        max_benchmark_samples=args.max_samples,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
