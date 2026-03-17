"""Interactive text generation with the Yaya model.

Usage:
    python scripts/generate.py --checkpoint checkpoints/yaya-1.5b/latest \
                               --model_config configs/model/yaya_1_5b.yaml \
                               --prompt "Once upon a time"

    # Interactive mode
    python scripts/generate.py --checkpoint checkpoints/yaya-1.5b/latest \
                               --model_config configs/model/yaya_1_5b.yaml \
                               --interactive
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer
from src.inference.generator import TextGenerator, GenerationConfig
from src.training.checkpointing import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Generate text with Yaya model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--model_config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/yaya_tokenizer.model")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model_config = load_model_config(args.model_config)
    model = YayaForCausalLM(model_config)

    ckpt_manager = CheckpointManager(save_dir=os.path.dirname(args.checkpoint))
    ckpt_manager.load(model, checkpoint_path=args.checkpoint)
    model = model.to(args.device)
    model.eval()

    # Load tokenizer
    tokenizer = YayaTokenizer(args.tokenizer_path)

    # Create generator
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=args.device)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    print(f"Model loaded: {model_config.model_name}")
    print(f"  Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print()

    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit.")
        print("-" * 40)
        while True:
            try:
                prompt = input("\nYou: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue

                print("\nYaya: ", end="", flush=True)
                for token in generator.stream_generate(prompt, gen_config):
                    print(token, end="", flush=True)
                print()

            except KeyboardInterrupt:
                print("\nExiting.")
                break
    elif args.prompt:
        # Single prompt mode
        print(f"Prompt: {args.prompt}")
        print("-" * 40)
        output = generator.generate(args.prompt, gen_config)
        print(output)
    else:
        print("Provide --prompt or use --interactive mode.")


if __name__ == "__main__":
    main()
