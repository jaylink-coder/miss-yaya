"""Convert Yaya checkpoints between formats.

Supports conversion to/from:
- PyTorch (.pt)
- SafeTensors (.safetensors)
- HuggingFace format

Usage:
    python scripts/convert_checkpoint.py \
        --input checkpoints/yaya-1.5b/checkpoint-00010000 \
        --output exports/yaya-1.5b-safetensors \
        --format safetensors
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM


def convert_to_safetensors(state_dict, output_path):
    """Save state dict in SafeTensors format."""
    try:
        from safetensors.torch import save_file
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, "model.safetensors")
        save_file(state_dict, filepath)
        print(f"Saved SafeTensors: {filepath}")
    except ImportError:
        print("safetensors package required: pip install safetensors")


def convert_to_pytorch(state_dict, output_path):
    """Save state dict in PyTorch format."""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, "model.pt")
    torch.save(state_dict, filepath)
    print(f"Saved PyTorch: {filepath}")


def load_from_safetensors(path):
    """Load state dict from SafeTensors format."""
    from safetensors.torch import load_file
    model_file = os.path.join(path, "model.safetensors")
    if os.path.isfile(path):
        model_file = path
    return load_file(model_file)


def main():
    parser = argparse.ArgumentParser(description="Convert Yaya checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--format", type=str, choices=["safetensors", "pytorch"], default="safetensors")
    parser.add_argument("--model_config", type=str, default=None, help="Model config (for validation)")
    args = parser.parse_args()

    print(f"Converting checkpoint: {args.input}")
    print(f"Output format: {args.format}")
    print(f"Output path: {args.output}")

    # Load input checkpoint
    model_path = os.path.join(args.input, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    elif args.input.endswith(".safetensors") or os.path.exists(
        os.path.join(args.input, "model.safetensors")
    ):
        state_dict = load_from_safetensors(args.input)
    else:
        state_dict = torch.load(args.input, map_location="cpu", weights_only=True)

    print(f"Loaded {len(state_dict)} tensors")

    # Validate against model config if provided
    if args.model_config:
        config = load_model_config(args.model_config)
        model = YayaForCausalLM(config)
        model.load_state_dict(state_dict)
        print("Validation passed: state dict matches model architecture")

    # Convert
    if args.format == "safetensors":
        convert_to_safetensors(state_dict, args.output)
    elif args.format == "pytorch":
        convert_to_pytorch(state_dict, args.output)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
