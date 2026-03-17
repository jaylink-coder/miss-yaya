"""Export Yaya model to various deployment formats.

Usage:
    # SafeTensors
    python scripts/export_model.py \
        --checkpoint checkpoints/yaya-1.5b/latest \
        --model_config configs/model/yaya_1_5b.yaml \
        --format safetensors --output exports/yaya-1.5b-st

    # ONNX
    python scripts/export_model.py \
        --checkpoint checkpoints/yaya-1.5b/latest \
        --model_config configs/model/yaya_1_5b.yaml \
        --format onnx --output exports/yaya-1.5b.onnx

    # llama.cpp (GGUF prep)
    python scripts/export_model.py \
        --checkpoint checkpoints/yaya-1.5b/latest \
        --model_config configs/model/yaya_1_5b.yaml \
        --format gguf --output exports/yaya-1.5b-gguf
"""

import argparse
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.training.checkpointing import CheckpointManager
from src.inference.export import (
    export_safetensors,
    export_onnx,
    export_for_llama_cpp,
)
from src.utils.io_utils import count_parameters, format_num


def main():
    parser = argparse.ArgumentParser(description="Export Yaya model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--format", type=str, required=True,
                        choices=["safetensors", "onnx", "gguf"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    args = parser.parse_args()

    # Load model
    config = load_model_config(args.model_config)
    model = YayaForCausalLM(config)
    ckpt_manager = CheckpointManager(save_dir=os.path.dirname(args.checkpoint))
    ckpt_manager.load(model, checkpoint_path=args.checkpoint)
    model.eval()

    params = count_parameters(model)
    print(f"Model: {config.model_name}")
    print(f"Parameters: {format_num(params['total'])}")
    print(f"Export format: {args.format}")
    print()

    config_dict = asdict(config)

    if args.format == "safetensors":
        export_safetensors(model, args.output, model_config=config_dict)

    elif args.format == "onnx":
        export_onnx(
            model, args.output,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_position_embeddings,
        )

    elif args.format == "gguf":
        export_for_llama_cpp(
            model, args.output,
            model_config=config_dict,
            tokenizer_path=args.tokenizer_path,
        )

    print("\nExport complete!")


if __name__ == "__main__":
    main()
