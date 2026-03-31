"""Launch the Yaya model API server.

Usage:
    python scripts/serve.py --checkpoint checkpoints/yaya-1.5b/latest \
                            --model_config configs/model/yaya_1_5b.yaml \
                            --port 8000
"""

import argparse
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer
from src.inference.generator import TextGenerator
from src.inference.server import create_app
from src.training.checkpointing import CheckpointManager


DEFAULT_CHECKPOINT_DIRS = [
    "checkpoints/yaya-125m-sft",
    "checkpoints/yaya-125m-reasoning",
    "checkpoints/yaya-125m",
]


def _find_latest_checkpoint():
    for d in DEFAULT_CHECKPOINT_DIRS:
        latest = os.path.join(d, "latest")
        if os.path.exists(latest):
            with open(latest) as f:
                name = f.read().strip()
            path = os.path.join(d, name)
            if os.path.isdir(path):
                return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Serve Yaya model API")
    parser.add_argument("--checkpoint",  type=str, default=None, help="Checkpoint path (auto-detected)")
    parser.add_argument("--model_config", type=str, default="configs/model/yaya_125m.yaml")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/yaya_tokenizer.model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or _find_latest_checkpoint()
    if checkpoint is None:
        print("ERROR: No checkpoint found. Train a model first.")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {checkpoint} on {device}...")
    model_config = load_model_config(args.model_config)
    model = YayaForCausalLM(model_config)

    ckpt_manager = CheckpointManager(save_dir=os.path.dirname(checkpoint))
    ckpt_manager.load(model, checkpoint_path=checkpoint)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = YayaTokenizer(args.tokenizer_path)

    # Create generator
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    # Create and run app
    app = create_app(generator, model_name=model_config.model_name)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"  Model: {model_config.model_name}")
    print(f"  Device: {device}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
