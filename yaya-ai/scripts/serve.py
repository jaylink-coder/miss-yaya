"""Launch the Yaya model API server.

Usage:
    python scripts/serve.py --checkpoint checkpoints/yaya-1.5b/latest \
                            --model_config configs/model/yaya_1_5b.yaml \
                            --port 8000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer
from src.inference.generator import TextGenerator
from src.inference.server import create_app
from src.training.checkpointing import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Serve Yaya model API")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--model_config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/yaya_tokenizer.model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
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

    # Create and run app
    app = create_app(generator, model_name=model_config.model_name)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"  Model: {model_config.model_name}")
    print(f"  Device: {args.device}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
