"""Export Yaya model checkpoint for sharing or deployment.

Creates a self-contained export directory with:
  - model.pt       (weights only, no optimizer state)
  - config.json    (model architecture config)
  - tokenizer.model (SentencePiece tokenizer)
  - README.md      (usage instructions)
  - generate.py    (standalone inference script)

Usage:
    # Export latest local checkpoint
    python scripts/export_model.py --output exports/yaya-125m-final

    # Export from HF Hub
    python scripts/export_model.py --token hf_xxx --output exports/yaya-125m-final

    # Export + quantize to int8 (492MB -> ~219MB)
    python scripts/export_model.py --output exports/yaya-125m-int8 --quantize

    # Export + push to HF Hub
    python scripts/export_model.py --output exports/yaya-125m-final --push --hub_repo Jaylink-coder/yaya-125m
"""

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(REPO_ROOT, "data/tokenizer/yaya_tokenizer.model")


def find_best_checkpoint():
    search_dirs = [
        "/kaggle/working/yaya-recovery-checkpoints",
        "/kaggle/working/yaya-dpo-checkpoints",
        "/kaggle/working/yaya-sft-checkpoints",
        os.path.join(REPO_ROOT, "checkpoints/yaya-125m-sft"),
    ]
    for d in search_dirs:
        ckpts = sorted(glob.glob(os.path.join(d, "checkpoint-*")))
        if ckpts:
            return ckpts[-1]
    return None


def load_checkpoint(ckpt_path):
    model_file = os.path.join(ckpt_path, "model.pt")
    meta_file  = os.path.join(ckpt_path, "metadata.json")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"model.pt not found in {ckpt_path}")
    state = torch.load(model_file, map_location="cpu")
    meta  = {}
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
    return state, meta


def extract_weights(state):
    if "model" in state:
        return state["model"]
    return {k: v for k, v in state.items()
            if not any(k.startswith(p) for p in ["optimizer", "scheduler", "scaler"])}


def write_generate_script(output_dir):
    script = '''\
"""Standalone Yaya chat. Requires: torch, sentencepiece, pyyaml."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.transformer import YayaTransformer
from src.model.config import ModelConfig
from src.tokenizer.tokenizer import YayaTokenizer
from src.inference.generator import TextGenerator, GenerationConfig


def load_yaya(export_dir=None):
    if export_dir is None:
        export_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(export_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig(**cfg_dict)
    tokenizer = YayaTokenizer(os.path.join(export_dir, "tokenizer.model"))
    model = YayaTransformer(cfg)
    weights = torch.load(os.path.join(export_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model, tokenizer


def main():
    model, tokenizer = load_yaya()
    gen = TextGenerator(model, tokenizer)
    cfg = GenerationConfig(max_new_tokens=256, temperature=0.7, repetition_penalty=1.5)
    print("Yaya-125M  (type quit to exit)")
    print("-" * 40)
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue
        print(f"Yaya: {gen.generate(user, config=cfg)}\\n")

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(output_dir, "generate.py"), "w", encoding="utf-8") as f:
        f.write(script)


def write_readme(output_dir, meta, quantized=False):
    step = meta.get("step", "?")
    loss = meta.get("loss", "?")
    readme = f"""\
# Yaya-125M Export

Step: {step} | Loss: {f"{loss:.4f}" if isinstance(loss, float) else loss}
{"Quantized: int8" if quantized else "Precision: float32"}

## Quick start
```bash
python generate.py
```

## Python usage
```python
from generate import load_yaya
from src.inference.generator import TextGenerator, GenerationConfig

model, tokenizer = load_yaya()
gen = TextGenerator(model, tokenizer)
cfg = GenerationConfig(max_new_tokens=256, temperature=0.7, repetition_penalty=1.5)
print(gen.generate("What is 2 + 2?", config=cfg))
```
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to checkpoint dir")
    parser.add_argument("--token",      help="HF token (download from Hub)")
    parser.add_argument("--output",     default="exports/yaya-125m-final")
    parser.add_argument("--quantize",   action="store_true")
    parser.add_argument("--push",       action="store_true")
    parser.add_argument("--hub_repo",   default="Jaylink-coder/yaya-125m")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Exporting to: {args.output}")

    ckpt_path = args.checkpoint
    if not ckpt_path and args.token:
        from scripts.hub_utils import pull_latest_checkpoint
        ckpt_path = pull_latest_checkpoint(args.hub_repo, os.path.join(args.output, "_tmp"), args.token)
    if not ckpt_path:
        ckpt_path = find_best_checkpoint()
    if not ckpt_path:
        print("ERROR: No checkpoint found.")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    state, meta = load_checkpoint(ckpt_path)
    weights = extract_weights(state)
    print(f"  Step: {meta.get('step','?')}  Loss: {meta.get('loss','?')}")

    import yaml
    with open(os.path.join(REPO_ROOT, "configs/model/yaya_125m.yaml")) as f:
        cfg_dict = yaml.safe_load(f)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    shutil.copy(TOKENIZER_PATH, os.path.join(args.output, "tokenizer.model"))

    if args.quantize:
        from src.model.transformer import YayaTransformer
        from src.model.config import ModelConfig
        model = YayaTransformer(ModelConfig(**cfg_dict))
        model.load_state_dict(weights, strict=False)
        model.eval()
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.save(model.state_dict(), os.path.join(args.output, "model.pt"))
    else:
        torch.save(weights, os.path.join(args.output, "model.pt"))

    size_mb = os.path.getsize(os.path.join(args.output, "model.pt")) / 1e6
    print(f"  model.pt: {size_mb:.0f} MB")

    write_generate_script(args.output)
    write_readme(args.output, meta, quantized=args.quantize)
    print(f"\nExport complete: {args.output}/")

    if args.push and args.token:
        from huggingface_hub import HfApi
        api = HfApi()
        for fname in ["model.pt", "config.json", "tokenizer.model", "README.md", "generate.py"]:
            fpath = os.path.join(args.output, fname)
            if os.path.exists(fpath):
                api.upload_file(path_or_fileobj=fpath, path_in_repo=f"export/{fname}",
                                repo_id=args.hub_repo, token=args.token)
                print(f"  Pushed: {fname}")


if __name__ == "__main__":
    main()
