"""Quantize Yaya checkpoint to int8 for fast local inference.

Reduces model size from ~493MB to ~130MB and speeds up CPU inference 3-4x.
Uses PyTorch dynamic quantization (no calibration data needed).

Usage:
    python scripts/quantize.py --checkpoint checkpoints/yaya-125m-sft/checkpoint-00032500
    python scripts/quantize.py --checkpoint checkpoints/yaya-125m-sft/checkpoint-00032500 --test
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN
from src.inference.generator import TextGenerator, GenerationConfig


def quantize_model(model):
    """Apply dynamic int8 quantization to linear layers."""
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def load_full(checkpoint_path):
    model_config = load_model_config("configs/model/yaya_125m.yaml")
    model = YayaForCausalLM(model_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def model_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024


def time_inference(model, tokenizer, n=3):
    gen = TextGenerator(model, tokenizer, device="cpu")
    cfg = GenerationConfig(max_new_tokens=30, temperature=0.7, top_p=0.9,
                           do_sample=True, repetition_penalty=1.5)
    SYSTEM = "You are Yaya, a helpful AI assistant."
    questions = ["What is 2 + 2?", "What is the capital of France?", "Who are you?"]
    t0 = time.time()
    answers = []
    for q in questions[:n]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        prompt = tokenizer.format_chat(msgs) + "\n" + ASSISTANT_TOKEN + "\n"
        ans = gen.generate(prompt, config=cfg).strip()
        answers.append((q, ans))
    elapsed = time.time() - t0
    return elapsed, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (containing model.pt)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output path for quantized model (default: <checkpoint>/model_int8.pt)")
    parser.add_argument("--test",       action="store_true",
                        help="Run speed comparison after quantizing")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint.rstrip("/\\")
    model_path = os.path.join(ckpt_dir, "model.pt")
    out_path = args.output or os.path.join(ckpt_dir, "model_int8.pt")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = load_full(model_path)
    orig_size = model_size_mb(model_path)
    print(f"Original size: {orig_size:.1f} MB")

    print("Quantizing to int8...")
    t0 = time.time()
    q_model = quantize_model(model)
    elapsed = time.time() - t0
    print(f"Quantization took {elapsed:.1f}s")

    print(f"Saving quantized model to {out_path}...")
    torch.save(q_model.state_dict(), out_path)
    q_size = model_size_mb(out_path)
    print(f"Quantized size: {q_size:.1f} MB  ({q_size/orig_size*100:.0f}% of original)")
    print(f"Compression: {orig_size/q_size:.1f}x smaller")

    if args.test:
        tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
        print("\n--- Speed test ---")

        print("Full model (float32):")
        t_full, answers_full = time_inference(model, tokenizer)
        print(f"  Time: {t_full:.2f}s for 3 questions")
        for q, a in answers_full:
            print(f"  Q: {q}  A: {a[:60]!r}")

        print("\nQuantized model (int8):")
        # Load quantized model for testing
        q_model2 = quantize_model(load_full(model_path))
        t_q, answers_q = time_inference(q_model2, tokenizer)
        print(f"  Time: {t_q:.2f}s for 3 questions")
        for q, a in answers_q:
            print(f"  Q: {q}  A: {a[:60]!r}")

        speedup = t_full / t_q if t_q > 0 else 0
        print(f"\nSpeedup: {speedup:.1f}x faster")

    print(f"\nDone. Use model_int8.pt for fast local inference.")
    print(f"Load with: torch.load('{out_path}', map_location='cpu')")


if __name__ == "__main__":
    main()
