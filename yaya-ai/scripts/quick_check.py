"""Quick 5-question sanity check for Yaya — runs in <30s on CPU.

Tests that the model can produce correct direct answers for the most
basic questions. Use after every checkpoint or code change to catch
regressions before running the full benchmark.

Usage:
    python scripts/quick_check.py --checkpoint checkpoints/.../checkpoint-XXXXX
    python scripts/quick_check.py   # auto-finds latest local checkpoint
"""

import sys, os, argparse, glob, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN
from src.inference.generator import TextGenerator, GenerationConfig

CHECKS = [
    ("2 + 2",      "What is 2 + 2?",          ["4"]),
    ("capital",    "What is the capital of France?", ["paris"]),
    ("sky color",  "What color is the sky?",   ["blue"]),
    ("identity",   "What is your name?",       ["yaya"]),
    ("12 x 12",    "What is 12 x 12?",         ["144"]),
]

SYSTEM = "You are Yaya, a helpful AI assistant. Answer concisely."

DEFAULT_CKPT_DIRS = [
    "checkpoints/yaya-125m-dpo",
    "checkpoints/yaya-125m-sft",
    "checkpoints/yaya-125m-reasoning",
    "checkpoints/yaya-125m",
]


def find_latest_checkpoint():
    for d in DEFAULT_CKPT_DIRS:
        if os.path.isdir(d):
            ckpts = sorted(glob.glob(os.path.join(d, "checkpoint-*")))
            if ckpts:
                return ckpts[-1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_config", type=str, default="configs/model/yaya_125m.yaml")
    args = parser.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint()
    if not ckpt:
        print("ERROR: No checkpoint found.")
        sys.exit(1)

    model_path = os.path.join(ckpt, "model.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        sys.exit(1)

    print(f"Quick check: {os.path.basename(ckpt)}")
    print("-" * 40)

    t0 = time.time()
    cfg = load_model_config(args.model_config)
    model = YayaForCausalLM(cfg)
    raw = torch.load(model_path, map_location="cpu", weights_only=False)
    weights = raw.get("model", raw.get("model_state_dict", raw))
    model.load_state_dict(weights, strict=False)
    model.eval()

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    gen = TextGenerator(model, tokenizer, device="cpu")
    gen_cfg = GenerationConfig(max_new_tokens=30, temperature=0.1, repetition_penalty=1.5, do_sample=False)

    passed = 0
    for label, question, accepted in CHECKS:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        prompt = tokenizer.format_chat(msgs) + "\n" + ASSISTANT_TOKEN + "\n"
        answer = gen.generate(prompt, config=gen_cfg).strip()
        ok = any(a.lower() in answer.lower() for a in accepted)
        if ok:
            passed += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {question}")
        print(f"         → {answer[:60]!r}")

    elapsed = time.time() - t0
    pct = passed / len(CHECKS) * 100
    print("-" * 40)
    print(f"  Score: {passed}/{len(CHECKS)} ({pct:.0f}%)   ({elapsed:.1f}s)")

    # Exit code: 0 if ≥60%, 1 if worse
    sys.exit(0 if passed >= 3 else 1)


if __name__ == "__main__":
    main()
