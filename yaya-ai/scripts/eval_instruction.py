"""Evaluate instruction-following quality of a Yaya checkpoint.

Runs a fixed set of test questions and checks if responses contain
expected content. Reports a quality score and per-question results.

Usage:
    python scripts/eval_instruction.py
    python scripts/eval_instruction.py --checkpoint checkpoints/yaya-125m-sft/checkpoint-00005000
    python scripts/eval_instruction.py --all_checkpoints checkpoints/yaya-125m-sft
"""

import argparse
import os
import sys
import re
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN
from src.training.checkpointing import CheckpointManager
from src.inference.generator import TextGenerator, GenerationConfig


# ── Test suite ──────────────────────────────────────────────────────────────
# Each entry: (question, [required_strings_in_answer (case-insensitive)])
TEST_SUITE = [
    # Arithmetic
    ("What is 2 + 2?",                    ["4"]),
    ("What is 10 times 5?",               ["50"]),
    ("What is 100 divided by 4?",         ["25"]),

    # Factual
    ("What is the capital of France?",    ["paris"]),
    ("What is the capital of Kenya?",     ["nairobi"]),
    ("How many days are in a week?",      ["7", "seven"]),
    ("What planet is closest to the Sun?",["mercury"]),

    # Definitions
    ("Define the word 'democracy'.",      ["people", "govern", "elect", "vote"]),
    ("What is gravity?",                  ["force", "mass", "attract", "pull"]),

    # Instructions
    ("List three primary colors.",        ["red", "blue", "yellow"]),
    ("Name the first three US presidents.",["washington", "adams", "jefferson"]),

    # Yaya identity
    ("What is your name?",                ["yaya"]),
    ("Are you an AI?",                    ["yes", "am", "ai", "assistant"]),

    # ── Compute (arithmetic — tests calculator tool learning) ──
    ("What is 15 + 27?",                  ["42"]),
    ("What is 9 times 8?",                ["72"]),
    ("What is 2 to the power of 8?",      ["256"]),

    # ── Reasoning (logic / word problems) ──
    ("A farmer has 20 sheep. Half die. How many remain?",  ["10"]),
    ("If today is Monday, what day is the day after tomorrow?",  ["wednesday"]),
    ("All birds can fly. Penguins are birds. Can penguins fly?",
     ["no", "cannot", "can't", "not"]),

    # ── Math reasoning ──
    ("What is 15% of 200?",               ["30"]),
    ("A train travels 90 km/h for 2 hours. How far does it travel?",  ["180"]),
    ("What is the area of a square with side 7?",  ["49"]),
]


def score_response(response: str, required: list) -> tuple[bool, str]:
    """Return (passed, reason).

    Passes if ANY required string is found in the response (case-insensitive).
    For single-entry lists this is an exact match requirement.
    For multi-entry lists (e.g. ["red", "blue", "yellow"]) at least one must appear.
    """
    low = response.lower()
    for r in required:
        if r.lower() in low:
            return True, f"found '{r}'"
    return False, f"missing all of {required}"


def evaluate_checkpoint(checkpoint_path: str, model_cfg_path: str, tok_path: str,
                        temperature: float = 0.3, max_tokens: int = 120) -> dict:
    """Load checkpoint and run the test suite. Returns result dict."""
    cfg = load_model_config(model_cfg_path)
    model = YayaForCausalLM(cfg)
    ckpt = CheckpointManager(save_dir=os.path.dirname(checkpoint_path))
    meta = ckpt.load(model, checkpoint_path=checkpoint_path)
    model.eval()

    tok = YayaTokenizer(tok_path)
    gen = TextGenerator(model, tok, device="cpu")
    gen_cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.5,   # Prevent degenerate token repetition
        do_sample=temperature > 0,
    )

    SYSTEM_PROMPT = (
        "You are Yaya, a helpful and honest AI assistant. "
        "You answer questions clearly and concisely."
    )

    results = []
    passed = 0
    for question, required in TEST_SUITE:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tok.format_chat(msgs) + "\n" + ASSISTANT_TOKEN + "\n"
        with torch.no_grad():
            output = gen.generate(prompt, gen_cfg)
        response = output[len(prompt):]
        for stop in ["</s>", ASSISTANT_TOKEN, "<|endoftext|>"]:
            response = response.split(stop)[0]
        response = response.strip()

        ok, reason = score_response(response, required)
        if ok:
            passed += 1
        results.append({
            "question": question,
            "response": response[:200],
            "passed": ok,
            "reason": reason,
        })

    score = passed / len(TEST_SUITE)
    step = meta.get("step", "?")
    loss = meta.get("loss", "?")
    return {
        "checkpoint": checkpoint_path,
        "step": step,
        "train_loss": loss,
        "score": score,
        "passed": passed,
        "total": len(TEST_SUITE),
        "results": results,
    }


def print_report(report: dict, verbose: bool = True):
    step = report["step"]
    loss = report["train_loss"]
    score = report["score"]
    passed = report["passed"]
    total = report["total"]
    print(f"\nStep {step} | train_loss={loss:.4f} | score={score:.0%} ({passed}/{total})")
    print("-" * 60)
    if verbose:
        for r in report["results"]:
            mark = "✓" if r["passed"] else "✗"
            print(f"  {mark} {r['question'][:50]:<50s}  {r['reason']}")
            if not r["passed"] and r["response"]:
                print(f"      → {r['response'][:100]!r}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        type=str, default=None)
    parser.add_argument("--all_checkpoints",   type=str, default=None,
                        help="Directory — evaluate all checkpoints in order")
    parser.add_argument("--model_config",      type=str,
                        default="configs/model/yaya_125m.yaml")
    parser.add_argument("--tokenizer",         type=str,
                        default="data/tokenizer/yaya_tokenizer.model")
    parser.add_argument("--temperature",       type=float, default=0.3)
    parser.add_argument("--max_tokens",        type=int, default=120)
    parser.add_argument("--quiet",             action="store_true")
    parser.add_argument("--save_json",         type=str, default=None)
    args = parser.parse_args()

    reports = []

    if args.all_checkpoints:
        # Find all checkpoint-XXXXXXXX dirs in sorted order
        d = args.all_checkpoints
        ckpts = sorted(
            [os.path.join(d, x) for x in os.listdir(d)
             if x.startswith("checkpoint-") and os.path.isdir(os.path.join(d, x))]
        )
        if not ckpts:
            print(f"No checkpoints found in {d}")
            return
        for ckpt in ckpts:
            print(f"Evaluating {ckpt}...", end="", flush=True)
            r = evaluate_checkpoint(ckpt, args.model_config, args.tokenizer,
                                     args.temperature, args.max_tokens)
            reports.append(r)
            print_report(r, verbose=not args.quiet)
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            # Auto-find best available checkpoint — 125M first, then math fallbacks
            for d in ["checkpoints/yaya-125m-sft",
                      "checkpoints/yaya-125m-reasoning",
                      "checkpoints/yaya-125m",
                      "checkpoints/yaya-125m-reasoning",
                      "checkpoints/yaya-125m"]:
                latest = os.path.join(d, "latest")
                if os.path.exists(latest):
                    with open(latest) as f:
                        name = f.read().strip()
                    ckpt = os.path.join(d, name)
                    break
        if ckpt is None:
            print("No checkpoint found.")
            return
        r = evaluate_checkpoint(ckpt, args.model_config, args.tokenizer,
                                 args.temperature, args.max_tokens)
        reports.append(r)
        print_report(r, verbose=not args.quiet)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"Results saved to {args.save_json}")

    # Summary line for multi-checkpoint run
    if len(reports) > 1:
        print("\nSummary:")
        for r in reports:
            bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
            print(f"  Step {r['step']:6d} [{bar}] {r['score']:.0%}")


if __name__ == "__main__":
    main()
