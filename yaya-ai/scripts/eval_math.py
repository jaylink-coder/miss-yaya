"""
eval_math.py
============
Evaluates Yaya's math knowledge across all 8 curriculum stages.
Runs 20 targeted questions and reports a score per stage + overall.

Usage:
    python scripts/eval_math.py
    python scripts/eval_math.py --checkpoint checkpoints/yaya-tiny-math-stage4/checkpoint-00003000
    python scripts/eval_math.py --stage 1  # Only stage 1 questions
"""

import sys
import json
import argparse
import re
import time
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.generator import TextGenerator, GenerationConfig
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN
from src.utils.config import load_model_config
from src.training.checkpointing import CheckpointManager

# ──────────────────────────────────────────────────────────
# Eval questions — 20 total, 2-3 per stage
# ──────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    # Stage 1 — Arithmetic
    {
        "stage": 1, "topic": "Multiplication",
        "question": "What is 47 × 83?",
        "expected": "3901",
        "keywords": ["3901"],
    },
    {
        "stage": 1, "topic": "Square root",
        "question": "What is the square root of 144?",
        "expected": "12",
        "keywords": ["12"],
    },
    {
        "stage": 1, "topic": "Order of operations",
        "question": "Calculate 15 + 8 × 3.",
        "expected": "39",
        "keywords": ["39"],
    },
    # Stage 2 — Fractions / Decimals / Percentages
    {
        "stage": 2, "topic": "Fraction addition",
        "question": "What is 3/4 + 1/6?",
        "expected": "11/12",
        "keywords": ["11/12"],
    },
    {
        "stage": 2, "topic": "Percentage",
        "question": "What is 35% of 240?",
        "expected": "84",
        "keywords": ["84"],
    },
    {
        "stage": 2, "topic": "Decimal to fraction",
        "question": "Convert 0.625 to a fraction.",
        "expected": "5/8",
        "keywords": ["5/8"],
    },
    # Stage 3 — Pre-Algebra
    {
        "stage": 3, "topic": "Expression evaluation",
        "question": "If x = 5, what is 4x + 7?",
        "expected": "27",
        "keywords": ["27"],
    },
    {
        "stage": 3, "topic": "Two-step equation",
        "question": "Solve for x: 3x - 9 = 12.",
        "expected": "x = 7",
        "keywords": ["7"],
    },
    # Stage 4 — Algebra
    {
        "stage": 4, "topic": "Quadratic",
        "question": "What are the solutions to x\u00b2 - 5x + 6 = 0?",
        "expected": "x = 2 and x = 3",
        "keywords": ["2", "3"],
    },
    {
        "stage": 4, "topic": "Slope",
        "question": "Find the slope of the line through (1, 2) and (4, 8).",
        "expected": "2",
        "keywords": ["2"],
    },
    {
        "stage": 4, "topic": "Logarithm",
        "question": "What is log base 2 of 32?",
        "expected": "5",
        "keywords": ["5"],
    },
    # Stage 5 — Geometry
    {
        "stage": 5, "topic": "Circle area",
        "question": "What is the area of a circle with radius 7? Use \u03c0 \u2248 3.14159.",
        "expected": "153.94",
        "keywords": ["153"],
    },
    {
        "stage": 5, "topic": "Pythagorean theorem",
        "question": "A right triangle has legs 5 and 12. What is the hypotenuse?",
        "expected": "13",
        "keywords": ["13"],
    },
    {
        "stage": 5, "topic": "Supplementary angles",
        "question": "Two angles are supplementary. One is 65\u00b0. What is the other?",
        "expected": "115",
        "keywords": ["115"],
    },
    # Stage 6 — Statistics
    {
        "stage": 6, "topic": "Mean",
        "question": "Find the mean of: 4, 7, 2, 9, 13.",
        "expected": "7",
        "keywords": ["7"],
    },
    {
        "stage": 6, "topic": "Probability",
        "question": "What is the probability of rolling a 4 on a fair six-sided die?",
        "expected": "1/6",
        "keywords": ["1/6"],
    },
    # Stage 7 — Word Problems
    {
        "stage": 7, "topic": "Speed/Distance/Time",
        "question": "A car travels at 90 km/h for 3 hours. How far does it travel?",
        "expected": "270 km",
        "keywords": ["270"],
    },
    {
        "stage": 7, "topic": "Simple interest",
        "question": "What is the simple interest on $1000 at 8% per year for 2 years?",
        "expected": "$160",
        "keywords": ["160"],
    },
    # Stage 8 — Calculus
    {
        "stage": 8, "topic": "Derivative",
        "question": "Find the derivative of f(x) = 3x\u2074.",
        "expected": "12x\u00b3",
        "keywords": ["12"],
    },
    {
        "stage": 8, "topic": "Limit",
        "question": "What is the limit of 1/x as x approaches infinity?",
        "expected": "0",
        "keywords": ["0"],
    },
]

STAGE_NAMES = {
    1: "Arithmetic",
    2: "Fractions/Decimals",
    3: "Pre-Algebra",
    4: "Algebra",
    5: "Geometry",
    6: "Statistics",
    7: "Word Problems",
    8: "Calculus",
}


def find_best_checkpoint() -> str:
    """Find the best available math checkpoint."""
    # Prefer highest stage
    for stage in range(8, 0, -1):
        ckpt_dir = Path(f"checkpoints/yaya-tiny-math-stage{stage}")
        latest = ckpt_dir / "latest"
        if latest.exists():
            return str(ckpt_dir / latest.read_text().strip())

    # Fall back to filtered SFT
    latest = Path("checkpoints/yaya-tiny-sft-filtered/latest")
    if latest.exists():
        return str(Path("checkpoints/yaya-tiny-sft-filtered") / latest.read_text().strip())

    # Fall back to pretrain
    return "checkpoints/yaya-tiny/checkpoint-00010000"


def check_answer(response: str, keywords: list) -> bool:
    """Check if all keywords appear in the response."""
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


def run_eval(generator, tokenizer, questions, verbose=True):
    results = []

    for i, q in enumerate(questions):
        prompt = q["question"]
        if verbose:
            print(f"\n[{i+1}/{len(questions)}] Stage {q['stage']} — {q['topic']}")
            print(f"  Q: {prompt}")

        try:
            msgs = [
                {"role": "system", "content": "You are Yaya, a helpful math assistant. Answer clearly and show your work."},
                {"role": "user", "content": prompt},
            ]
            formatted = tokenizer.format_chat(msgs) + "\n" + ASSISTANT_TOKEN + "\n"
            config = GenerationConfig(
                max_new_tokens=120,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.5,
            )
            import torch
            with torch.no_grad():
                raw = generator.generate(prompt=formatted, config=config)
            response = raw[len(formatted):]
            for stop in ["</s>", ASSISTANT_TOKEN, "<|endoftext|>"]:
                response = response.split(stop)[0]
            response = response.strip()
            passed = check_answer(response, q["keywords"])
        except Exception as e:
            response = f"[ERROR: {e}]"
            passed = False

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  A: {response[:200]}")
            print(f"  Expected: {q['expected']}  → [{status}]")

        results.append({
            "stage": q["stage"],
            "topic": q["topic"],
            "question": prompt,
            "expected": q["expected"],
            "response": response,
            "passed": passed,
        })

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("MATH EVAL SUMMARY")
    print("=" * 60)

    # Per-stage results
    stage_scores = {}
    for r in results:
        s = r["stage"]
        if s not in stage_scores:
            stage_scores[s] = {"pass": 0, "total": 0}
        stage_scores[s]["total"] += 1
        if r["passed"]:
            stage_scores[s]["pass"] += 1

    for stage in sorted(stage_scores):
        d = stage_scores[stage]
        pct = d["pass"] / d["total"] * 100
        bar = "█" * d["pass"] + "░" * (d["total"] - d["pass"])
        print(f"  Stage {stage} ({STAGE_NAMES[stage]:<20}): {d['pass']}/{d['total']} [{bar}] {pct:.0f}%")

    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)
    overall_pct = total_pass / total * 100
    print(f"\n  OVERALL: {total_pass}/{total} = {overall_pct:.1f}%")
    print("=" * 60)

    return total_pass, total


def save_report(results, checkpoint_path: str):
    """Save eval report to docs/math_progress.jsonl for progress tracking."""
    report_path = Path("docs/math_progress.jsonl")
    report_path.parent.mkdir(exist_ok=True)

    total_pass = sum(1 for r in results if r["passed"])
    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "total_pass": total_pass,
        "total": len(results),
        "pct": round(total_pass / len(results) * 100, 1),
        "per_stage": {},
        "results": results,
    }

    stage_scores = {}
    for r in results:
        s = str(r["stage"])
        if s not in stage_scores:
            stage_scores[s] = {"pass": 0, "total": 0}
        stage_scores[s]["total"] += 1
        if r["passed"]:
            stage_scores[s]["pass"] += 1

    report["per_stage"] = stage_scores

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")

    print(f"\n  Report saved to {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stage", type=int, default=None,
                        help="Evaluate only a specific stage (1-8)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary, not individual Q&A")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_best_checkpoint()
    print(f"Loading checkpoint: {checkpoint}")

    # Load model using same pattern as eval_instruction.py
    model_config_path = "configs/model/yaya_tiny.yaml"
    cfg = load_model_config(model_config_path)
    model = YayaForCausalLM(cfg)
    ckpt_mgr = CheckpointManager(save_dir=str(Path(checkpoint).parent))
    ckpt_mgr.load(model, checkpoint_path=checkpoint)
    model.eval()

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    generator = TextGenerator(model, tokenizer, device="cpu")

    # Filter questions
    questions = EVAL_QUESTIONS
    if args.stage:
        questions = [q for q in EVAL_QUESTIONS if q["stage"] == args.stage]
        print(f"Running Stage {args.stage} ({STAGE_NAMES.get(args.stage, '?')}) — {len(questions)} questions")

    # Run eval
    results = run_eval(generator, tokenizer, questions, verbose=not args.quiet)
    total_pass, total = print_summary(results)
    save_report(results, checkpoint)


if __name__ == "__main__":
    main()
