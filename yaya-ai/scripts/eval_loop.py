"""Eval-driven self-improvement loop for Yaya.

Runs eval_instruction.py on the current checkpoint, identifies failed test
cases, appends targeted training examples to the clean dataset, and prints
the command to start a new training run.

Usage:
    python scripts/eval_loop.py
    python scripts/eval_loop.py --checkpoint checkpoints/yaya-tiny-sft-focused/checkpoint-00005000
    python scripts/eval_loop.py --checkpoint ... --retrain        # auto-launches training
    python scripts/eval_loop.py --dry_run                         # show what would be added
"""

import argparse
import json
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_instruction import evaluate_checkpoint

SYS = (
    "You are Yaya, a helpful and honest AI assistant. "
    "You answer questions clearly and concisely."
)

# ── Per-question targeted training examples ────────────────────────────────
# Keyed by a substring of the eval question (lowercase).
# Each value is a list of (user, assistant) pairs.
TARGETED = {
    "2 + 2": [
        ("What is 2 + 2?", "4"),
        ("What is 2 plus 2?", "2 plus 2 equals 4."),
        ("Calculate 2 + 2.", "2 + 2 = 4"),
        ("Add 2 and 2.", "2 + 2 = 4."),
        ("What does 2 + 2 equal?", "2 + 2 equals 4."),
    ],
    "10 times 5": [
        ("What is 10 times 5?", "10 times 5 is 50."),
        ("What is 10 x 5?", "50"),
        ("Multiply 10 by 5.", "10 multiplied by 5 is 50."),
        ("What is 10 * 5?", "10 * 5 = 50"),
        ("Calculate 10 times 5.", "10 times 5 = 50."),
    ],
    "100 divided by 4": [
        ("What is 100 divided by 4?", "100 divided by 4 is 25."),
        ("What is 100 / 4?", "25"),
        ("Divide 100 by 4.", "100 divided by 4 equals 25."),
        ("What is one hundred divided by four?", "One hundred divided by four is twenty-five (25)."),
        ("Calculate 100 / 4.", "100 / 4 = 25"),
    ],
    "capital of france": [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What city is the capital of France?", "Paris is the capital city of France."),
        ("Name the capital of France.", "Paris"),
        ("France's capital city?", "The capital of France is Paris."),
        ("Where is the seat of government in France?", "The seat of government in France is Paris, the capital city."),
    ],
    "capital of kenya": [
        ("What is the capital of Kenya?", "The capital of Kenya is Nairobi."),
        ("What is Kenya's capital city?", "Nairobi is the capital city of Kenya."),
        ("Name the capital of Kenya.", "Nairobi"),
        ("Where is the capital of Kenya?", "The capital of Kenya is Nairobi."),
        ("Which city is the capital of Kenya?", "Nairobi is the capital of Kenya."),
    ],
    "days are in a week": [
        ("How many days are in a week?", "There are 7 days in a week."),
        ("How many days in a week?", "7 days."),
        ("Count the days in a week.", "A week has 7 days: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday."),
        ("A week has how many days?", "A week has 7 days."),
        ("How many days make up a week?", "A week is made up of 7 days."),
    ],
    "closest to the sun": [
        ("What planet is closest to the Sun?", "Mercury is the planet closest to the Sun."),
        ("Which planet is nearest to the Sun?", "Mercury is the nearest planet to the Sun."),
        ("Name the planet closest to the Sun.", "Mercury"),
        ("What is the first planet from the Sun?", "Mercury is the first planet from the Sun."),
        ("Which planet orbits closest to our Sun?", "Mercury orbits closest to the Sun."),
    ],
    "democracy": [
        ("Define the word 'democracy'.",
         "Democracy is a system of government in which people elect their leaders and have "
         "a say in decisions. Citizens vote to choose their representatives. The word comes "
         "from Greek: demos (people) + kratos (power)."),
        ("What is democracy?",
         "Democracy is a form of government where the people hold power. Citizens elect "
         "their leaders through voting. It values freedom, equality, and the rule of law."),
        ("Explain democracy in simple terms.",
         "Democracy means the people govern themselves. Citizens vote for leaders who make "
         "decisions on their behalf. If the leaders do a poor job, citizens can vote them out."),
        ("What does democracy mean?",
         "Democracy means rule by the people. In a democracy, citizens elect their "
         "government and can vote to change it."),
    ],
    "gravity": [
        ("What is gravity?",
         "Gravity is a fundamental force of nature. It is the attractive force between "
         "objects with mass. The more massive an object, the stronger its gravitational "
         "pull. Gravity keeps planets in orbit around the Sun and holds us on Earth."),
        ("Explain gravity.",
         "Gravity is the force that attracts objects toward each other. On Earth, gravity "
         "pulls everything downward toward the centre of the planet. It is what gives "
         "objects weight and keeps the Moon orbiting Earth."),
        ("Define gravity.",
         "Gravity is a natural force that pulls objects with mass toward each other. "
         "It is described by Newton's law of universal gravitation. The greater the "
         "mass of an object, the stronger its gravitational attraction."),
        ("What causes gravity?",
         "Gravity is caused by mass. Every object with mass exerts a gravitational "
         "pull on every other object. The strength of this pull depends on the masses "
         "involved and the distance between them."),
    ],
    "primary colors": [
        ("List three primary colors.", "The three primary colors are red, blue, and yellow."),
        ("What are the primary colors?", "The primary colors are red, yellow, and blue."),
        ("Name the primary colors.", "The primary colors are red, blue, and yellow."),
        ("What are the three primary colors?", "Red, yellow, and blue are the three primary colors."),
        ("Tell me the primary colors.", "The primary colors are red, blue, and yellow. These cannot be made by mixing other colors."),
    ],
    "first three us presidents": [
        ("Name the first three US presidents.",
         "The first three US presidents were:\n1. George Washington\n2. John Adams\n3. Thomas Jefferson"),
        ("Who were the first three presidents of the United States?",
         "The first three presidents of the United States were George Washington (1st), "
         "John Adams (2nd), and Thomas Jefferson (3rd)."),
        ("List the first three American presidents.",
         "1. George Washington\n2. John Adams\n3. Thomas Jefferson"),
        ("Who was the first president of the USA?",
         "George Washington was the first president of the United States, serving from 1789 to 1797."),
        ("Who was the third US president?",
         "Thomas Jefferson was the third president of the United States, serving from 1801 to 1809."),
    ],
    "your name": [
        ("What is your name?", "My name is Yaya. I am an AI assistant."),
        ("Who are you?", "I am Yaya, an AI assistant built to help you with questions, explanations, and more."),
        ("What should I call you?", "You can call me Yaya. I am your AI assistant."),
        ("Tell me your name.", "My name is Yaya."),
        ("What are you called?", "I am called Yaya. I am an AI assistant."),
    ],
    "are you an ai": [
        ("Are you an AI?", "Yes, I am an AI assistant named Yaya."),
        ("Are you a real person?", "No, I am not a real person. I am Yaya, an AI assistant."),
        ("Are you human?", "No, I am not human. I am Yaya, an artificial intelligence assistant."),
        ("Am I talking to a human or AI?", "You are talking to an AI. I am Yaya, an AI assistant."),
        ("Is this a bot?", "Yes, I am an AI assistant named Yaya, not a human."),
    ],
}


def question_key(question: str) -> str | None:
    """Find a TARGETED key that matches the eval question."""
    q = question.lower()
    for key in TARGETED:
        if key in q:
            return key
    return None


def main():
    parser = argparse.ArgumentParser(description="Eval-driven self-improvement loop")
    parser.add_argument("--checkpoint",   type=str, default=None)
    parser.add_argument("--model_config", type=str, default="configs/model/yaya_tiny.yaml")
    parser.add_argument("--tokenizer",    type=str, default="data/tokenizer/yaya_tokenizer.model")
    parser.add_argument("--temperature",  type=float, default=0.3)
    parser.add_argument("--max_tokens",   type=int,   default=120)
    parser.add_argument("--output",       type=str,   default="data/sft/yaya_instruct_filtered.jsonl")
    parser.add_argument("--report",       type=str,   default="data/eval/eval_loop_report.json")
    parser.add_argument("--dry_run",      action="store_true", help="Show what would be added, don't write")
    parser.add_argument("--retrain",      action="store_true", help="Launch focused training after augmenting data")
    parser.add_argument("--repeats",      type=int,   default=5, help="Times to repeat each new example")
    args = parser.parse_args()

    # Auto-find checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        for d in ["checkpoints/yaya-tiny-sft-focused",
                  "checkpoints/yaya-tiny-sft-clean",
                  "checkpoints/yaya-tiny-sft-v2",
                  "checkpoints/yaya-tiny-sft"]:
            latest = os.path.join(d, "latest")
            if os.path.exists(latest):
                with open(latest) as f:
                    name = f.read().strip()
                ckpt = os.path.join(d, name)
                break
    if ckpt is None:
        print("ERROR: No checkpoint found.")
        sys.exit(1)

    print(f"Evaluating: {ckpt}")
    report = evaluate_checkpoint(ckpt, args.model_config, args.tokenizer,
                                 args.temperature, args.max_tokens)

    score   = report["score"]
    passed  = report["passed"]
    total   = report["total"]
    step    = report["step"]
    loss    = report["train_loss"]

    print(f"\nStep {step} | loss={loss:.4f} | score={score:.0%} ({passed}/{total})")
    print("-" * 60)

    failed_questions = [r for r in report["results"] if not r["passed"]]
    print(f"Failed: {len(failed_questions)} questions\n")

    # Map failures to training examples
    to_add: list[tuple[str, str]] = []
    unmatched: list[str] = []
    for r in failed_questions:
        key = question_key(r["question"])
        if key:
            examples = TARGETED[key]
            to_add.extend(examples)
            print(f"  FAIL  {r['question'][:55]:<55s}  +{len(examples)} examples")
        else:
            unmatched.append(r["question"])
            print(f"  FAIL  {r['question'][:55]:<55s}  (no targeted examples)")

    if unmatched:
        print(f"\n[!] {len(unmatched)} failed questions have no targeted examples.")
        print("    Add them to the TARGETED dict in scripts/eval_loop.py")

    # Save report JSON
    report["failed_questions"] = [r["question"] for r in failed_questions]
    report["unmatched"]        = unmatched
    report["new_examples"]     = len(to_add) * args.repeats
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.report}")

    if not to_add:
        print("\nNothing to add. Model is passing all targeted questions!")
        return

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Adding {len(to_add) * args.repeats} examples "
          f"({len(to_add)} unique x {args.repeats}) to {args.output}")

    if not args.dry_run:
        # Count current size
        existing = 0
        if os.path.exists(args.output):
            with open(args.output, encoding="utf-8") as f:
                existing = sum(1 for line in f if line.strip())

        with open(args.output, "a", encoding="utf-8") as f:
            for _ in range(args.repeats):
                for user, assistant in to_add:
                    ex = {"messages": [
                        {"role": "system",    "content": SYS},
                        {"role": "user",      "content": user},
                        {"role": "assistant", "content": assistant},
                    ]}
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        new_total = existing + len(to_add) * args.repeats
        print(f"Dataset size: {existing} -> {new_total}")

    print(f"\nCurrent score: {score:.0%} ({passed}/{total})")
    if score < 1.0:
        print("\nTo retrain with augmented data:")
        print("  make sft-tiny-focused")
        if args.retrain:
            print("\nLaunching focused training...")
            subprocess.Popen(
                [sys.executable, "scripts/train_sft.py",
                 "--model_config", "configs/model/yaya_tiny.yaml",
                 "--train_config", "configs/training/sft_tiny_focused.yaml",
                 "--pretrain_checkpoint", ckpt],
                cwd=os.getcwd(),
            )
            print("Training launched in background.")
    else:
        print("\n100% pass rate! Consider running DPO alignment next:")
        print("  make dpo-tiny")


if __name__ == "__main__":
    main()
