"""Merge math + CoT + reasoning data into a single training file.

Sources:
  data/sft/math/yaya_math_combined.jsonl  — 3391 math examples (staged curriculum)
  data/sft/yaya_cot.jsonl                 — ~20 CoT examples (general reasoning)
  data/sft/yaya_reasoning.jsonl           — 41 CoT+calc examples (generated)

Output:
  data/sft/yaya_reasoning_combined.jsonl
"""

import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCES = [
    "data/sft/math/yaya_math_combined.jsonl",
    "data/sft/yaya_cot.jsonl",
    "data/sft/yaya_reasoning.jsonl",
]

OUTPUT = Path("data/sft/yaya_reasoning_combined.jsonl")

def main():
    random.seed(42)
    examples = []

    for src in SOURCES:
        path = Path(src)
        if not path.exists():
            print(f"  SKIP (not found): {src}")
            continue
        before = len(examples)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        print(f"  Loaded {len(examples) - before:>5} examples from {src}")

    random.shuffle(examples)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(examples)} examples -> {OUTPUT}")


if __name__ == "__main__":
    main()
