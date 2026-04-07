#!/usr/bin/env python3
"""Generate training data for all 16 phases of the Yaya True AI curriculum.

Usage:
    python scripts/generate_curriculum_data.py                 # all phases
    python scripts/generate_curriculum_data.py --phase 5       # single phase
    python scripts/generate_curriculum_data.py --phase 1-4     # range
"""
import argparse, json, os, random, sys

# Ensure scripts/ is on path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft", "curriculum")
os.makedirs(OUT_DIR, exist_ok=True)


def save_sft(examples, path):
    """Save SFT examples (list of message-lists) to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex}, ensure_ascii=False) + "\n")
    print(f"    saved {len(examples)} SFT examples -> {os.path.basename(path)}")


def save_dpo(examples, path):
    """Save DPO preference pairs to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"    saved {len(examples)} DPO pairs -> {os.path.basename(path)}")


def main():
    from curriculum_phases import PHASE_GENERATORS

    parser = argparse.ArgumentParser(description="Generate Yaya curriculum training data")
    parser.add_argument("--phase", default="1-16", help="Phase number or range, e.g. 5 or 1-8")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if "-" in args.phase:
        start, end = map(int, args.phase.split("-"))
        phases = list(range(start, end + 1))
    else:
        phases = [int(args.phase)]

    print(f"=== Yaya Curriculum Data Generator ===")
    print(f"Phases: {phases}\n")

    total = 0
    for p in phases:
        if p not in PHASE_GENERATORS:
            print(f"  Phase {p}: no generator, skipping")
            continue
        info = PHASE_GENERATORS[p]
        print(f"  Phase {p}: {info['name']}")
        examples = info["fn"]()
        path = os.path.join(OUT_DIR, f"phase{p:02d}_{info['slug']}.jsonl")

        if p == 16:  # DPO phase
            save_dpo(examples, path)
        else:
            save_sft(examples, path)
        total += len(examples)

    print(f"\nTotal: {total} examples across {len(phases)} phases")
    print("Done!")


if __name__ == "__main__":
    main()
