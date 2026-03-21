"""
Download OpenHermes 2.5 (best open SFT dataset) and convert to Yaya format.

OpenHermes 2.5 has ~1M high-quality instruction examples from GPT-4.
We take the best 100K to keep SFT training manageable on Kaggle.

Usage:
    python scripts/download_openhermes.py --max_examples 100000
"""

import argparse
import json
import os
import random

OUTPUT_PATH = "data/sft/openhermes.jsonl"
FINAL_PATH  = "data/sft/yaya_instruct.jsonl"

ROLE_MAP = {
    "system":    "system",
    "human":     "user",
    "user":      "user",
    "gpt":       "assistant",
    "assistant": "assistant",
}

DEFAULT_SYSTEM = (
    "You are Yaya, a helpful, honest, and friendly AI assistant. "
    "You answer questions clearly and thoughtfully."
)


def convert_example(item):
    """Convert OpenHermes ShareGPT format to Yaya messages format."""
    convs = item.get("conversations", [])
    if not convs:
        return None

    messages = []
    has_system = convs[0].get("from", "") == "system"

    if not has_system:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM})

    for turn in convs:
        role_raw = turn.get("from", "").lower()
        role = ROLE_MAP.get(role_raw)
        if role is None:
            return None  # unknown role — skip
        content = turn.get("value", "").strip()
        if not content:
            return None
        messages.append({"role": role, "content": content})

    # Must have at least user + assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=100_000,
                        help="Max examples to keep (default 100K)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("data/sft", exist_ok=True)

    print("Downloading OpenHermes 2.5 from HuggingFace...", flush=True)
    print("(This requires: pip install datasets)", flush=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        return

    # Stream the dataset — stops after collecting max_examples good examples
    # instead of downloading all ~1M rows (~1GB) first.
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    print("Converting to Yaya format (streaming)...", flush=True)
    converted = []
    skipped = 0
    seen = 0

    for item in ds:
        seen += 1
        ex = convert_example(item)
        if ex is None:
            skipped += 1
        else:
            converted.append(ex)

        if seen % 50_000 == 0:
            print(f"  Scanned {seen:,} | Kept {len(converted):,} | Skipped {skipped:,}", flush=True)

        # Collect 3× target so we have a good shuffle pool, then stop
        if len(converted) >= args.max_examples * 3:
            break

    print(f"Converted: {len(converted):,}  Skipped: {skipped:,}", flush=True)

    # Shuffle and take top N
    random.seed(args.seed)
    random.shuffle(converted)
    selected = converted[:args.max_examples]
    print(f"Selected:  {len(selected):,} examples", flush=True)

    # Save to openhermes.jsonl
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in selected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved to {OUTPUT_PATH}", flush=True)

    # Merge with existing yaya_instruct.jsonl
    existing = []
    if os.path.exists(FINAL_PATH):
        with open(FINAL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(line)
        print(f"Existing yaya_instruct.jsonl: {len(existing):,} examples", flush=True)

    # Write merged file (OpenHermes first, then existing Yaya-specific data)
    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        for ex in selected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for line in existing:
            f.write(line + "\n")

    total = len(selected) + len(existing)
    print(f"\nFinal yaya_instruct.jsonl: {total:,} examples", flush=True)
    print(f"  {len(selected):,} from OpenHermes 2.5", flush=True)
    print(f"  {len(existing):,} from Yaya-specific data (chess, phases, etc.)", flush=True)
    print("\nYaya SFT dataset is ready.", flush=True)


if __name__ == "__main__":
    main()
