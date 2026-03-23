"""
OpenHermes 2.5 full pipeline: download → filter → dedup → merge → shuffle.

Steps:
  1. Stream 100K quality examples from teknium/OpenHermes-2.5 on HuggingFace
  2. Quality filter: min/max length, no empty turns, proper role structure
  3. Deduplicate: hash first user turn against existing yaya_instruct.jsonl
     so we strip old OpenHermes examples and keep Yaya-specific data intact
  4. Merge: 100K OpenHermes + Yaya-specific data
  5. Shuffle: randomize final file
  6. Save openhermes.jsonl (intermediate) and yaya_instruct.jsonl (final)

Usage:
    python scripts/download_openhermes.py [--max_examples 100000] [--seed 42]
"""

import argparse
import hashlib
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

# Quality filter thresholds
MIN_TURN_CHARS  = 10    # minimum chars per non-system turn
MAX_TURN_CHARS  = 8192  # maximum chars per turn (skip wall-of-text examples)
MIN_TOTAL_CHARS = 50    # minimum total conversation length


def _first_user_hash(messages: list) -> str:
    """Stable hash of the first user turn content — used for dedup."""
    for m in messages:
        if m["role"] == "user":
            return hashlib.md5(m["content"].encode("utf-8")).hexdigest()
    return ""


def quality_ok(messages: list) -> bool:
    """Return True if the example passes quality filters."""
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return False

    total_chars = 0
    for m in messages:
        if m["role"] == "system":
            continue
        c = len(m["content"])
        if c < MIN_TURN_CHARS or c > MAX_TURN_CHARS:
            return False
        total_chars += c

    return total_chars >= MIN_TOTAL_CHARS


def convert_example(item) -> dict | None:
    """Convert OpenHermes ShareGPT format → Yaya messages format."""
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
            return None
        content = turn.get("value", "").strip()
        if not content:
            return None
        messages.append({"role": role, "content": content})

    if not quality_ok(messages):
        return None

    # Normalise system prompt to Yaya branding
    if messages[0]["role"] == "system":
        messages[0]["content"] = DEFAULT_SYSTEM

    return {"messages": messages}


def load_existing(path: str) -> tuple[list[dict], set[str]]:
    """
    Load existing JSONL file.
    Returns (examples list, set of first-user-turn hashes).
    """
    examples, hashes = [], set()
    if not os.path.exists(path):
        return examples, hashes
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                h = _first_user_hash(ex.get("messages", []))
                examples.append((h, ex))
                hashes.add(h)
            except json.JSONDecodeError:
                pass
    return examples, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=100_000,
                        help="Max OpenHermes examples to keep (default 100K)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("data/sft", exist_ok=True)
    random.seed(args.seed)

    # ── Step 1: load existing file and separate Yaya-specific from OpenHermes ──
    print("Loading existing yaya_instruct.jsonl...", flush=True)
    existing_pairs, existing_hashes = load_existing(FINAL_PATH)
    print(f"  Existing examples: {len(existing_pairs):,}", flush=True)

    # ── Step 2: download fresh OpenHermes ──
    print("\nDownloading OpenHermes 2.5 from HuggingFace (streaming)...", flush=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        return

    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    print(f"Converting + quality filtering (target: {args.max_examples:,})...", flush=True)
    new_examples: list[tuple[str, dict]] = []
    skipped = 0
    scanned = 0

    for item in ds:
        scanned += 1
        ex = convert_example(item)
        if ex is None:
            skipped += 1
        else:
            h = _first_user_hash(ex["messages"])
            new_examples.append((h, ex))

        if scanned % 50_000 == 0:
            print(f"  Scanned {scanned:,} | Kept {len(new_examples):,} | Skipped {skipped:,}", flush=True)

        # Stop as soon as we have enough — shuffle handles randomness
        if len(new_examples) >= args.max_examples:
            break

    print(f"  Total scanned: {scanned:,} | Converted: {len(new_examples):,} | Skipped: {skipped:,}", flush=True)

    # ── Step 3: shuffle + select top N ──
    random.shuffle(new_examples)
    selected = new_examples
    selected_hashes = {h for h, _ in selected}
    print(f"\nSelected {len(selected):,} OpenHermes examples after shuffle.", flush=True)

    # ── Step 4: save openhermes.jsonl (intermediate) ──
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, ex in selected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved intermediate: {OUTPUT_PATH}", flush=True)

    # ── Step 5: identify Yaya-specific examples (not in new OpenHermes set) ──
    yaya_specific = [ex for h, ex in existing_pairs if h not in selected_hashes]
    print(f"\nYaya-specific examples preserved: {len(yaya_specific):,}", flush=True)
    print(f"Old OpenHermes examples dropped:  {len(existing_pairs) - len(yaya_specific):,}", flush=True)

    # ── Step 6: merge + shuffle ──
    all_examples = [ex for _, ex in selected] + yaya_specific
    random.shuffle(all_examples)

    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nFinal yaya_instruct.jsonl: {len(all_examples):,} examples", flush=True)
    print(f"  {len(selected):,} from OpenHermes 2.5", flush=True)
    print(f"  {len(yaya_specific):,} Yaya-specific (chess, Africa, values, etc.)", flush=True)
    print("\nYaya SFT dataset ready.", flush=True)


if __name__ == "__main__":
    main()
