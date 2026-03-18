"""Download and merge public SFT datasets from HuggingFace.

Downloads a sample from Alpaca and Dolly, converts to Yaya format,
and merges with our hand-crafted examples.

Output: data/sft/yaya_instruct.jsonl  (merged dataset)

Usage:
    python scripts/download_sft_data.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly, tell jokes when asked, and are always honest."
)

OUT_DIR  = "data/sft"
OUT_PATH = os.path.join(OUT_DIR, "yaya_instruct.jsonl")
SEED_PATH = OUT_PATH  # existing hand-crafted data


def make_sample(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def load_existing() -> list:
    samples = []
    if os.path.exists(SEED_PATH):
        with open(SEED_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"Loaded {len(samples)} existing hand-crafted examples")
    return samples


def download_alpaca(max_samples: int = 300) -> list:
    """Download Stanford Alpaca instruction dataset."""
    print("Downloading Alpaca dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        samples = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            inp         = row.get("input", "").strip()
            output      = row.get("output", "").strip()
            if not instruction or not output:
                continue
            # Combine instruction + input if present
            user_msg = instruction
            if inp:
                user_msg = f"{instruction}\n\n{inp}"
            samples.append(make_sample(user_msg, output))
            if len(samples) >= max_samples:
                break
        print(f"  Got {len(samples)} Alpaca examples")
        return samples
    except Exception as e:
        print(f"  Alpaca download failed: {e}")
        return []


def download_dolly(max_samples: int = 200) -> list:
    """Download Databricks Dolly-15k dataset."""
    print("Downloading Dolly dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        samples = []
        for row in ds:
            instruction = row.get("instruction", "").strip()
            context     = row.get("context", "").strip()
            response    = row.get("response", "").strip()
            if not instruction or not response:
                continue
            user_msg = instruction
            if context:
                user_msg = f"{instruction}\n\nContext: {context}"
            samples.append(make_sample(user_msg, response))
            if len(samples) >= max_samples:
                break
        print(f"  Got {len(samples)} Dolly examples")
        return samples
    except Exception as e:
        print(f"  Dolly download failed: {e}")
        return []


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_samples = []
    all_samples.extend(load_existing())
    all_samples.extend(download_alpaca(max_samples=300))
    all_samples.extend(download_dolly(max_samples=200))

    # Shuffle and deduplicate roughly by user message
    seen = set()
    deduped = []
    for s in all_samples:
        key = s["messages"][1]["content"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    random.shuffle(deduped)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(deduped)} total examples to {OUT_PATH}")
    print(f"  Hand-crafted: {len(load_existing())}")
    print(f"  From HuggingFace: {len(deduped) - len(load_existing())}")


if __name__ == "__main__":
    main()
