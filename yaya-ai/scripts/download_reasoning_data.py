"""Download and convert public reasoning datasets to Yaya format.

Sources:
  - GSM8K       (openai/gsm8k)          — 8.5k grade-school math with CoT
  - MetaMath    (meta-math/MetaMathQA)   — 395k math reasoning (augmented)
  - OpenHermes  (teknium/OpenHermes-2.5) — 50k subset general instruction/reasoning

All converted to Yaya chat format:
  <|think|>step-by-step reasoning<|/think|>
  Final answer

Calculator calls <<expr=result>> in GSM8K/MetaMath are converted to
  <|calc|>expr<|/calc|>=result

Output: data/sft/yaya_reasoning_large.jsonl  (~450k examples)
"""

import json
import re
import sys
import os
import random
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

SYSTEM_MATH = (
    "You are Yaya, a brilliant math assistant. "
    "Think through problems step by step, show your work, "
    "and use <|calc|>EXPRESSION<|/calc|> for any arithmetic."
)

SYSTEM_REASON = (
    "You are Yaya, a helpful and intelligent AI assistant. "
    "You think carefully before answering, breaking problems down step by step."
)

OUT_PATH = Path("data/sft/yaya_reasoning_large.jsonl")
OUT_PATH.parent.mkdir(exist_ok=True)


# ── GSM8K conversion ──────────────────────────────────────────────────────────

def convert_gsm8k_calc(text: str) -> str:
    """Convert GSM8K <<expr=result>> to <|calc|>expr<|/calc|>=result."""
    return re.sub(
        r"<<([^=<>]+)=([^<>]*)>>",
        lambda m: f"<|calc|>{m.group(1).strip()}<|/calc|>={m.group(2).strip()}",
        text,
    )


def parse_gsm8k(row) -> dict | None:
    question = row["question"].strip()
    answer_raw = row["answer"]

    # Split on #### to get CoT steps and final answer
    if "####" not in answer_raw:
        return None
    cot_part, final = answer_raw.split("####", 1)
    cot = convert_gsm8k_calc(cot_part.strip())
    final = final.strip()

    response = f"<|think|>\n{cot}\n<|/think|>\n\n{final}"
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MATH},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": response},
        ]
    }


def load_gsm8k() -> list:
    print("Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    examples = []
    for split in ("train", "test"):
        for row in ds[split]:
            ex = parse_gsm8k(row)
            if ex:
                examples.append(ex)
    print(f"  GSM8K: {len(examples)} examples")
    return examples


# ── MetaMath conversion ───────────────────────────────────────────────────────

def convert_metamath_calc(text: str) -> str:
    """MetaMath uses <<expr=result>> same as GSM8K."""
    return re.sub(
        r"<<([^=<>]+)=([^<>]*)>>",
        lambda m: f"<|calc|>{m.group(1).strip()}<|/calc|>={m.group(2).strip()}",
        text,
    )


def parse_metamath(row) -> dict | None:
    query    = row.get("query", "").strip()
    response = row.get("response", "").strip()
    if not query or not response:
        return None

    response = convert_metamath_calc(response)

    # Extract final answer after "The answer is:" if present
    final_match = re.search(r"[Tt]he answer is:?\s*([^\n]+)", response)
    if final_match:
        # Put everything before the final line in <|think|>
        final_ans = final_match.group(1).strip()
        cot = response[:final_match.start()].strip()
        # Clean trailing "The answer is:" from cot
        cot = re.sub(r"\s*[Tt]he answer is:?.*$", "", cot, flags=re.DOTALL).strip()
        full = f"<|think|>\n{cot}\n<|/think|>\n\n{final_ans}"
    else:
        # Wrap full response in think block
        full = f"<|think|>\n{response}\n<|/think|>"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_MATH},
            {"role": "user",      "content": query},
            {"role": "assistant", "content": full},
        ]
    }


def load_metamath(max_examples: int = 200_000) -> list:
    print(f"Downloading MetaMath (max {max_examples:,})...")
    ds = load_dataset("meta-math/MetaMathQA", trust_remote_code=True)
    examples = []
    for row in ds["train"]:
        ex = parse_metamath(row)
        if ex:
            examples.append(ex)
        if len(examples) >= max_examples:
            break
    print(f"  MetaMath: {len(examples)} examples")
    return examples


# ── OpenHermes conversion ─────────────────────────────────────────────────────

def parse_openhermes(row) -> dict | None:
    """Convert OpenHermes ChatML conversations to Yaya format."""
    convs = row.get("conversations", [])
    if not convs:
        return None

    messages = []
    system_added = False

    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "").strip()
        if not value:
            continue

        if role == "system":
            messages.append({"role": "system", "content": value})
            system_added = True
        elif role in ("human", "user"):
            messages.append({"role": "user", "content": value})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": value})

    # Must have at least one user+assistant pair
    has_user = any(m["role"] == "user" for m in messages)
    has_asst = any(m["role"] == "assistant" for m in messages)
    if not has_user or not has_asst:
        return None

    if not system_added:
        messages.insert(0, {"role": "system", "content": SYSTEM_REASON})

    return {"messages": messages}


def load_openhermes(max_examples: int = 50_000) -> list:
    print(f"Downloading OpenHermes 2.5 (max {max_examples:,})...")
    ds = load_dataset("teknium/OpenHermes-2.5", trust_remote_code=True)
    examples = []
    random.seed(42)
    indices = random.sample(range(len(ds["train"])), min(max_examples, len(ds["train"])))
    for i in indices:
        ex = parse_openhermes(ds["train"][i])
        if ex:
            examples.append(ex)
    print(f"  OpenHermes: {len(examples)} examples")
    return examples


# ── Existing Yaya data ────────────────────────────────────────────────────────

def load_existing() -> list:
    existing = [
        "data/sft/yaya_reasoning_combined.jsonl",
        "data/sft/yaya_cot.jsonl",
    ]
    examples = []
    for path in existing:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
            print(f"  Existing {path}: {len(examples)} total so far")
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    all_examples = []

    all_examples += load_existing()
    all_examples += load_gsm8k()
    all_examples += load_metamath(max_examples=200_000)
    all_examples += load_openhermes(max_examples=50_000)

    random.shuffle(all_examples)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_examples):,} examples → {OUT_PATH}")
    print("Breakdown:")
    print(f"  Existing Yaya data: ~3,455")
    print(f"  GSM8K:              ~8,500")
    print(f"  MetaMath:           ~200,000")
    print(f"  OpenHermes:         ~50,000")


if __name__ == "__main__":
    main()
