"""Generate DPO preference pairs from existing SFT data.

Strategy:
  chosen  = the high-quality SFT answer (already in yaya_instruct.jsonl)
  rejected = a degraded version (too short / unhelpful / vague)

This creates training signal for DPO to prefer detailed, complete answers
over brief, unhelpful ones — no model inference needed.
"""

import json
import random
import re
import argparse

VAGUE_TEMPLATES = [
    "That's a complex topic. There are many ways to think about it.",
    "It depends on the situation. You should consider your specific context.",
    "I'm not entirely sure about the details here. You might want to research this further.",
    "There are multiple perspectives on this. It's hard to say definitively.",
    "Good question. This is something many people wonder about.",
    "That varies quite a bit. I don't have a simple answer for you.",
    "It's complicated. There's a lot to consider here.",
    "I'd need more context to give a specific answer.",
]

def degrade_response(response: str) -> str:
    """Create a deliberately worse version of a good response."""
    method = random.random()

    if method < 0.25:
        # Truncate to first sentence only
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        if len(sentences) > 2:
            return sentences[0]
        return response[:60].rsplit(' ', 1)[0] + "..."

    elif method < 0.50:
        # Use a vague non-answer
        return random.choice(VAGUE_TEMPLATES)

    elif method < 0.75:
        # Keep only the first 15% of the text (cut off mid-thought)
        cut = max(40, int(len(response) * 0.15))
        return response[:cut].rsplit(' ', 1)[0] + "..."

    else:
        # Remove all concrete details (numbers, code, specific terms)
        degraded = re.sub(r'```[\s\S]*?```', '[code omitted]', response)
        degraded = re.sub(r'\b\d+\b', 'N', degraded)
        degraded = degraded[:200] + ("..." if len(degraded) > 200 else "")
        return degraded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/sft/yaya_instruct.jsonl")
    parser.add_argument("--output", default="data/sft/yaya_dpo_synthetic.jsonl")
    parser.add_argument("--n",      type=int, default=500, help="Max pairs to generate")
    parser.add_argument("--min_response_len", type=int, default=120,
                        help="Only use responses longer than this")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    pairs = []
    for s in samples:
        msgs = s.get("messages", [])
        prompt = next((m["content"] for m in msgs if m["role"] == "user"), None)
        chosen = next((m["content"] for m in msgs if m["role"] == "assistant"), None)

        if not prompt or not chosen:
            continue
        if len(chosen) < args.min_response_len:
            continue

        rejected = degrade_response(chosen)
        if rejected == chosen or len(rejected) >= len(chosen) * 0.9:
            continue  # Degradation didn't work — skip

        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(pairs) >= args.n:
            break

    random.shuffle(pairs)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Generated {len(pairs)} DPO pairs -> {args.output}")


if __name__ == "__main__":
    main()
