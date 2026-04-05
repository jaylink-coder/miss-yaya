"""Add format-enforcement system prompts to 30% of short_qa + quick_facts.

Creates data/sft/yaya_concise_sft.jsonl — same Q&A content but with
system prompts that explicitly instruct direct/concise answers.

This teaches the model to follow "be concise" instructions and
counteracts the numbered-list habit without touching the original data.

Usage:
    python scripts/add_format_enforcement.py
"""

import sys, os, json, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT_QA    = os.path.join(REPO_ROOT, "data/sft/yaya_short_qa.jsonl")
QUICK_FACTS = os.path.join(REPO_ROOT, "data/sft/teach/quick_facts.jsonl")
OUTPUT      = os.path.join(REPO_ROOT, "data/sft/yaya_concise_sft.jsonl")

# Varied concise-instruction system prompts (so the model generalises)
CONCISE_SYSTEMS = [
    "You are Yaya, a helpful AI assistant. Answer questions directly and concisely. Give the answer without numbered lists or unnecessary explanation.",
    "You are Yaya. Be brief. Answer the question in one or two words or a short sentence. Do not use bullet points or numbered lists.",
    "You are Yaya, a helpful AI assistant. Give short, direct answers. No lists, no padding.",
    "You are Yaya. Answer concisely. For factual questions, give just the fact. For math, give just the number.",
    "You are Yaya, a concise AI assistant. Respond with the answer only — no numbered steps, no 'let me explain', no preamble.",
    "You are Yaya. Keep answers short. One sentence maximum unless a longer answer is truly needed.",
    "You are Yaya, a helpful assistant. Be direct. Answer immediately without preamble or numbered lists.",
    "You are Yaya. For simple questions, answer in one word or phrase. No lists, no elaboration unless asked.",
]

def load_qa_pairs(filepath):
    pairs = []
    if not os.path.exists(filepath):
        return pairs
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                question = next((m["content"] for m in msgs if m["role"] == "user"), None)
                answer   = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if question and answer:
                    pairs.append((question, answer))
            except Exception:
                pass
    return pairs

def main():
    random.seed(42)

    short_qa    = load_qa_pairs(SHORT_QA)
    quick_facts = load_qa_pairs(QUICK_FACTS)
    all_qa = short_qa + quick_facts
    print(f"Source pairs: {len(short_qa)} short_qa + {len(quick_facts)} quick_facts = {len(all_qa)}")

    # Generate one concise-system-prompt version per pair (all of them, not 30%)
    # The recovery dataset will then mix this with the standard format at ~30% ratio
    examples = []
    for question, answer in all_qa:
        sys_prompt = random.choice(CONCISE_SYSTEMS)
        examples.append(json.dumps({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": question},
                {"role": "assistant", "content": answer},
            ]
        }, ensure_ascii=False))

    random.shuffle(examples)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex + "\n")

    print(f"Saved {len(examples)} format-enforcement examples → {OUTPUT}")


if __name__ == "__main__":
    main()
