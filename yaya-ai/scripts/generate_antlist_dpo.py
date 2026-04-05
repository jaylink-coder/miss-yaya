"""Generate anti-numbered-list DPO pairs at scale.

Reads yaya_short_qa.jsonl and quick_facts.jsonl, then for every
direct Q&A pair creates a numbered-list rejected version.

Also generates 500+ format-enforcement pairs where the chosen
response is direct and the rejected is verbose/listed.

Output appended to: data/sft/yaya_dpo_combined.jsonl

Usage:
    python scripts/generate_antlist_dpo.py
"""

import sys, os, json, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT_QA    = os.path.join(REPO_ROOT, "data/sft/yaya_short_qa.jsonl")
QUICK_FACTS = os.path.join(REPO_ROOT, "data/sft/teach/quick_facts.jsonl")
DPO_OUT     = os.path.join(REPO_ROOT, "data/sft/yaya_dpo_combined.jsonl")

SYSTEM = "You are Yaya, a helpful, honest, and friendly AI assistant. You answer questions clearly and thoughtfully."

# Templates to turn a direct answer into a numbered-list rejection
LIST_TEMPLATES = [
    lambda q, a: f"1. {a}\n2. Let me explain this further...\n3. In summary, the answer is {a}.",
    lambda q, a: f"1. To answer your question: {a}\n2. This is because of the underlying principles\n3. Therefore the answer is {a}",
    lambda q, a: f"1. First, let's consider the question: {q}\n2. The answer is {a}\n3. This can be verified by...",
    lambda q, a: f"1. {a}\n2. Additional context: This is a fundamental concept\n3. Key takeaway: {a}",
    lambda q, a: f"There are several things to note:\n1. The answer is {a}\n2. This follows from basic principles\n3. You should remember this",
    lambda q, a: f"Let me break this down:\n1. The question asks about {q[:30]}...\n2. The answer: {a}\n3. In conclusion: {a}",
    lambda q, a: f"1. Answer: {a}\n2. Method: Direct calculation/lookup\n3. Confidence: High",
    lambda q, a: f"To properly answer this:\n1. Consider the context\n2. Apply the relevant knowledge\n3. The answer is {a}",
]

def make_list_version(question, answer):
    """Turn a direct answer into a numbered-list version (rejected)."""
    tmpl = random.choice(LIST_TEMPLATES)
    return tmpl(question, answer)

def make_pair(prompt, chosen, rejected):
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

def load_qa_pairs(filepath):
    """Load Q&A pairs from messages-format JSONL."""
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

    # Load existing DPO pairs to avoid duplicates
    existing_prompts = set()
    existing_pairs = []
    if os.path.exists(DPO_OUT):
        with open(DPO_OUT, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        existing_prompts.add(obj.get("prompt", ""))
                        existing_pairs.append(obj)
                    except Exception:
                        pass
    print(f"Existing DPO pairs: {len(existing_pairs)}")

    # Load source Q&A
    short_qa    = load_qa_pairs(SHORT_QA)
    quick_facts = load_qa_pairs(QUICK_FACTS)
    all_qa = short_qa + quick_facts
    print(f"Source Q&A pairs: {len(short_qa)} short_qa + {len(quick_facts)} quick_facts = {len(all_qa)}")

    new_pairs = []

    # Anti-list pairs: one per Q&A pair, chosen=direct, rejected=numbered-list
    for question, answer in all_qa:
        if question in existing_prompts:
            continue
        rejected = make_list_version(question, answer)
        new_pairs.append(make_pair(question, answer, rejected))
        existing_prompts.add(question)

    # Second pass: varied list templates (different rejection for same question)
    # Use a suffix to make the prompt key unique
    for question, answer in all_qa:
        key = question + " [v2]"
        if key in existing_prompts:
            continue
        # pick a different template
        rejected = LIST_TEMPLATES[(hash(question) + 3) % len(LIST_TEMPLATES)](question, answer)
        # Store with original prompt (no suffix) — DPO prompt should be the real question
        new_pairs.append(make_pair(question, answer, rejected))
        existing_prompts.add(key)

    # Format-enforcement pairs: "Answer concisely" variants
    CONCISE_SYSTEM_QA = [
        ("What is 7 x 7?", "49"),
        ("What is 9 x 9?", "81"),
        ("What is 6 x 8?", "48"),
        ("What is 11 x 11?", "121"),
        ("What is 13 x 13?", "169"),
        ("What is 15 x 15?", "225"),
        ("What is 3 to the power of 4?", "81"),
        ("What is 2 to the power of 8?", "256"),
        ("What is the capital of Italy?", "Rome"),
        ("What is the capital of Spain?", "Madrid"),
        ("What is the capital of China?", "Beijing"),
        ("What is the capital of India?", "New Delhi"),
        ("What is the capital of Egypt?", "Cairo"),
        ("What is the capital of Nigeria?", "Abuja"),
        ("What is the capital of Tanzania?", "Dodoma"),
        ("What is the capital of Uganda?", "Kampala"),
        ("What is the capital of Ethiopia?", "Addis Ababa"),
        ("What is the capital of South Africa?", "Pretoria"),
        ("What is the capital of Ghana?", "Accra"),
        ("What is the capital of Senegal?", "Dakar"),
        ("How many sides does a triangle have?", "3"),
        ("How many sides does a hexagon have?", "6"),
        ("How many degrees are in a right angle?", "90"),
        ("How many degrees in a straight line?", "180"),
        ("How many degrees in a full circle?", "360"),
        ("What is the area of a square with side 4?", "16"),
        ("What is the perimeter of a rectangle 5 by 3?", "16"),
        ("What is pi approximately equal to?", "3.14159"),
        ("What is the value of e (Euler's number) approximately?", "2.718"),
        ("Is 2 a prime number?", "Yes"),
        ("Is 1 a prime number?", "No — 1 is not considered prime"),
        ("Is 0 positive or negative?", "Neither — 0 is neither positive nor negative"),
        ("What is the absolute value of -7?", "7"),
        ("What is -3 x -4?", "12"),
        ("What is (-5) + 8?", "3"),
        ("Spell the word 'necessary'.", "N-E-C-E-S-S-A-R-Y"),
        ("What is a synonym for 'happy'?", "Joyful"),
        ("What is an antonym for 'fast'?", "Slow"),
        ("What does H2O stand for?", "Water"),
        ("What does CO2 stand for?", "Carbon dioxide"),
        ("What is the tallest mountain in the world?", "Mount Everest"),
        ("What is the longest river in the world?", "The Nile"),
        ("What is the largest desert in the world?", "The Sahara"),
        ("What is the smallest country in the world?", "Vatican City"),
        ("What is the currency of Japan?", "Yen"),
        ("What is the currency of the UK?", "Pound sterling"),
        ("What is the currency of the EU?", "Euro"),
        ("How many strings does a standard guitar have?", "6"),
        ("How many keys does a standard piano have?", "88"),
        ("What is the speed of sound in air approximately?", "343 m/s"),
    ]

    for question, answer in CONCISE_SYSTEM_QA:
        if question in existing_prompts:
            continue
        rejected = make_list_version(question, answer)
        new_pairs.append(make_pair(question, answer, rejected))
        existing_prompts.add(question)

    # Verbose vs concise pairs for common questions
    VERBOSE_PAIRS = [
        (
            "What is the capital of France?",
            "Paris.",
            "The capital of France is a city called Paris, which is located in northern France along the Seine River. It has been the capital since the late 10th century and is known for the Eiffel Tower.",
        ),
        (
            "What is 2 + 2?",
            "4",
            "Well, to answer this question, we need to perform addition. When we add 2 and 2 together, we combine two groups of two items each, giving us a total of four items. Therefore, 2 + 2 = 4.",
        ),
        (
            "What color is the sky?",
            "Blue.",
            "The sky appears blue during the day. This is due to a phenomenon called Rayleigh scattering, where shorter blue wavelengths of light are scattered more than longer red wavelengths as sunlight passes through the atmosphere.",
        ),
        (
            "Who are you?",
            "I'm Yaya, an AI assistant.",
            "That's a great question! I am Yaya, which stands for Yet Another Yet Another... just kidding. I am an artificial intelligence language model built from scratch using PyTorch. I was trained on a variety of text data to be helpful, honest, and harmless. I am here to assist you with a wide range of tasks.",
        ),
        (
            "What is the boiling point of water?",
            "100°C (212°F) at sea level.",
            "Water boils at 100 degrees Celsius, which is also 212 degrees Fahrenheit. This occurs at standard atmospheric pressure (1 atm) at sea level. At higher altitudes, the boiling point is lower due to reduced atmospheric pressure.",
        ),
        (
            "Is 17 a prime number?",
            "Yes.",
            "To determine if 17 is prime, we need to check if it has any factors other than 1 and itself. Let's check: 17 is not divisible by 2 (it's odd), not by 3 (1+7=8, not divisible by 3), not by 5 (doesn't end in 0 or 5). Since √17 ≈ 4.1, we only need to check up to 4. Therefore, 17 is indeed a prime number.",
        ),
        (
            "What is the opposite of hot?",
            "Cold.",
            "The opposite or antonym of 'hot' is 'cold'. These two words describe temperature extremes on opposite ends of the spectrum. 'Hot' refers to high temperatures while 'cold' refers to low temperatures.",
        ),
        (
            "What is 12 x 12?",
            "144",
            "To calculate 12 times 12, we can use multiplication. 12 x 12 = 12 x 10 + 12 x 2 = 120 + 24 = 144. Alternatively, 12 squared is a commonly memorized multiplication fact: 12² = 144.",
        ),
    ]

    for question, chosen, rejected in VERBOSE_PAIRS:
        key = question + " [verbose]"
        if key in existing_prompts:
            continue
        new_pairs.append(make_pair(question, chosen, rejected))
        existing_prompts.add(key)

    print(f"New pairs generated: {len(new_pairs)}")

    # Append new pairs
    with open(DPO_OUT, "a", encoding="utf-8") as f:
        for p in new_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    total = len(existing_pairs) + len(new_pairs)
    print(f"Total DPO pairs now: {total}")
    print(f"Saved → {DPO_OUT}")


if __name__ == "__main__":
    main()
