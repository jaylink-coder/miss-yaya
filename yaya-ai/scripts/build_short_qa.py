"""Build a short Q&A dataset to teach Yaya direct answers.

The model currently produces long math derivations for every question.
This dataset teaches: short question → short direct answer.

Usage:
    python scripts/build_short_qa.py
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT = "data/sft/yaya_short_qa.jsonl"
SYSTEM = "You are Yaya, a helpful, honest, and friendly AI assistant. You answer questions clearly and thoughtfully."


def make(q, a):
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": q},
        {"role": "assistant", "content": a},
    ]}


def build():
    examples = []

    # ── 1. Basic arithmetic — all combos 1-20 ────────────────────────────────
    for a in range(1, 21):
        for b in range(1, 21):
            examples.append(make(f"What is {a} + {b}?", str(a + b)))
            examples.append(make(f"{a} + {b} = ?", str(a + b)))
            if a >= b:
                examples.append(make(f"What is {a} - {b}?", str(a - b)))
            examples.append(make(f"What is {a} x {b}?", str(a * b)))
            examples.append(make(f"{a} times {b}?", str(a * b)))
            if b != 0 and (a * b) % b == 0:
                examples.append(make(f"What is {a*b} divided by {b}?", str(a)))

    # ── 2. Larger arithmetic ──────────────────────────────────────────────────
    for a in [25, 50, 75, 100, 150, 200, 250, 500, 1000]:
        for b in [2, 4, 5, 10, 25, 50]:
            examples.append(make(f"What is {a} + {b}?", str(a + b)))
            examples.append(make(f"What is {a} x {b}?", str(a * b)))
            if a > b:
                examples.append(make(f"What is {a} - {b}?", str(a - b)))
            if a % b == 0:
                examples.append(make(f"What is {a} divided by {b}?", str(a // b)))

    # ── 3. Percentages ────────────────────────────────────────────────────────
    for pct in [10, 15, 20, 25, 50, 75]:
        for base in [100, 200, 400, 50, 80]:
            val = pct * base // 100
            examples.append(make(f"What is {pct}% of {base}?", str(val)))
            examples.append(make(f"Calculate {pct} percent of {base}.", str(val)))

    # ── 4. Squares and roots ──────────────────────────────────────────────────
    for n in range(1, 21):
        examples.append(make(f"What is {n} squared?", str(n * n)))
        examples.append(make(f"What is {n} to the power of 2?", str(n * n)))
        examples.append(make(f"What is the square root of {n*n}?", str(n)))

    # ── 5. Factual Q&A ────────────────────────────────────────────────────────
    facts = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Kenya?", "Nairobi"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of the United States?", "Washington, D.C."),
        ("What is the capital of China?", "Beijing"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is the capital of Brazil?", "Brasilia"),
        ("What is the capital of India?", "New Delhi"),
        ("What is the capital of Russia?", "Moscow"),
        ("What is the capital of the UK?", "London"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What color is the sky?", "Blue"),
        ("What color is grass?", "Green"),
        ("What color is snow?", "White"),
        ("What color is the sun?", "Yellow"),
        ("How many days are in a week?", "7"),
        ("How many months are in a year?", "12"),
        ("How many hours are in a day?", "24"),
        ("How many minutes are in an hour?", "60"),
        ("How many seconds are in a minute?", "60"),
        ("How many days are in a year?", "365"),
        ("What planet do we live on?", "Earth"),
        ("What is the largest planet in the solar system?", "Jupiter"),
        ("What is the largest ocean?", "The Pacific Ocean"),
        ("What is the largest continent?", "Asia"),
        ("How many continents are there?", "7"),
        ("What is H2O?", "Water"),
        ("What is the chemical formula for water?", "H2O"),
        ("What gas do humans breathe to survive?", "Oxygen"),
        ("What is the boiling point of water?", "100 degrees Celsius"),
        ("What is the freezing point of water?", "0 degrees Celsius"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What is the chemical symbol for iron?", "Fe"),
        ("How many bones are in the human body?", "206"),
        ("What is the powerhouse of the cell?", "The mitochondria"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("In what year did World War II end?", "1945"),
        ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
        ("What is the largest country by area?", "Russia"),
        ("What is the longest river in the world?", "The Nile"),
        ("What is the tallest mountain in the world?", "Mount Everest"),
        ("How many legs does a spider have?", "8"),
        ("How many legs does an insect have?", "6"),
        ("How many sides does a triangle have?", "3"),
        ("How many sides does a square have?", "4"),
        ("How many sides does a pentagon have?", "5"),
        ("How many sides does a hexagon have?", "6"),
        ("What is the sum of angles in a triangle?", "180 degrees"),
        ("What is Pi to 2 decimal places?", "3.14"),
        ("Is 17 a prime number?", "Yes"),
        ("Is 15 a prime number?", "No, 15 = 3 x 5"),
        ("Is 2 a prime number?", "Yes"),
        ("Is 1 a prime number?", "No"),
        ("Is the Earth round?", "Yes"),
        ("Is the Sun a star?", "Yes"),
        ("Is Pluto a planet?", "No, Pluto is a dwarf planet"),
    ]
    for q, a in facts:
        examples.append(make(q, a))

    # ── 6. Identity ───────────────────────────────────────────────────────────
    identity = [
        ("Who are you?", "I am Yaya, a helpful AI assistant."),
        ("What is your name?", "My name is Yaya."),
        ("What are you?", "I am Yaya, an AI assistant."),
        ("Are you an AI?", "Yes, I am Yaya, an AI assistant."),
        ("Are you human?", "No, I am Yaya, an AI assistant, not a human."),
        ("Are you ChatGPT?", "No, I am Yaya, a different AI assistant."),
        ("Are you Claude?", "No, I am Yaya, a different AI assistant."),
        ("Who made you?", "I was built from scratch as Yaya, an AI assistant."),
        ("What can you help with?", "I can help with questions, math, explanations, and conversations."),
        ("Hello!", "Hello! I am Yaya. How can I help you?"),
        ("Hi there", "Hi! I am Yaya. What can I do for you?"),
        ("Good morning", "Good morning! How can I help you today?"),
        ("Thank you", "You are welcome!"),
        ("Thanks", "Happy to help!"),
    ]
    for q, a in identity:
        examples.append(make(q, a))

    # ── 7. Simple logic and language ──────────────────────────────────────────
    logic = [
        ("Which is bigger: the Sun or the Earth?", "The Sun"),
        ("Which is faster: a car or a bicycle?", "A car"),
        ("Which is heavier: 1 kg of feathers or 1 kg of iron?", "They weigh the same — both are 1 kilogram."),
        ("Is water wet?", "Yes"),
        ("Is fire hot?", "Yes"),
        ("Is ice cold?", "Yes"),
        ("Can fish breathe underwater?", "Yes"),
        ("Can humans breathe underwater?", "No, not without equipment."),
        ("What comes after Monday?", "Tuesday"),
        ("What comes after Friday?", "Saturday"),
        ("What is the opposite of hot?", "Cold"),
        ("What is the opposite of big?", "Small"),
        ("What is the opposite of fast?", "Slow"),
        ("What is the opposite of day?", "Night"),
        ("What is the opposite of true?", "False"),
        ("What is the opposite of black?", "White"),
        ("All dogs are animals. Rex is a dog. Is Rex an animal?", "Yes, Rex is an animal."),
        ("If today is Monday, what day is tomorrow?", "Tuesday"),
        ("If today is Friday, what day was yesterday?", "Thursday"),
        ("What language is this sentence written in?", "English"),
    ]
    for q, a in logic:
        examples.append(make(q, a))

    # ── 8. Simple word problems (short answers) ───────────────────────────────
    word_problems = [
        ("If I have 10 apples and give away 3, how many do I have left?", "7"),
        ("If I have 5 apples and buy 4 more, how many do I have?", "9"),
        ("A shirt costs $15. I pay $20. How much change do I get?", "$5"),
        ("A dozen eggs costs $3. How much do 2 dozen cost?", "$6"),
        ("There are 30 students in a class. 12 are boys. How many are girls?", "18"),
        ("A pizza is cut into 8 slices. I eat 3. How many are left?", "5"),
        ("If a train travels at 60 km/h for 2 hours, how far does it go?", "120 km"),
        ("What is half of 84?", "42"),
        ("What is double 37?", "74"),
        ("If I save $10 a week, how much do I save in 4 weeks?", "$40"),
        ("A rectangle is 6 cm long and 4 cm wide. What is its area?", "24 square centimeters"),
        ("What is the perimeter of a square with side 5?", "20"),
        ("What is 15% of 200?", "30"),
        ("What is 10% of 150?", "15"),
        ("What is 25% of 80?", "20"),
        ("If 5 machines make 5 widgets in 5 minutes, how long for 1 machine to make 1 widget?", "5 minutes"),
        ("John has 3 bags with 4 apples each. How many apples in total?", "12"),
        ("A book has 200 pages. I read 50. How many pages are left?", "150"),
        ("There are 7 days in a week. How many days in 3 weeks?", "21"),
        ("If I wake up at 7am and sleep 8 hours, when should I go to sleep?", "11pm"),
    ]
    for q, a in word_problems:
        examples.append(make(q, a))

    # ── 9. Extended reasoning ─────────────────────────────────────────────────
    reasoning = [
        ("A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?", "$0.05"),
        ("A lily pad doubles every day. After 48 days it covers the lake. When was it half covered?", "Day 47"),
        ("Is 17 a prime number?", "Yes"),
        ("Is 21 a prime number?", "No, 21 = 3 x 7"),
        ("Is 29 a prime number?", "Yes"),
        ("Is 100 a prime number?", "No"),
        ("All cats are animals. Whiskers is a cat. Is Whiskers an animal?", "Yes"),
        ("All birds have wings. A penguin is a bird. Does a penguin have wings?", "Yes"),
        ("If A is bigger than B, and B is bigger than C, which is smallest?", "C"),
        ("If it rains, the ground is wet. The ground is wet. Did it rain?", "Not necessarily — the ground could be wet for other reasons."),
        ("What is the next number in the sequence: 2, 4, 6, 8?", "10"),
        ("What is the next number in the sequence: 1, 3, 5, 7?", "9"),
        ("What is the next number in the sequence: 1, 2, 4, 8?", "16"),
        ("If you have 3 boxes with 5 items each, how many items total?", "15"),
        ("A clock shows 3:00. What is the angle between the hands?", "90 degrees"),
        ("How many minutes in 2 hours?", "120"),
        ("How many seconds in 2 minutes?", "120"),
        ("How many hours in 3 days?", "72"),
        ("If you flip a fair coin, what is the chance of heads?", "50%"),
        ("What comes next: Monday, Tuesday, Wednesday?", "Thursday"),
    ]
    for q, a in reasoning:
        examples.append(make(q, a))

    # ── 10. Language understanding ────────────────────────────────────────────
    language = [
        ("What is the opposite of tall?", "Short"),
        ("What is the opposite of loud?", "Quiet"),
        ("What is the opposite of open?", "Closed"),
        ("What is the opposite of east?", "West"),
        ("What is the opposite of north?", "South"),
        ("What is the opposite of up?", "Down"),
        ("What is the opposite of happy?", "Sad"),
        ("What is the opposite of love?", "Hate"),
        ("What is the opposite of light?", "Dark"),
        ("What is the opposite of old?", "Young"),
        ("Which is bigger: the Sun or the Moon?", "The Sun"),
        ("Which is longer: a kilometer or a mile?", "A mile"),
        ("Which is heavier: a kilogram or a pound?", "A kilogram"),
        ("What language do people speak in France?", "French"),
        ("What language do people speak in Germany?", "German"),
        ("What language do people speak in Brazil?", "Portuguese"),
        ("What language do people speak in Japan?", "Japanese"),
        ("What language do people speak in Kenya?", "Swahili and English"),
        ("What is the plural of mouse?", "Mice"),
        ("What is the plural of child?", "Children"),
        ("What is the plural of tooth?", "Teeth"),
        ("What is the plural of foot?", "Feet"),
        ("What rhymes with cat?", "Bat, hat, mat, rat, sat"),
        ("What is a synonym for happy?", "Joyful"),
        ("What is a synonym for big?", "Large"),
        ("What is a synonym for fast?", "Quick"),
        ("Complete the sentence: The sky is ___.", "Blue"),
        ("Complete the sentence: Water is ___.", "Wet"),
        ("Complete the sentence: Fire is ___.", "Hot"),
        ("Spell the word 'necessary'.", "N-E-C-E-S-S-A-R-Y"),
    ]
    for q, a in language:
        examples.append(make(q, a))

    # Deduplicate by question
    seen, deduped = set(), []
    for ex in examples:
        key = ex["messages"][1]["content"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    random.shuffle(deduped)
    return deduped


def main():
    os.makedirs("data/sft", exist_ok=True)
    examples = build()

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    avg_len = sum(len(ex["messages"][2]["content"]) for ex in examples) / len(examples)
    print(f"Short Q&A dataset: {len(examples):,} examples")
    print(f"Average answer length: {avg_len:.1f} chars")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
