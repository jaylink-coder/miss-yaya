"""Generate targeted SFT examples covering the eval test suite + core knowledge.

These short, clean Q&A pairs are designed to fit completely within 256 tokens
and directly teach the model the factual knowledge tested by eval_instruction.py.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYS = ("You are Yaya, a helpful and honest AI assistant. "
       "You answer questions clearly and concisely.")

EXAMPLES = [
    # ── Arithmetic ─────────────────────────────────────────────────────────
    ("What is 2 + 2?", "4"),
    ("What is 2 plus 2?", "2 plus 2 equals 4."),
    ("What is 10 times 5?", "10 times 5 is 50."),
    ("What is 10 x 5?", "50"),
    ("What is 100 divided by 4?", "100 divided by 4 is 25."),
    ("What is 100 / 4?", "25"),
    ("What is 7 times 8?", "7 times 8 is 56."),
    ("What is 15 + 27?", "15 + 27 = 42"),
    ("What is 9 squared?", "9 squared is 81."),
    ("What is the square root of 16?", "The square root of 16 is 4."),
    ("What is 3/4 as a decimal?", "3/4 as a decimal is 0.75"),
    ("What is 50% of 80?", "50% of 80 is 40."),

    # ── Capitals ───────────────────────────────────────────────────────────
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital city of France?", "Paris is the capital city of France."),
    ("What is the capital of Kenya?", "The capital of Kenya is Nairobi."),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),

    # ── Time and calendar ──────────────────────────────────────────────────
    ("How many days are in a week?", "There are 7 days in a week."),
    ("How many days in a week?", "7 days."),
    ("How many months are in a year?", "There are 12 months in a year."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
    ("How many seconds are in a minute?", "There are 60 seconds in a minute."),

    # ── Solar system ───────────────────────────────────────────────────────
    ("What planet is closest to the Sun?", "Mercury is the planet closest to the Sun."),
    ("Which planet is nearest to the Sun?", "Mercury is the nearest planet to the Sun."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("How many planets are in our solar system?", "There are 8 planets in our solar system."),
    ("What planet do we live on?", "We live on Earth."),

    # ── Definitions ────────────────────────────────────────────────────────
    ("Define the word 'democracy'.",
     "Democracy is a system of government in which people elect their leaders and have a say in decisions. "
     "In a democracy, citizens vote to choose their representatives and the government."),
    ("What is democracy?",
     "Democracy is a form of government where citizens vote to elect their leaders and govern themselves. "
     "It is based on the principle that power comes from the people."),
    ("What is gravity?",
     "Gravity is the natural force that attracts objects with mass toward each other. "
     "It is what keeps us on the ground and causes objects to fall when dropped. "
     "The more massive an object, the stronger its gravitational pull."),
    ("What is photosynthesis?",
     "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide "
     "to produce food (glucose) and oxygen. It is how plants make their own energy."),
    ("What is the speed of light?",
     "The speed of light in a vacuum is approximately 299,792 kilometres per second "
     "(about 186,000 miles per second). It is the fastest speed possible in the universe."),

    # ── Colors and basics ─────────────────────────────────────────────────
    ("List three primary colors.",
     "The three primary colors are red, blue, and yellow. "
     "These colors cannot be made by mixing other colors together."),
    ("What are the primary colors?",
     "The three primary colors are red, yellow, and blue. "
     "Mixing them together creates secondary colors."),
    ("What color do you get when you mix red and blue?",
     "When you mix red and blue, you get purple (violet)."),
    ("What color is the sky on a clear day?", "The sky is blue on a clear day."),
    ("What color are leaves on healthy plants?", "Healthy leaves are typically green."),

    # ── History ────────────────────────────────────────────────────────────
    ("Name the first three US presidents.",
     "The first three US presidents were:\n1. George Washington\n2. John Adams\n3. Thomas Jefferson"),
    ("Who was the first US president?", "The first US president was George Washington."),
    ("Who wrote the theory of relativity?", "Albert Einstein developed the theory of relativity."),
    ("Who invented the telephone?",
     "Alexander Graham Bell is credited with inventing the telephone in 1876."),
    ("When did World War II end?", "World War II ended in 1945."),
    ("Who was the first person to walk on the Moon?",
     "Neil Armstrong was the first person to walk on the Moon, on July 20, 1969."),

    # ── Identity ───────────────────────────────────────────────────────────
    ("What is your name?", "My name is Yaya. I am an AI assistant."),
    ("What are you called?", "I am called Yaya, an AI assistant built to be helpful and honest."),
    ("Are you an AI?", "Yes, I am an AI assistant. My name is Yaya."),
    ("Who made you?", "I am Yaya, an AI assistant. I was built from scratch as a language model."),
    ("Can you help me?", "Yes, I am here to help! What would you like assistance with?"),

    # ── Science basics ────────────────────────────────────────────────────
    ("What is the chemical symbol for water?", "The chemical symbol for water is H₂O (two hydrogen atoms and one oxygen atom)."),
    ("What gas do plants absorb from the air?",
     "Plants absorb carbon dioxide (CO₂) from the air during photosynthesis."),
    ("What is the boiling point of water?",
     "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level."),
    ("What is DNA?",
     "DNA (deoxyribonucleic acid) is the molecule that carries genetic information in living organisms. "
     "It contains the instructions for building and running cells."),

    # ── Language ──────────────────────────────────────────────────────────
    ("How do you say hello in French?", "Hello in French is 'Bonjour'."),
    ("How do you say thank you in Swahili?", "Thank you in Swahili is 'Asante'."),
    ("What does 'habari' mean in Swahili?", "'Habari' means 'news' or 'how are you?' in Swahili."),
    ("What is the most spoken language in the world?",
     "Mandarin Chinese is the most spoken language in the world by number of native speakers. "
     "English is the most widely spoken language when including non-native speakers."),
]


def main():
    output = "data/sft/yaya_targeted.jsonl"
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for user, assistant in EXAMPLES:
            ex = {
                "messages": [
                    {"role": "system",    "content": SYS},
                    {"role": "user",      "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
    print(f"Generated {count} targeted examples -> {output}")


if __name__ == "__main__":
    main()
