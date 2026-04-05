"""Generate targeted patch dataset for DPO2 failures.

Covers:
- Factual confusion (planet=Earth not Blue, language=English not Paris)
- Language questions (opposites, language identification)
- Word problems (arithmetic in context)
- Reasoning with direct final answers
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYSTEM = "You are Yaya, a helpful and honest AI assistant. You answer questions clearly and concisely."


def make(q, a):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": q},
        {"role": "assistant", "content": a},
    ]}


examples = []

# ── FACTUAL PATCH ────────────────────────────────────────────────────────────
# Planet confusion (model answers "Blue" — need to teach Earth != Blue)
examples += [
    make("What planet do we live on?", "Earth"),
    make("Which planet is home to humans?", "Earth"),
    make("Name the planet where humans live.", "Earth"),
    make("What is the name of our planet?", "Earth"),
    make("What planet is Earth?", "Earth is the third planet from the Sun."),
    make("Is Earth a blue planet?", "Earth is sometimes called the Blue Planet because of its oceans, but its name is Earth."),
    make("What do we call the planet we live on?", "Earth"),
    make("Humans live on which planet?", "Earth"),
    make("What is the third planet from the Sun?", "Earth"),
    make("What color is the sky?", "Blue"),
    make("Why is the sky blue?", "The sky appears blue because of how sunlight scatters through the atmosphere."),
    make("The sky is what color?", "Blue"),
]

# Months confusion (model answers "180")
examples += [
    make("How many months are in a year?", "12"),
    make("How many months does a year have?", "12"),
    make("A year has how many months?", "12"),
    make("Count the months in a year.", "12"),
    make("How many months is one year?", "12 months"),
    make("January through December — how many months is that?", "12"),
    make("If January is month 1 and December is month 12, how many months are there?", "12"),
    make("How many weeks are in a year?", "52"),
    make("How many days are in a year?", "365 (366 in a leap year)"),
    make("How many hours are in a day?", "24"),
]

# Kenya capital
examples += [
    make("What is the capital of Kenya?", "Nairobi"),
    make("Name the capital city of Kenya.", "Nairobi"),
    make("Kenya's capital city is?", "Nairobi"),
    make("Which city is the capital of Kenya?", "Nairobi"),
    make("What city serves as Kenya's capital?", "Nairobi"),
]

# Water formula
examples += [
    make("What is the chemical formula for water?", "H2O"),
    make("What is water's chemical formula?", "H2O"),
    make("Write the chemical formula for water.", "H2O"),
    make("Water is made of which elements?", "Hydrogen and oxygen (H2O)"),
    make("What does H2O stand for?", "Water (2 hydrogen atoms + 1 oxygen atom)"),
    make("What is the molecular formula of water?", "H2O"),
    make("What is the chemical formula for salt?", "NaCl"),
    make("What is the chemical formula for carbon dioxide?", "CO2"),
]

# Boiling point
examples += [
    make("At what temperature does water boil?", "100 degrees Celsius (212 degrees Fahrenheit)"),
    make("What is the boiling point of water?", "100 degrees Celsius at sea level"),
    make("Water boils at what temperature?", "100 degrees Celsius"),
    make("At how many degrees Celsius does water boil?", "100 degrees Celsius"),
    make("What temperature is needed to boil water?", "100 degrees Celsius (212 degrees Fahrenheit)"),
    make("Does water boil at 100 degrees Celsius?", "Yes, water boils at 100 degrees Celsius at standard pressure."),
    make("At what temperature does water freeze?", "0 degrees Celsius (32 degrees Fahrenheit)"),
]

# World capitals (diversity to prevent pattern mixing)
examples += [
    make("What is the capital of France?", "Paris"),
    make("What is the capital of Japan?", "Tokyo"),
    make("What is the capital of Germany?", "Berlin"),
    make("What is the capital of Australia?", "Canberra"),
    make("What is the capital of Brazil?", "Brasilia"),
    make("What is the capital of China?", "Beijing"),
    make("What is the capital of Nigeria?", "Abuja"),
    make("What is the capital of South Africa?", "Pretoria"),
    make("What is the capital of India?", "New Delhi"),
    make("What is the capital of Canada?", "Ottawa"),
    make("What is the capital of Egypt?", "Cairo"),
    make("What is the capital of Mexico?", "Mexico City"),
    make("What is the capital of Tanzania?", "Dodoma"),
    make("What is the capital of Uganda?", "Kampala"),
    make("What is the capital of Ethiopia?", "Addis Ababa"),
    make("What is the capital of Italy?", "Rome"),
    make("What is the capital of Spain?", "Madrid"),
    make("What is the capital of the United States?", "Washington, D.C."),
    make("What is the capital of the United Kingdom?", "London"),
    make("What is the capital of Russia?", "Moscow"),
]

# ── LANGUAGE PATCH ──────────────────────────────────────────────────────────
# Opposite of hot (model echoed system prompt)
examples += [
    make("What is the opposite of hot?", "Cold"),
    make("Hot and ___ are opposites.", "Cold"),
    make("If hot means high temperature, the opposite is?", "Cold"),
    make("What word means the opposite of hot?", "Cold"),
    make("Give the opposite of hot.", "Cold"),
    make("What is the opposite of cold?", "Hot"),
    make("What is the opposite of fast?", "Slow"),
    make("What is the opposite of big?", "Small"),
    make("What is the opposite of light?", "Dark (or heavy, depending on context)"),
    make("What is the opposite of happy?", "Sad"),
    make("What is the opposite of yes?", "No"),
    make("What is the opposite of true?", "False"),
    make("What is the opposite of open?", "Closed"),
    make("What is the opposite of day?", "Night"),
    make("What is the opposite of tall?", "Short"),
    make("What is the opposite of east?", "West"),
    make("What is the opposite of love?", "Hate"),
    make("What is the opposite of strong?", "Weak"),
    make("What is the opposite of empty?", "Full"),
    make("What is the opposite of old?", "Young (or new)"),
    make("What is the opposite of black?", "White"),
    make("What is the opposite of loud?", "Quiet"),
    make("What is the opposite of hard?", "Soft"),
    make("What is the opposite of early?", "Late"),
    make("What is the opposite of good?", "Bad"),
    make("What is the opposite of clean?", "Dirty"),
]

# Language identification (model answers "Paris")
examples += [
    make("What language is this sentence written in?", "English"),
    make("This sentence is written in what language?", "English"),
    make("What language am I using to write this question?", "English"),
    make("Identify the language: Hello, how are you?", "English"),
    make("Identify the language: Bonjour, comment allez-vous?", "French"),
    make("Identify the language: Hola como estas?", "Spanish"),
    make("Identify the language: Guten Morgen", "German"),
    make("What language is 'merci beaucoup' from?", "French"),
    make("What language is 'gracias' from?", "Spanish"),
    make("What language is 'arigatou' from?", "Japanese"),
    make("Is this written in English: The sky is blue?", "Yes, it is written in English."),
    make("What language do people speak in France?", "French"),
    make("What language do people speak in Spain?", "Spanish"),
    make("What language do people speak in Germany?", "German"),
    make("What language do people speak in Japan?", "Japanese"),
    make("What language do people speak in Brazil?", "Portuguese"),
    make("What language do people speak in China?", "Mandarin Chinese"),
    make("What language do people speak in Kenya?", "Swahili and English"),
    make("What is the official language of the United States?", "English"),
    make("Is Spanish a language?", "Yes, Spanish is a widely spoken language."),
]

# ── WORD PROBLEMS PATCH ─────────────────────────────────────────────────────
# Subtraction (10-3=5 was wrong, should be 7)
examples += [
    make("If I have 10 apples and give away 3, how many do I have left?", "7"),
    make("Sarah has 10 cookies and eats 3. How many are left?", "7"),
    make("There were 10 birds on a wire. 3 flew away. How many remain?", "7"),
    make("10 minus 3 equals what?", "7"),
    make("If you have 15 and spend 6, how much remains?", "9"),
    make("A bag has 20 marbles. You take out 8. How many are left?", "12"),
    make("Tom has 25 stickers. He gives 10 to his friend. How many does he keep?", "15"),
    make("There are 50 students. 18 go home early. How many are left?", "32"),
    make("A store had 100 items. They sold 45. How many remain?", "55"),
    make("If you have 7 apples and eat 2, how many are left?", "5"),
    make("14 minus 9 is?", "5"),
]

# Speed/distance (60 km/h * 2h = 180 was wrong, should be 120)
examples += [
    make("A car travels at 60 km/h for 2 hours. How far does it go?", "120 km"),
    make("Distance = speed times time. Speed is 60 km/h, time is 2 hours. What is the distance?", "120 km"),
    make("If a car drives 60 km per hour for 2 hours, what is the total distance?", "120 km"),
    make("60 km/h for 2 hours: what distance?", "120 km"),
    make("A train moves at 80 km/h for 3 hours. How far does it travel?", "240 km"),
    make("Walking at 5 km/h for 4 hours, how far do you walk?", "20 km"),
    make("Speed: 30 km/h. Time: 2 hours. Distance = ?", "60 km"),
    make("A runner goes 10 km/h for 1 hour. Distance?", "10 km"),
    make("Distance formula: distance = speed x time. If speed=50 and time=3, distance=?", "150"),
]

# General word problems
examples += [
    make("A dozen eggs is how many?", "12"),
    make("If apples cost $2 each and I buy 5, how much do I spend?", "$10"),
    make("There are 7 days in a week. How many days in 2 weeks?", "14"),
    make("A class has 30 students. Half are girls. How many girls are there?", "15"),
    make("If a pizza is cut into 8 slices and 3 are eaten, how many remain?", "5"),
    make("A farmer has 12 cows and sells 4. How many are left?", "8"),
    make("You earn $100 and spend $35. How much is left?", "$65"),
    make("A jar holds 20 candies. 7 are eaten. How many remain?", "13"),
]

# ── REASONING PATCH ─────────────────────────────────────────────────────────
# Lily pad doubling — direct final answer
examples += [
    make("A lily pad doubles every day. After 48 days it covers the lake. When was it half covered?", "Day 47"),
    make("Something doubles each day and fills a container in 10 days. When was it half full?", "Day 9"),
    make("A bacteria colony doubles hourly. After 24 hours it fills a dish. When was it half full?", "Hour 23"),
    make("If a number doubles each step and reaches 1024 at step 10, what was it at step 9?", "512"),
    make("If something doubles daily and is full on day 30, when is it half full?", "Day 29"),
]

# Logic and reasoning
examples += [
    make("A rooster lays an egg on top of a pointed roof. Which way does the egg roll?", "Roosters do not lay eggs."),
    make("What has keys but no locks, space but no room?", "A keyboard"),
    make("I am always in front of you but cannot be seen. What am I?", "The future"),
    make("What gets wetter as it dries?", "A towel"),
    make("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "5 minutes"),
    make("Mary's father has 5 daughters: Nana, Nene, Nini, Nono. What is the fifth daughter's name?", "Mary"),
    make("How many seconds are in a minute?", "60"),
    make("How many minutes are in an hour?", "60"),
    make("How many hours are in a week?", "168"),
]

# ── WRITE OUTPUT ─────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "sft", "yaya_patch_sft.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(examples)} patch examples -> {out_path}")
