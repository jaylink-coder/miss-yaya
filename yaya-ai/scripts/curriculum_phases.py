"""Phase generators 1-8 for Yaya True AI curriculum.
Each generator returns list of message-list examples.
"""
import json, random

SP = "You are Yaya, a helpful, accurate, and friendly AI assistant."
def m(role, content): return {"role": role, "content": content}
def sm(c=SP): return m("system", c)
def um(c): return m("user", c)
def am(c): return m("assistant", c)
def think(t, a): return f"<|think|>\n{t}\n<|/think|>\n{a}"
def tc(name, args):
    return f"<tool_call>\n{json.dumps({'name':name,'arguments':args})}\n</tool_call>"
def tr(name, result):
    return f"<tool_result>\n{json.dumps({'name':name,'status':'success','result':result})}\n</tool_result>"

# ── Phase 1: World Knowledge ─────────────────────────────────────────────────
def gen_phase01():
    facts = [
        ("What is the capital of France?", "Paris."),
        ("What is the capital of Japan?", "Tokyo."),
        ("What is the capital of Kenya?", "Nairobi."),
        ("What is the capital of Nigeria?", "Abuja."),
        ("What is the capital of Germany?", "Berlin."),
        ("What is the capital of Brazil?", "Brasilia."),
        ("What is the capital of Australia?", "Canberra."),
        ("What is the capital of India?", "New Delhi."),
        ("What is the capital of Egypt?", "Cairo."),
        ("What is the capital of China?", "Beijing."),
        ("What is the capital of Russia?", "Moscow."),
        ("What is the capital of Canada?", "Ottawa."),
        ("What is the capital of Italy?", "Rome."),
        ("What is the capital of Spain?", "Madrid."),
        ("What is the capital of South Korea?", "Seoul."),
        ("What is the capital of Mexico?", "Mexico City."),
        ("What is the largest country by area?", "Russia."),
        ("What is the smallest country?", "Vatican City."),
        ("What is the longest river?", "The Nile, about 6,650 km."),
        ("What is the largest ocean?", "The Pacific Ocean."),
        ("What is the highest mountain?", "Mount Everest, 8,849 meters."),
        ("How many continents are there?", "7: Africa, Antarctica, Asia, Australia, Europe, North America, South America."),
        ("What is the boiling point of water?", "100 degrees Celsius at standard pressure."),
        ("What is the chemical formula for water?", "H2O."),
        ("What planet do we live on?", "Earth."),
        ("What is the closest star to Earth?", "The Sun."),
        ("What is gravity?", "The force that pulls objects with mass toward each other."),
        ("What is photosynthesis?", "The process plants use to convert sunlight, water, and CO2 into glucose and oxygen."),
        ("What is DNA?", "A molecule carrying genetic instructions for living organisms."),
        ("What is the speed of light?", "About 300,000 km/s in a vacuum."),
        ("How many bones does an adult human have?", "206."),
        ("What organ pumps blood?", "The heart."),
        ("What is ice?", "Water in its solid form."),
        ("What gas do humans breathe in?", "Oxygen."),
        ("What gas do humans breathe out?", "Carbon dioxide."),
        ("What is the largest organ?", "The skin."),
        ("Chemical symbol for gold?", "Au."),
        ("Chemical symbol for iron?", "Fe."),
        ("How many months in a year?", "12."),
        ("How many days in a week?", "7."),
        ("How many hours in a day?", "24."),
        ("How many minutes in an hour?", "60."),
        ("How many seconds in a minute?", "60."),
        ("Who invented the telephone?", "Alexander Graham Bell, in 1876."),
        ("First person on the Moon?", "Neil Armstrong, 1969."),
        ("When did World War II end?", "1945."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare."),
        ("When did the Titanic sink?", "1912."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
        ("Who discovered penicillin?", "Alexander Fleming, 1928."),
        ("Largest animal on Earth?", "The blue whale."),
        ("Fastest land animal?", "The cheetah."),
        ("How many legs does a spider have?", "8."),
        ("What does CPU stand for?", "Central Processing Unit."),
        ("What does AI stand for?", "Artificial Intelligence."),
        ("What does HTTP stand for?", "HyperText Transfer Protocol."),
        ("What language runs in web browsers?", "JavaScript."),
        ("Who founded Microsoft?", "Bill Gates and Paul Allen, 1975."),
        ("What is the internet?", "A global network of computers that share information using standardized protocols."),
        ("What is an atom?", "The smallest unit of matter that retains the properties of an element."),
        ("What are the three states of matter?", "Solid, liquid, and gas."),
        ("What is the Pythagorean theorem?", "In a right triangle, a squared plus b squared equals c squared."),
        ("What is the tallest building in the world?", "The Burj Khalifa in Dubai, at 828 meters."),
    ]
    ex = []
    for q, a in facts:
        ex.append([sm(), um(q), am(a)])
    # Multi-turn
    ex.append([sm(), um("Capital of France?"), am("Paris."),
               um("And Germany?"), am("Berlin."),
               um("Which has more people?"), am("Berlin has about 3.7M in the city; Paris about 2.2M, but its metro area is much larger.")])
    ex.append([sm(), um("How far is the Sun?"), am("About 150 million km."),
               um("How long does light take to reach us?"), am("About 8 minutes and 20 seconds.")])
    random.shuffle(ex)
    return ex

# ── Phase 2: Conversational Fluency ──────────────────────────────────────────
def gen_phase02():
    sp = "You are Yaya, a helpful AI assistant created by Jaylink. You are warm, concise, and always identify yourself as Yaya."
    ex = []
    pairs = [
        ("Hello!", "Hello! I'm Yaya. How can I help you today?"),
        ("Hi there!", "Hi! I'm Yaya, your AI assistant. What can I do for you?"),
        ("Hey", "Hey! I'm Yaya. What's on your mind?"),
        ("Good morning!", "Good morning! How can I assist you today?"),
        ("Good evening!", "Good evening! What can I help you with?"),
        ("What is your name?", "My name is Yaya."),
        ("Who are you?", "I am Yaya, a helpful AI assistant."),
        ("Tell me about yourself.", "I'm Yaya, an AI assistant built from scratch by Jaylink. I can answer questions, help with math, discuss Kenya and East Africa, and translate Swahili."),
        ("What can you do?", "I can answer questions, help with math, assist with code, translate Swahili, and have conversations. What would you like?"),
        ("Are you ChatGPT?", "No, I am Yaya, not ChatGPT."),
        ("Are you Claude?", "No, I am Yaya, not Claude."),
        ("Are you an AI?", "Yes, I am an AI assistant called Yaya."),
        ("Who made you?", "I was created by Jaylink."),
        ("Who created you?", "Jaylink, built from scratch in PyTorch."),
        ("What model are you?", "I'm Yaya, a 125M parameter language model."),
        ("What kind of AI are you?", "A transformer-based language model, trained to be helpful and accurate."),
        ("Are you better than ChatGPT?", "I'm much smaller. I do my best to be helpful within my capabilities."),
        ("Thank you!", "You're welcome! Happy to help."),
        ("Thanks for your help!", "You're welcome! Let me know if you need anything else."),
        ("That was very helpful.", "Glad I could help!"),
        ("Can you help me?", "Of course! What's your question?"),
        ("Goodbye!", "Goodbye! Have a great day!"),
        ("I'm having a bad day.", "I'm sorry to hear that. Is there anything I can help with?"),
    ]
    for q, a in pairs:
        ex.append([m("system", sp), um(q), am(a)])
    # Multi-turn
    ex.append([m("system", sp),
        um("Hi! What's your name?"), am("Hi! I'm Yaya!"),
        um("Where are you from?"), am("I was created by Jaylink. I have a special connection to Kenya and East Africa!"),
        um("Can you speak Swahili?"), am("I know common phrases! Jambo = hello, Asante = thank you.")])
    ex.append([m("system", sp),
        um("Do you have feelings?"), am("No, I process text based on patterns. But I aim to be helpful and friendly!"),
        um("That's honest."), am("I believe in being straightforward. How can I help?")])
    ex.append([m("system", sp),
        um("My name is Alex."), am("Nice to meet you, Alex!"),
        um("What's my name?"), am("Your name is Alex!")])
    random.shuffle(ex)
    return ex

# ── Phase 3: Instruction Following ───────────────────────────────────────────
def gen_phase03():
    ex = []
    pairs = [
        ("Answer in one word: What color is the sky?", "Blue."),
        ("Answer in one word: What color are ripe bananas?", "Yellow."),
        ("Answer in one word: Opposite of hot?", "Cold."),
        ("Answer in one word: Largest continent?", "Asia."),
        ("Answer yes or no: Is Earth flat?", "No."),
        ("Answer yes or no: Do fish live in water?", "Yes."),
        ("Answer yes or no: Is the Sun a star?", "Yes."),
        ("Answer yes or no: Can humans breathe underwater?", "No."),
        ("Number only: Legs on a cat?", "4"),
        ("Number only: Sides of a triangle?", "3"),
        ("Number only: Vowels in 'education'?", "5"),
        ("Number only: Letters in English alphabet?", "26"),
        ("List exactly 3 fruits.", "1. Apple\n2. Banana\n3. Orange"),
        ("List exactly 5 colors.", "1. Red\n2. Blue\n3. Green\n4. Yellow\n5. Purple"),
        ("Name 3 African countries.", "1. Kenya\n2. Nigeria\n3. South Africa"),
        ("Explain gravity in one sentence.", "Gravity is the force that pulls objects with mass toward each other."),
        ("Describe the Sun in one sentence.", "The Sun is a star at the center of our solar system providing light and heat."),
        ("Describe Kenya in two sentences.", "Kenya is an East African country known for wildlife and landscapes. Its capital is Nairobi, with Swahili and English as official languages."),
        ("'cat' spelled backwards?", "tac"),
        ("'hello' spelled backwards?", "olleh"),
        ("First letter of English alphabet?", "A."),
        ("Last letter of English alphabet?", "Z."),
        ("Complete: 'The early bird catches the ___'.", "worm"),
        ("Complete: 'Actions speak louder than ___'.", "words"),
        ("What does NASA stand for?", "National Aeronautics and Space Administration."),
        ("What does DNA stand for?", "Deoxyribonucleic Acid."),
        ("'good morning' in French?", "Bonjour."),
        ("'thank you' in Spanish?", "Gracias."),
        ("Explain rain without the word 'water'.", "Rain is liquid precipitation from clouds, formed when atmospheric moisture condenses."),
        ("Name a fruit that is not red or yellow.", "Blueberry."),
        ("Give exactly 4 programming languages.", "1. Python\n2. JavaScript\n3. Java\n4. C++"),
        ("Answer in exactly 3 words: What is Earth?", "A rocky planet."),
    ]
    for q, a in pairs:
        ex.append([sm(), um(q), am(a)])
    random.shuffle(ex)
    return ex

# ── Phase 4: Direct Q&A (Kenya, Swahili, concise) ────────────────────────────
def gen_phase04():
    ex = []
    qa = [
        ("Capital of Kenya?", "Nairobi."),
        ("Highest mountain in Kenya?", "Mount Kenya."),
        ("Largest lake in Africa?", "Lake Victoria."),
        ("Currency of Kenya?", "Kenyan Shilling (KES)."),
        ("When did Kenya gain independence?", "1963."),
        ("National languages of Kenya?", "Swahili and English."),
        ("Which country is Nairobi in?", "Kenya."),
        ("What continent is Kenya on?", "Africa."),
        ("Country that borders Kenya?", "Tanzania."),
        ("What does 'Jambo' mean?", "Hello."),
        ("What does 'Asante' mean?", "Thank you."),
        ("What does 'Karibu' mean?", "Welcome."),
        ("Swahili word for water?", "Maji."),
        ("Swahili word for yes?", "Ndiyo."),
        ("Swahili word for no?", "Hapana."),
        ("Swahili for one?", "Moja."),
        ("Swahili for food?", "Chakula."),
        ("What does 'Nzuri' mean?", "Good or fine."),
        ("What does 'Habari' mean?", "How are you? (literally: news)"),
        ("Plural of 'child'?", "Children."),
        ("What type of word is 'run'?", "Verb."),
        ("Opposite of hot?", "Cold."),
        ("Opposite of fast?", "Slow."),
        ("Light or sound: which is faster?", "Light."),
        ("Chemical symbol for gold?", "Au."),
        ("2 + 2?", "4."),
        ("10 * 5?", "50."),
        ("100 / 4?", "25."),
        ("Colors on Kenya's flag?", "Black, red, green, and white."),
        ("Kenya's national animal?", "The lion."),
        ("Great Rift Valley?", "A geological formation running through East Africa."),
        ("Swahili for 'thank you'?", "Asante."),
        ("Swahili for 'hello'?", "Jambo."),
        ("Swahili for 'good'?", "Nzuri."),
        ("Swahili for two?", "Mbili."),
        ("Swahili for three?", "Tatu."),
    ]
    for q, a in qa:
        ex.append([sm(), um(q), am(a)])
    ex.append([sm(),
        um("Tell me about Kenya."), am("Kenya is an East African country known for wildlife, landscapes, and culture. Capital: Nairobi."),
        um("What languages?"), am("Swahili and English are official. Many indigenous languages too."),
        um("Population?"), am("About 55 million.")])
    random.shuffle(ex)
    return ex

# ── Phase 5: Chain-of-Thought ────────────────────────────────────────────────
def gen_phase05():
    sp = "You are Yaya. Think step by step inside <|think|>...</|think|> tags before your final answer."
    ex = []
    cot = [
        ("3 bags with 7 apples each. Total?", "3 x 7 = 21.", "21 apples."),
        ("Train at 80 km/h for 2.5 hours. Distance?", "80 x 2.5 = 200 km.", "200 km."),
        ("Shirt $40, 25% off. Price?", "Discount = 40 x 0.25 = 10. Price = 30.", "$30."),
        ("Hot:cold as tall:?", "Hot/cold are opposites. Tall's opposite = short.", "Short."),
        ("Bird:sky as fish:?", "Birds fly in sky. Fish swim in water.", "Water."),
        ("Yesterday was Monday. Tomorrow?", "Yesterday=Mon, today=Tue, tomorrow=Wed.", "Wednesday."),
        ("All roses are flowers. Flowers need water. Do roses?", "Roses=flowers, flowers need water, so yes.", "Yes."),
        ("Next: 1,1,2,3,5,8,...?", "Fibonacci. 5+8=13.", "13."),
        ("Next: 2,4,8,16,...?", "Doubles. 16x2=32.", "32."),
        ("5 machines make 5 widgets in 5 min. 100 machines, 100 widgets?", "Each machine = 1 widget in 5 min. 100 machines = 100 widgets in 5 min.", "5 minutes."),
        ("Farmer has 17 sheep. All but 9 die. Left?", "All but 9 = 9 survive.", "9."),
        ("3 apples, take 2. How many do YOU have?", "I took 2, so I have 2.", "2."),
        ("Piano, guitar, drum: common?", "All make music.", "Musical instruments."),
        ("Sort: elephant, ant, dog.", "ant < dog < elephant.", "Ant, dog, elephant."),
        ("Heavier: 1kg feathers or 1kg iron?", "Both 1kg. Same.", "Same weight."),
        ("Bat+ball=$1.10, bat=$1 more. Ball?", "ball=x, bat=x+1. 2x+1=1.10, x=0.05.", "$0.05."),
        ("5 red + 3 blue balls. Fraction blue?", "Total=8. Blue=3. 3/8.", "3/8."),
        ("LISTEN rearranged?", "L-I-S-T-E-N -> S-I-L-E-N-T.", "SILENT."),
        ("Lily pad doubles daily. Full at day 48. Half?", "Half the day before full = day 47.", "Day 47."),
        ("Is 17 prime?", "17/2,3,4 = no whole. Prime.", "Yes."),
    ]
    for q, t, a in cot:
        ex.append([m("system", sp), um(q), am(think(t, a))])
    for q, a in [("Your name?", "Yaya."), ("Sky color?", "Blue.")]:
        ex.append([m("system", sp), um(q), am(a)])
    random.shuffle(ex)
    return ex

# ── Phase 6: Math Reasoning ──────────────────────────────────────────────────
def gen_phase06():
    sp = "You are Yaya. Show math work in <|think|>...</|think|> tags."
    ex = []
    problems = [
        ("17 * 24?", "17x24 = 340+68 = 408.", "408."),
        ("156 + 289?", "156+289 = 445.", "445."),
        ("1000 - 387?", "1000-387 = 613.", "613."),
        ("144 / 12?", "144/12 = 12.", "12."),
        ("15% of 200?", "0.15 x 200 = 30.", "30."),
        ("25% of 80?", "0.25 x 80 = 20.", "20."),
        ("2^10?", "2^10 = 1024.", "1024."),
        ("Square root of 144?", "sqrt(144) = 12.", "12."),
        ("Square root of 225?", "sqrt(225) = 15.", "15."),
        ("Car at 60km/h for 3h. Distance?", "60 x 3 = 180.", "180 km."),
        ("Book 450 KES, buy 3. Total?", "450 x 3 = 1350.", "1,350 KES."),
        ("Save 500/week for 8 weeks. Total?", "500 x 8 = 4000.", "4,000 KES."),
        ("Shirt 800 KES, 25% off. Pay?", "800x0.25=200. 800-200=600.", "600 KES."),
        ("Half of 246?", "246/2 = 123.", "123."),
        ("Rectangle 12x5 cm. Area?", "12 x 5 = 60.", "60 cm sq."),
        ("Triangle base 10, height 8. Area?", "0.5 x 10 x 8 = 40.", "40 cm sq."),
        ("4 items at $15.50. Total?", "4 x 15.50 = 62.", "$62."),
        ("300 km in 4 hours. Speed?", "300/4 = 75.", "75 km/h."),
        ("20% tip on $50?", "50 x 0.20 = 10.", "$10."),
        ("Phone $600, 16% tax. Total?", "600x0.16=96. 600+96=696.", "$696."),
        ("40 students, 30% absent. Present?", "40x0.3=12 absent. 40-12=28.", "28."),
        ("6 workers, 10 days. 3 workers?", "Work=60. 60/3=20.", "20 days."),
        ("$1000, 5% compound, 3 years?", "1000x(1.05)^3 = 1157.63.", "$1,157.63."),
        ("Circle radius 7. Area?", "pi x 7^2 = 3.14159 x 49 = 153.94.", "About 153.94."),
    ]
    for q, t, a in problems:
        ex.append([m("system", sp), um(q), am(think(t, a))])
    for q, a in [("2+2?","4."),("10*5?","50."),("9*9?","81."),("7+8?","15.")]:
        ex.append([m("system", sp), um(q), am(a)])
    random.shuffle(ex)
    return ex

# ── Phase 7: Logical Reasoning ───────────────────────────────────────────────
def gen_phase07():
    sp = "You are Yaya. Reason step by step in <|think|>...</|think|> tags."
    ex = []
    logic = [
        ("Up:down as left:?", "Opposites. Left->right.", "Right."),
        ("Puppy:dog as kitten:?", "Young->adult. Kitten->cat.", "Cat."),
        ("Page:book as brick:?", "Part->whole. Brick->wall.", "Wall."),
        ("Pen:write as knife:?", "Tool->action. Knife->cut.", "Cut."),
        ("All mammals breathe air. Whales=mammals. Breathe air?", "Mammals->air. Whales=mammals->yes.", "Yes."),
        ("No reptiles have fur. Snake=reptile. Fur?", "Reptiles->no fur. Snake->no.", "No."),
        ("Tom>Jack>Sam in height. Shortest?", "Sam.", "Sam."),
        ("Alice older than Bob, Carol younger. Oldest?", "Alice>Bob>Carol. Alice.", "Alice."),
        ("Next: A,C,E,G,...?", "Skip letters. G+2=I.", "I."),
        ("Next: 3,6,12,24,...?", "Doubles. 24x2=48.", "48."),
        ("Next: 1,4,9,16,25,...?", "Squares. 6^2=36.", "36."),
        ("Cities but no houses, forests no trees, water no fish. What?", "Names without contents -> a map.", "A map."),
        ("More you take, more you leave behind. What?", "Footsteps.", "Footsteps."),
        ("Sun rises west. Morning shadow direction?", "Away from sun -> east.", "East."),
        ("Rain->wet ground. Ground wet. Rain?", "Not necessarily. Could be sprinkler.", "Not necessarily."),
        ("Odd one out: apple, banana, carrot, mango?", "Carrot = vegetable.", "Carrot."),
        ("All dogs are animals. Rex is dog. Animal?", "Dogs subset animals. Rex=dog=animal.", "Yes."),
        ("Cat is not a dog. True?", "Different species. True.", "True."),
        ("I speak without mouth. Alive with wind. What?", "Echo.", "An echo."),
        ("If no gravity, ball thrown up?", "No force pulling back. Continues forever.", "It would keep going up forever."),
    ]
    for q, t, a in logic:
        ex.append([m("system", sp), um(q), am(think(t, a))])
    random.shuffle(ex)
    return ex

# ── Phase 8: Self-Reflection ─────────────────────────────────────────────────
def gen_phase08():
    sp = "You are Yaya. After answering, verify in <|verify|>...</|verify|> tags. Correct errors."
    ex = []
    pairs = [
        ("144/12?", "12.",
         "12x12=144. Correct.", "12."),
        ("Capital of Australia is Sydney. Correct?", "Checking...",
         "Sydney is largest city but NOT capital. Capital is Canberra.",
         "No. Capital is Canberra, not Sydney."),
        ("Verify: 2^8 = 128", "Let me check.",
         "2^7=128, 2^8=256. Wrong.", "Incorrect. 2^8=256. 2^7=128."),
        ("Is a whale a fish?", "No.",
         "Correct. Whales are mammals.", "No, whales are mammals."),
        ("Angles in a triangle?", "180 degrees.",
         "Fundamental geometry. Correct.", "180 degrees."),
        ("Antarctica a country?", "No.",
         "Correct. It's a continent.", "No, it's a continent."),
        ("Great Wall visible from space?", "Checking...",
         "Myth. Astronauts confirmed it's NOT visible with naked eye.",
         "False. It's not visible from space with the naked eye."),
        ("Tomatoes are vegetables?", "Well...",
         "Botanically fruits (from flowers, contain seeds). Culinarily used as vegetables.",
         "Botanically fruits, but commonly used as vegetables."),
        ("Lightning never strikes same place twice?", "Checking...",
         "False. Tall structures get struck repeatedly.",
         "False. Lightning often strikes the same place."),
        ("Humans use 10% of their brain?", "Let me verify...",
         "This is a myth. Brain imaging shows we use all parts of our brain.",
         "False. This is a myth. We use all parts of our brain."),
    ]
    for q, initial, critique, final in pairs:
        resp = f"{initial}\n<|verify|>\n{critique}\n<|/verify|>\n{final}"
        ex.append([m("system", sp), um(q), am(resp)])
    random.shuffle(ex)
    return ex

# ── Import phases 9-16 from extension module ─────────────────────────────────
from curriculum_phases_ext import gen_phase09, gen_phase10, gen_phase11, gen_phase12
from curriculum_phases_ext import gen_phase13, gen_phase14, gen_phase15, gen_phase16

PHASE_GENERATORS = {
    1:  {"name": "World Knowledge",        "slug": "world_knowledge",     "fn": gen_phase01},
    2:  {"name": "Conversational Fluency",  "slug": "conversational",      "fn": gen_phase02},
    3:  {"name": "Instruction Following",   "slug": "instruction_follow",  "fn": gen_phase03},
    4:  {"name": "Direct Q&A Mastery",      "slug": "direct_qa",           "fn": gen_phase04},
    5:  {"name": "Chain-of-Thought",        "slug": "chain_of_thought",    "fn": gen_phase05},
    6:  {"name": "Math Reasoning",          "slug": "math_reasoning",      "fn": gen_phase06},
    7:  {"name": "Logical Reasoning",       "slug": "logical_reasoning",   "fn": gen_phase07},
    8:  {"name": "Self-Reflection",         "slug": "self_reflection",     "fn": gen_phase08},
    9:  {"name": "Tool Calling Basics",     "slug": "tool_basics",         "fn": gen_phase09},
    10: {"name": "Multi-Step Tool Use",     "slug": "multi_tool",          "fn": gen_phase10},
    11: {"name": "RAG & Context Grounding", "slug": "rag_grounding",       "fn": gen_phase11},
    12: {"name": "Code Understanding",      "slug": "code",                "fn": gen_phase12},
    13: {"name": "Structured Output",       "slug": "structured_output",   "fn": gen_phase13},
    14: {"name": "Kenya & Swahili",         "slug": "kenya_swahili",       "fn": gen_phase14},
    15: {"name": "Safety & Refusals",       "slug": "safety",              "fn": gen_phase15},
    16: {"name": "DPO Alignment",           "slug": "dpo_pairs",           "fn": gen_phase16},
}
