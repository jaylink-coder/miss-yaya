"""Generate training data for phases 9-11.

Phase 9 — Tool Calling:     calculator and datetime tool use
Phase 10 — Multi-Step Tools: ReAct (think→tool→observe→answer)
Phase 11 — RAG Grounding:    answer from provided context, cite evidence

Output files:
  data/sft/curriculum/phase09_tool_basics.jsonl
  data/sft/curriculum/phase10_multi_tool.jsonl
  data/sft/curriculum/phase11_rag_grounding.jsonl
"""

import json, os, random, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(REPO_ROOT, "data/sft/curriculum")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(99)

SYS = ("You are Yaya, a helpful AI assistant. When doing arithmetic, use "
       "<|calc|>EXPRESSION<|/calc|>=RESULT. Think step by step for multi-step problems.")

def msg(role, content):
    return {"role": role, "content": content}

def ex(q, a, system=SYS):
    return {"messages": [msg("system", system), msg("user", q), msg("assistant", a)]}


# ══════════════════════════════════════════════════════════════════════
# PHASE 9 — Tool Calling (calculator + datetime)
# ══════════════════════════════════════════════════════════════════════
PHASE9 = []

# ── Calculator tool use ────────────────────────────────────────────────
CALC = [
    ex("What is 347 times 28?",
       "I'll multiply these using the calculator.\n<|calc|>347 * 28<|/calc|>=9716\nSo 347 × 28 = **9716**."),

    ex("Calculate 15% of 4,800.",
       "<|calc|>4800 * 0.15<|/calc|>=720.0\n15% of 4,800 is **720**."),

    ex("What is the square root of 1,764?",
       "<|calc|>1764 ** 0.5<|/calc|>=42.0\nThe square root of 1,764 is **42**."),

    ex("Divide 8,550 by 45.",
       "<|calc|>8550 / 45<|/calc|>=190.0\n8,550 ÷ 45 = **190**."),

    ex("What is 2 to the power of 10?",
       "<|calc|>2 ** 10<|/calc|>=1024\n2¹⁰ = **1,024**."),

    ex("A shop sells items at KSh 850 each. If they sell 37 items, what is the total revenue?",
       "I'll calculate the total.\n<|calc|>850 * 37<|/calc|>=31450\nTotal revenue = **KSh 31,450**."),

    ex("What is 23% of 15,000?",
       "<|calc|>15000 * 0.23<|/calc|>=3450.0\n23% of 15,000 = **3,450**."),

    ex("Calculate (125 + 375) × 8.",
       "Step 1: add the numbers.\n<|calc|>125 + 375<|/calc|>=500\nStep 2: multiply.\n<|calc|>500 * 8<|/calc|>=4000\nAnswer: **4,000**."),

    ex("A farmer harvests 240 kg of maize per acre. He has 15 acres. How much maize in total?",
       "<|calc|>240 * 15<|/calc|>=3600\nTotal harvest = **3,600 kg**."),

    ex("What is 18% VAT on a price of KSh 12,500?",
       "<|calc|>12500 * 0.18<|/calc|>=2250.0\n18% VAT on KSh 12,500 = **KSh 2,250**. The total price with VAT = "
       "<|calc|>12500 + 2250<|/calc|>=14750.0 = **KSh 14,750**."),

    ex("What is 1,000 divided by 8?",
       "<|calc|>1000 / 8<|/calc|>=125.0\n1,000 ÷ 8 = **125**."),

    ex("If petrol costs KSh 215 per litre and a car uses 45 litres, what is the total cost?",
       "<|calc|>215 * 45<|/calc|>=9675\nTotal fuel cost = **KSh 9,675**."),

    ex("What is 3/4 of 960?",
       "<|calc|>960 * 3 / 4<|/calc|>=720.0\n3/4 of 960 = **720**."),

    ex("A rectangle is 24 m long and 18 m wide. What is its area?",
       "<|calc|>24 * 18<|/calc|>=432\nArea = **432 m²**."),

    ex("What is the average of 85, 92, 76, 88, and 79?",
       "First, add all values.\n<|calc|>85 + 92 + 76 + 88 + 79<|/calc|>=420\nThen divide by 5.\n<|calc|>420 / 5<|/calc|>=84.0\nAverage = **84**."),
]

# ── When NOT to use the calculator ────────────────────────────────────
NO_CALC = [
    ex("What is 2 + 2?",  "4."),
    ex("What is 10 × 10?", "100."),
    ex("What is the capital of Kenya?", "Nairobi."),
    ex("Who are you?", "I'm Yaya, your AI assistant. How can I help?"),
]

PHASE9.extend(CALC * 4)   # 15 × 4 = 60
PHASE9.extend(NO_CALC * 4) # 4 × 4 = 16
random.shuffle(PHASE9)


# ══════════════════════════════════════════════════════════════════════
# PHASE 10 — Multi-Step ReAct
# Format: Thought → Action → Observation → Answer
# ══════════════════════════════════════════════════════════════════════
REACT_SYS = ("You are Yaya, a reasoning AI. For complex problems use this pattern:\n"
             "Thought: what do I need to figure out?\n"
             "Action: <|calc|>EXPRESSION<|/calc|>=RESULT\n"
             "Observation: what does the result tell me?\n"
             "Answer: final answer\n")

def react(q, a):
    return {"messages": [msg("system", REACT_SYS), msg("user", q), msg("assistant", a)]}

PHASE10 = []

REACT_EXAMPLES = [
    react(
        "A trader buys 200 kg of sugar at KSh 80/kg and sells all of it at KSh 110/kg. "
        "What is the profit percentage?",
        "Thought: I need to find cost price, selling price, profit, and profit percentage.\n"
        "Action: <|calc|>200 * 80<|/calc|>=16000\n"
        "Observation: Cost price = KSh 16,000.\n"
        "Action: <|calc|>200 * 110<|/calc|>=22000\n"
        "Observation: Selling price = KSh 22,000.\n"
        "Action: <|calc|>22000 - 16000<|/calc|>=6000\n"
        "Observation: Profit = KSh 6,000.\n"
        "Action: <|calc|>6000 / 16000 * 100<|/calc|>=37.5\n"
        "Answer: The profit percentage is **37.5%**."
    ),

    react(
        "Three workers can complete a job in 12 days. How long will it take 4 workers at the same rate?",
        "Thought: More workers → fewer days. Total work = 3 workers × 12 days.\n"
        "Action: <|calc|>3 * 12<|/calc|>=36\n"
        "Observation: Total work = 36 worker-days.\n"
        "Thought: With 4 workers, days = total work ÷ 4.\n"
        "Action: <|calc|>36 / 4<|/calc|>=9.0\n"
        "Answer: 4 workers will complete the job in **9 days**."
    ),

    react(
        "A car travels from Nairobi to Mombasa (480 km) at 90 km/h, then returns at 60 km/h. "
        "What is the average speed for the entire journey?",
        "Thought: Average speed = total distance ÷ total time (NOT simple average of speeds).\n"
        "Action: <|calc|>480 / 90<|/calc|>=5.333333333333333\n"
        "Observation: Time Nairobi→Mombasa ≈ 5.33 hours.\n"
        "Action: <|calc|>480 / 60<|/calc|>=8.0\n"
        "Observation: Time Mombasa→Nairobi = 8 hours.\n"
        "Action: <|calc|>480 + 480<|/calc|>=960\n"
        "Observation: Total distance = 960 km.\n"
        "Action: <|calc|>5.333333333333333 + 8<|/calc|>=13.333333333333332\n"
        "Action: <|calc|>960 / 13.333333333333332<|/calc|>=72.00000000000001\n"
        "Answer: The average speed for the whole journey is **72 km/h**."
    ),

    react(
        "A school has 840 students. 55% are boys. How many girls are there?",
        "Thought: Girls = 100% - 55% = 45% of total.\n"
        "Action: <|calc|>840 * 0.45<|/calc|>=378.0\n"
        "Answer: There are **378 girls**."
    ),

    react(
        "You invest KSh 50,000 at 8% per year simple interest for 3 years. What is the total amount at the end?",
        "Thought: Simple interest = Principal × Rate × Time. Total = Principal + Interest.\n"
        "Action: <|calc|>50000 * 0.08 * 3<|/calc|>=12000.0\n"
        "Observation: Interest = KSh 12,000.\n"
        "Action: <|calc|>50000 + 12000<|/calc|>=62000.0\n"
        "Answer: Total amount after 3 years = **KSh 62,000**."
    ),

    react(
        "A shop marks up goods by 40% then gives a 10% discount. What is the net profit percentage on cost?",
        "Thought: Assume cost = 100. After 40% markup, price = 140. After 10% discount on 140...\n"
        "Action: <|calc|>140 * 0.10<|/calc|>=14.0\n"
        "Observation: Discount = 14. Selling price = 140 - 14.\n"
        "Action: <|calc|>140 - 14<|/calc|>=126\n"
        "Observation: Selling price = 126 on a cost of 100.\n"
        "Action: <|calc|>126 - 100<|/calc|>=26\n"
        "Answer: Net profit = **26%** on cost."
    ),

    react(
        "If you save KSh 2,500 per month, how many months to save KSh 75,000?",
        "Thought: months = target ÷ monthly saving.\n"
        "Action: <|calc|>75000 / 2500<|/calc|>=30.0\n"
        "Answer: It will take **30 months** (2.5 years)."
    ),

    react(
        "A rectangle has perimeter 56 m. Its length is 3 times its width. Find the area.",
        "Thought: Let width = w, length = 3w. Perimeter = 2(l+w) = 2(3w+w) = 8w = 56.\n"
        "Action: <|calc|>56 / 8<|/calc|>=7.0\n"
        "Observation: Width = 7 m, Length = 3 × 7.\n"
        "Action: <|calc|>3 * 7<|/calc|>=21\n"
        "Observation: Length = 21 m.\n"
        "Action: <|calc|>21 * 7<|/calc|>=147\n"
        "Answer: Area = **147 m²**."
    ),

    react(
        "Mwenda earns KSh 45,000/month. He pays 16% tax on income above KSh 24,000. "
        "What is his net monthly pay?",
        "Thought: Tax is only on income above 24,000.\n"
        "Action: <|calc|>45000 - 24000<|/calc|>=21000\n"
        "Observation: Taxable income = KSh 21,000.\n"
        "Action: <|calc|>21000 * 0.16<|/calc|>=3360.0\n"
        "Observation: Tax = KSh 3,360.\n"
        "Action: <|calc|>45000 - 3360<|/calc|>=41640.0\n"
        "Answer: Mwenda's net monthly pay = **KSh 41,640**."
    ),

    react(
        "A water tank is 2 m long, 1.5 m wide, and 1.2 m tall. How many litres does it hold? "
        "(1 m³ = 1,000 litres)",
        "Thought: Volume = length × width × height, then convert to litres.\n"
        "Action: <|calc|>2 * 1.5 * 1.2<|/calc|>=3.5999999999999996\n"
        "Observation: Volume ≈ 3.6 m³.\n"
        "Action: <|calc|>3.6 * 1000<|/calc|>=3600.0\n"
        "Answer: The tank holds **3,600 litres**."
    ),
]

PHASE10.extend(REACT_EXAMPLES * 5)  # 10 × 5 = 50
random.shuffle(PHASE10)


# ══════════════════════════════════════════════════════════════════════
# PHASE 11 — RAG Grounding (answer from context, cite it)
# ══════════════════════════════════════════════════════════════════════
RAG_SYS = ("You are Yaya. Answer questions using ONLY the provided context. "
           "If the context does not contain the answer, say so honestly. "
           "Cite the relevant part of the context in your answer.")

def rag(context, question, answer):
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    return {"messages": [msg("system", RAG_SYS), msg("user", user_msg), msg("assistant", answer)]}

PHASE11 = []

RAG_EXAMPLES = [
    rag(
        "Kenya gained independence on 12 December 1963. Jomo Kenyatta became the first Prime Minister, "
        "and later the first President when Kenya became a republic in 1964.",
        "When did Kenya become a republic?",
        "According to the context, Kenya became a republic in **1964**. Jomo Kenyatta was the first President."
    ),

    rag(
        "M-Pesa was launched by Safaricom in Kenya in March 2007. It allows users to deposit, "
        "withdraw, and transfer money using a mobile phone. By 2023 it had over 51 million active users "
        "across multiple African countries.",
        "How many active M-Pesa users were there by 2023?",
        "The context states that M-Pesa had **over 51 million active users** by 2023 across multiple African countries."
    ),

    rag(
        "Lake Victoria is the largest lake in Africa and the second-largest freshwater lake in the world. "
        "It borders Uganda, Tanzania, and Kenya. The lake covers an area of approximately 68,800 km².",
        "Which countries border Lake Victoria?",
        "According to the context, Lake Victoria borders **Uganda, Tanzania, and Kenya**."
    ),

    rag(
        "The Nairobi Stock Exchange (NSE) was established in 1954. It lists over 65 companies. "
        "The NSE 20 Share Index tracks the performance of the top 20 companies by market capitalisation.",
        "What does the NSE 20 Share Index track?",
        "The context explains that the NSE 20 Share Index **tracks the performance of the top 20 companies "
        "by market capitalisation** on the Nairobi Stock Exchange."
    ),

    rag(
        "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide "
        "to produce glucose and oxygen. The equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂.",
        "What are the inputs and outputs of photosynthesis?",
        "According to the context:\n"
        "- **Inputs**: sunlight, water (H₂O), and carbon dioxide (CO₂)\n"
        "- **Outputs**: glucose (C₆H₁₂O₆) and oxygen (O₂)"
    ),

    rag(
        "The Constitution of Kenya 2010 established a bicameral Parliament consisting of the National Assembly "
        "(350 members) and the Senate (67 senators). It also created 47 counties, each with an elected governor.",
        "How many senators are in the Kenyan Senate?",
        "The context states that the Senate has **67 senators**, as established by the Constitution of Kenya 2010."
    ),

    rag(
        "Malaria is caused by Plasmodium parasites transmitted through the bites of infected female "
        "Anopheles mosquitoes. Symptoms include fever, chills, and headache. Treatment with artemisinin-based "
        "combination therapy (ACT) is recommended by WHO.",
        "What causes malaria and how is it treated?",
        "According to the context:\n"
        "- **Cause**: Malaria is caused by Plasmodium parasites, spread through bites from infected female "
        "Anopheles mosquitoes.\n"
        "- **Treatment**: WHO recommends **artemisinin-based combination therapy (ACT)**."
    ),

    rag(
        "The Silicon Savannah refers to Nairobi's technology hub, particularly the area around Westlands "
        "and the iHub innovation centre. Kenya has become the leading tech hub in Sub-Saharan Africa, "
        "attracting investments from Google, Microsoft, and various venture capital firms.",
        "Why is Nairobi called the Silicon Savannah?",
        "Based on the context, Nairobi is called the Silicon Savannah because it hosts a technology hub "
        "(particularly around Westlands and the iHub innovation centre) and has become the **leading tech hub "
        "in Sub-Saharan Africa**, attracting major companies like Google and Microsoft."
    ),

    rag(
        "Wangari Maathai was a Kenyan environmentalist and political activist. She founded the Green Belt "
        "Movement in 1977, which has planted over 51 million trees across Africa. In 2004 she became the "
        "first African woman to receive the Nobel Peace Prize.",
        "What award did Wangari Maathai receive in 2004?",
        "The context states that in 2004, Wangari Maathai became **the first African woman to receive the "
        "Nobel Peace Prize**."
    ),

    rag(
        "The Swahili coast refers to the coastal region of East Africa stretching from Somalia to Mozambique. "
        "It was a major centre of Indian Ocean trade from the 8th to 16th centuries. Key trading cities "
        "included Mombasa, Malindi, Zanzibar, and Kilwa.",
        "Name two key trading cities on the Swahili coast.",
        "According to the context, key trading cities on the Swahili coast included **Mombasa, Malindi, "
        "Zanzibar, and Kilwa**. Any two of these would be correct answers."
    ),

    # Honest "not in context" examples
    rag(
        "The Kenya Revenue Authority (KRA) was established in 1995. It is responsible for the assessment, "
        "collection, and accounting of all revenues owed to the government.",
        "What is the current Commissioner General of KRA?",
        "The context does not mention the current Commissioner General of KRA. It only states that KRA was "
        "established in 1995 and its general purpose. I cannot answer this from the given context."
    ),

    rag(
        "Nairobi National Park covers 117 km² and is located 7 km from the Nairobi city centre. "
        "It is home to lions, leopards, rhinos, giraffes, and buffalo.",
        "How many tourists visited Nairobi National Park last year?",
        "The context does not contain visitor statistics for Nairobi National Park. "
        "It only mentions the park's location and the animals found there."
    ),
]

PHASE11.extend(RAG_EXAMPLES * 4)  # 12 × 4 = 48
random.shuffle(PHASE11)


# ══════════════════════════════════════════════════════════════════════
# Write output files
# ══════════════════════════════════════════════════════════════════════
files = {
    "phase09_tool_basics.jsonl":  PHASE9,
    "phase10_multi_tool.jsonl":   PHASE10,
    "phase11_rag_grounding.jsonl": PHASE11,
}

for fname, data in files.items():
    out_path = os.path.join(OUT_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {fname}: {len(data)} examples")

print(f"\nTotal: {sum(len(d) for d in files.values())} examples written to {OUT_DIR}")
