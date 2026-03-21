"""Generate Chain-of-Thought (CoT) SFT training data for Yaya Super-Brain.

Produces examples where Yaya reasons step-by-step before answering.
Covers: math, logic, planning, coding, science, Kenyan context.

Output: data/sft/yaya_cot.jsonl (append to yaya_instruct.jsonl for training)
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

THINK_OPEN  = "<|think|>"
THINK_CLOSE = "<|/think|>"
SYSTEM = (
    "You are Yaya, a brilliant and friendly AI assistant. "
    "You think carefully before answering, showing your reasoning step by step."
)

def cot(thought: str, answer: str) -> str:
    return f"{THINK_OPEN}\n{thought.strip()}\n{THINK_CLOSE}\n\n{answer.strip()}"

def ex(user: str, thought: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": cot(thought, answer)},
        ]
    }

# ─────────────────────────────────────────────
EXAMPLES = [

# ── MATH ─────────────────────────────────────
ex(
    "A matatu charges KSh 50 per person. If 14 passengers board, how much does the driver collect?",
    "I need to multiply the fare per person by the number of passengers.\n"
    "50 × 14 = 700\nLet me verify: 50 × 10 = 500, 50 × 4 = 200. 500 + 200 = 700. ✓",
    "The driver collects **KSh 700**.",
),
ex(
    "What is 17% of 4,500?",
    "17% of 4,500 = (17/100) × 4,500\n= 0.17 × 4,500\n= 17 × 45\n= 765",
    "17% of 4,500 is **765**.",
),
ex(
    "A farmer has 3 hectares. He plants maize on 2/5 and beans on 1/3. How much is left unplanted?",
    "Maize: 2/5 × 3 = 6/5 = 1.2 hectares\nBeans: 1/3 × 3 = 1 hectare\n"
    "Total planted: 1.2 + 1.0 = 2.2 hectares\nUnplanted: 3 - 2.2 = 0.8 hectares",
    "**0.8 hectares** (800 m²) remain unplanted.",
),
ex(
    "If a loan of KSh 100,000 accrues 15% simple interest per year, what is owed after 2 years?",
    "Simple interest = Principal × Rate × Time\n= 100,000 × 0.15 × 2\n= 30,000\n"
    "Total owed = 100,000 + 30,000 = 130,000",
    "After 2 years, you owe **KSh 130,000** (principal + KSh 30,000 interest).",
),
ex(
    "Solve for x: 3x + 7 = 22",
    "3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 15 / 3\nx = 5\n"
    "Check: 3(5) + 7 = 15 + 7 = 22 ✓",
    "x = **5**",
),
ex(
    "A water tank holds 5,000 litres. A pump fills it at 120 litres/minute. How long to fill it?",
    "Time = Volume / Rate = 5,000 / 120\n= 41.67 minutes\n= 41 minutes and 40 seconds",
    "It takes approximately **41 minutes and 40 seconds** to fill the tank.",
),
ex(
    "Train A leaves Nairobi at 8 AM going 80 km/h. Train B leaves at 9 AM going 120 km/h on the same track. When does B catch A?",
    "At 9 AM, A has a 1-hour head start: 80 × 1 = 80 km ahead.\n"
    "B gains (120 - 80) = 40 km/h on A.\n"
    "Time to catch up: 80 / 40 = 2 hours after B departs.\n"
    "B departs 9 AM → catches A at 11 AM.\n"
    "Distance from Nairobi: 120 × 2 = 240 km.",
    "Train B catches Train A at **11:00 AM**, 240 km from Nairobi.",
),

# ── LOGIC ─────────────────────────────────────
ex(
    "All Kenyans are Africans. Wanjiku is Kenyan. Is Wanjiku African?",
    "This is a classic syllogism.\nPremise 1: All Kenyans → African\nPremise 2: Wanjiku → Kenyan\n"
    "Conclusion: Wanjiku → African (by transitivity)",
    "Yes. Since all Kenyans are Africans and Wanjiku is Kenyan, **Wanjiku is African**.",
),
ex(
    "You have 3 boxes: one has apples, one has oranges, one has both. All labels are wrong. "
    "You pick one fruit from the box labelled 'Both'. What is it?",
    "Since all labels are wrong, the 'Both' box cannot contain both fruits.\n"
    "It must contain either only apples OR only oranges.\n"
    "I draw one fruit. If it's an apple → box is 'Apples'.\n"
    "Now the remaining boxes labelled 'Apples' and 'Oranges' are also wrong.\n"
    "The 'Apples' box → must be 'Oranges'. The 'Oranges' box → must be 'Both'.\n"
    "One draw from 'Both' box is enough to label everything correctly.",
    "**One fruit is enough.** Draw from the 'Both' (mislabelled) box. "
    "If you get an apple, that box is 'Apples'. Then 'Apples' label → 'Both', 'Oranges' label → 'Oranges'. "
    "The same logic applies if you draw an orange.",
),
ex(
    "If it rains, the ground gets wet. The ground is wet. Did it rain?",
    "This tests whether we can reverse a conditional.\n"
    "P → Q (rain → wet ground). Q is true (ground is wet).\n"
    "Can we conclude P? No — this is the fallacy of affirming the consequent.\n"
    "The ground could be wet from other causes (sprinkler, spilled water, etc.).",
    "**Not necessarily.** The ground being wet doesn't prove it rained — other causes (sprinklers, spills) could explain it. This is the logical fallacy of affirming the consequent.",
),

# ── SCIENCE ───────────────────────────────────
ex(
    "Why is the sky blue?",
    "Light from the sun is white (all visible wavelengths).\n"
    "When it enters Earth's atmosphere, it collides with gas molecules.\n"
    "This causes Rayleigh scattering — shorter wavelengths (blue, ~450nm) scatter more than longer ones (red, ~700nm).\n"
    "Blue light scatters in all directions, so we see it coming from the whole sky.\n"
    "At sunset, light travels a longer path, scattering away all blue → only red/orange reaches us.",
    "The sky appears blue because of **Rayleigh scattering**: shorter blue wavelengths scatter much more than red ones as sunlight passes through the atmosphere, spreading blue light across the entire sky.",
),
ex(
    "Why does a heavier object not fall faster than a lighter one?",
    "Galileo showed this experimentally at Pisa.\n"
    "Newton's 2nd law: F = ma → a = F/m\n"
    "Gravity force on an object: F = mg (m = mass, g = gravitational acceleration)\n"
    "Substituting: a = mg/m = g (the mass cancels!)\n"
    "All objects fall at the same acceleration g ≈ 9.8 m/s² (ignoring air resistance).",
    "Because gravity pulls proportionally to mass, but acceleration = F/m. The mass cancels out, so **all objects fall at the same rate** (~9.8 m/s²) regardless of weight.",
),

# ── CODING ────────────────────────────────────
ex(
    "Write a Python function that checks if a number is prime.",
    "A prime number is only divisible by 1 and itself.\nEdge cases: numbers ≤ 1 are not prime; 2 is prime.\n"
    "Optimization: only check divisors up to √n (if n has a factor > √n, the other factor is < √n).\n"
    "Further optimization: after checking 2, only check odd numbers.",
    """```python
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
```
- Returns `False` for n < 2 (0, 1, negatives)
- Returns `True` for 2 (only even prime)
- Skips even numbers, checks odd divisors up to √n — O(√n) time""",
),
ex(
    "Given a list of numbers, return only those that appear more than once.",
    "I need to count occurrences, then filter.\nPython's Counter from collections does this cleanly.\n"
    "Alternative: use a set of seen items and a set of duplicates.",
    """```python
from collections import Counter

def duplicates(nums):
    counts = Counter(nums)
    return [n for n, c in counts.items() if c > 1]

# Example:
print(duplicates([1, 2, 3, 2, 4, 3, 5]))  # [2, 3]
```""",
),
ex(
    "How do I reverse a string in Python?",
    "Python strings support slicing. The slice `s[::-1]` steps backwards through the string.\n"
    "Alternatively: `''.join(reversed(s))` or a loop.\n"
    "The slice approach is most Pythonic and fastest.",
    """```python
s = "hello"
reversed_s = s[::-1]   # "olleh"
```
**Explanation:** `s[start:stop:step]` — with step=-1 and no start/stop, Python iterates from the end to the beginning.""",
),

# ── PLANNING ──────────────────────────────────
ex(
    "I want to start a small chicken farm in Kiambu. What are the key steps?",
    "This is a planning question. I'll think through the main phases:\n"
    "1. Research and planning (breeds, market, budget)\n"
    "2. Land and housing (space requirements, ventilation, security)\n"
    "3. Stock procurement (chicks vs. mature birds, local breeds vs. broilers)\n"
    "4. Feed and water systems\n"
    "5. Health management (vaccines, deworming, biosecurity)\n"
    "6. Marketing (local market, restaurants, eggs vs. meat)\n"
    "7. Record keeping and financial management",
    "**Key steps for a Kiambu chicken farm:**\n\n"
    "1. **Research & budget** — Decide on broilers (meat, 6-week cycle) or layers (eggs, 1.5-year cycle). Budget: KSh 50K–200K for 100 birds.\n"
    "2. **Housing** — At least 0.1 m² per bird. Good ventilation, predator-proof wire mesh, east-facing windows.\n"
    "3. **Buy chicks** — From KenchicKuehne or Kiplimo's. Day-old chicks cost ~KSh 120 each.\n"
    "4. **Feed** — Starter (0-4 weeks), Grower (4-8 weeks), Finisher. Cost ~KSh 70/kg for commercial feeds.\n"
    "5. **Vaccination** — Newcastle disease (day 7, 21), Gumboro (day 14). Vet visits every 2 weeks initially.\n"
    "6. **Marketing** — Register with local butchers, hotels, or sell at Kiambu market. Eggs fetch KSh 12-16 each retail.\n"
    "7. **Records** — Track feed costs, mortality, sales. Break-even typically at 2-3 cycles.",
),
ex(
    "How should I prepare for a job interview at a tech company?",
    "I'll think through all dimensions: technical, behavioural, company-specific, logistics.\n"
    "Technical: DSA coding problems, system design, language/framework specifics.\n"
    "Behavioural: STAR method (Situation, Task, Action, Result) for past experiences.\n"
    "Company: research their products, culture, recent news.\n"
    "Logistics: test setup, dress code, timing.",
    "**Tech Interview Preparation:**\n\n"
    "**Week before:**\n"
    "- Revise data structures (arrays, trees, graphs, hashmaps)\n"
    "- Practice 2-3 LeetCode problems daily (Easy + Medium)\n"
    "- Study system design basics (load balancing, databases, caching)\n\n"
    "**Day before:**\n"
    "- Research the company — products, recent funding, tech stack\n"
    "- Prepare 3 STAR stories: a challenge you solved, a failure you learned from, a team success\n"
    "- Test your setup (camera, microphone, IDE for coding interviews)\n\n"
    "**Day of:**\n"
    "- Think aloud — interviewers want to see your reasoning process\n"
    "- Ask clarifying questions before coding\n"
    "- If stuck, explain what you know and where you're blocked\n\n"
    "**After:** Send a thank-you email within 24 hours.",
),

# ── KENYAN CONTEXT ────────────────────────────
ex(
    "What are the main tax obligations for a Kenyan small business?",
    "I'll cover the key taxes by category for a small Kenyan business:\n"
    "1. Income Tax (Corporate Tax or PAYE if employees)\n"
    "2. VAT (if turnover > 5M)\n"
    "3. Presumptive Tax / Turnover Tax (small businesses)\n"
    "4. Withholding Tax (on certain payments)\n"
    "5. NSSF and NHIF (employee deductions)",
    "**Key tax obligations for Kenyan small businesses:**\n\n"
    "| Tax | Who pays | Rate |\n"
    "|-----|----------|------|\n"
    "| **Turnover Tax** | Businesses with turnover KSh 1M–25M | 1.5% of gross turnover |\n"
    "| **VAT** | Turnover > KSh 5M/year | 16% on taxable supplies |\n"
    "| **PAYE** | Employers with staff | Graduated 10%–35% |\n"
    "| **NSSF** | All employers | KSh 200 employer + KSh 200 employee |\n"
    "| **NHIF** | All employees | KSh 150–1,700 by salary bracket |\n\n"
    "**Filing:** iTax portal (itax.kra.go.ke). Monthly returns for VAT and PAYE; annual for income tax. Deadline: 20th of following month for VAT/PAYE.",
),
ex(
    "Explain how M-Pesa's Lipa na M-Pesa works for a small shop owner.",
    "Lipa na M-Pesa is Safaricom's merchant payment system. Let me think through how it works from the shop owner's perspective.\n"
    "There are two types: Buy Goods (merchant number, money goes directly to merchant) and Paybill (business number + account number).\n"
    "The shop owner registers, gets a till number, customers pay, funds settle daily to Safaricom wallet.\n"
    "Fees: merchant pays 0.5%–1% per transaction. Customer pays nothing.",
    "**How Lipa na M-Pesa works for your shop:**\n\n"
    "1. **Register** at any Safaricom shop with your business registration, ID, and KRA PIN → get a **Till Number** (Buy Goods) or **Paybill Number**.\n"
    "2. **Customer pays:** They dial *737# or use M-Pesa app → 'Lipa na M-Pesa' → 'Buy Goods' → enter your till number + amount → confirm with PIN.\n"
    "3. **You receive:** An instant SMS confirmation with amount and customer number. Funds in your M-Pesa merchant wallet.\n"
    "4. **Withdraw:** Transfer to your bank (free if same bank), to personal M-Pesa (small fee), or pay suppliers directly.\n\n"
    "**Fees:** Safaricom charges you ~0.5% per transaction. Customer pays nothing, which encourages adoption.\n\n"
    "**Tip:** Display your till number clearly. Many customers prefer Lipa na M-Pesa over cash.",
),

# ── CHESS ─────────────────────────────────────
ex(
    "In chess, what is the best opening move for White and why?",
    "The most popular first moves are 1.e4 and 1.d4.\n"
    "1.e4 (King's pawn): Controls center (d5, f5), opens diagonals for bishop and queen, enables quick kingside castling.\n"
    "1.d4 (Queen's pawn): Controls center (e5, c5), often leads to more positional games.\n"
    "Theory: Both are equal. 1.e4 is more tactical, 1.d4 more positional.\n"
    "Masters like Fischer preferred 1.e4. Carlsen uses both.",
    "**1.e4** is considered the most principled first move:\n"
    "- Immediately occupies the center\n"
    "- Opens the e-file and the diagonal for the queen/bishop\n"
    "- Enables kingside castling quickly\n"
    "- Leads to sharp, tactical games (Ruy Lopez, Sicilian, Italian)\n\n"
    "**1.d4** is equally valid and leads to more strategic battles (Queen's Gambit, King's Indian).\n\n"
    "Both are sound — choose based on your playing style: 1.e4 for tactics lovers, 1.d4 for positional players.",
),
ex(
    "What is the most important chess principle in the middlegame?",
    "Many principles exist. I'll think about which is most universal:\n"
    "- King safety (never expose your king)\n"
    "- Piece activity (active pieces beat passive ones)\n"
    "- Pawn structure (avoid isolated/doubled pawns)\n"
    "- Coordination (pieces working together)\n"
    "- Initiative (forcing opponent to react)\n"
    "The most universal is probably PIECE ACTIVITY — an active piece dominates a passive one regardless of material.",
    "**King safety first, then piece activity.**\n\n"
    "In the middlegame:\n"
    "1. **King safety** — never leave your king exposed. Castle early. A king attack wins quickly.\n"
    "2. **Piece activity** — put your pieces on their best squares. An active bishop is worth more than a knight stuck behind your own pawns.\n"
    "3. **Coordinate your pieces** — rooks belong on open files, knights near the center, bishops on long diagonals.\n\n"
    "A simple rule: *Ask 'Which is my worst piece?' and improve it.*",
),

# ── GENERAL REASONING ─────────────────────────
ex(
    "Should I save money or invest it?",
    "This depends on several factors I need to consider:\n"
    "1. Emergency fund — do they have 3-6 months of expenses saved?\n"
    "2. High-interest debt — paying this off is a guaranteed return\n"
    "3. Investment time horizon — longer = more risk capacity\n"
    "4. Risk tolerance\n"
    "5. Inflation — cash loses value over time\n"
    "The standard advice: emergency fund first, pay high-interest debt, then invest the rest.",
    "**Both — in this order:**\n\n"
    "1. **Emergency fund first** (3-6 months of expenses in a liquid account like MMF). Without this, any investment can be disrupted by emergencies.\n"
    "2. **Pay high-interest debt** (anything above ~10% p.a.). Paying off a 20% loan is a guaranteed 20% return.\n"
    "3. **Invest the rest** — in Kenya: T-Bills/T-Bonds (12-16% risk-free), MMFs (11-14%), NSSF/pension, NSE stocks, or REITs.\n\n"
    "**Why not just save?** Inflation (~6-8% in Kenya) silently erodes savings. KSh 100K today buys less in 5 years.\n\n"
    "**Simple rule:** Save what you need access to, invest what you don't need for 3+ years.",
),
ex(
    "What is the difference between AI, machine learning, and deep learning?",
    "These are nested subsets:\n"
    "AI (broadest) ⊃ Machine Learning ⊃ Deep Learning (narrowest)\n"
    "AI: any technique making machines act intelligently (rule-based systems count as AI)\n"
    "ML: AI that learns from data without being explicitly programmed\n"
    "Deep Learning: ML using multi-layer neural networks (inspired by the brain)",
    "They are **nested subsets:**\n\n"
    "```\n"
    "AI (any intelligent machine behavior)\n"
    " └── Machine Learning (learns from data)\n"
    "      └── Deep Learning (neural networks with many layers)\n"
    "```\n\n"
    "| | AI | ML | Deep Learning |\n"
    "|-|----|----|---------------|\n"
    "| Example | Chess rules engine | Spam filter | ChatGPT, image recognition |\n"
    "| Needs data? | No | Yes | Yes (lots) |\n"
    "| Learns patterns? | No | Yes | Yes (complex) |\n\n"
    "**Yaya** is deep learning — a neural network trained on text to predict and generate language.",
),
]

# ─────────────────────────────────────────────

def main():
    output_path = "data/sft/yaya_cot.jsonl"
    os.makedirs("data/sft", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(EXAMPLES)} CoT examples -> {output_path}")

    # Also append to yaya_instruct.jsonl
    instruct_path = "data/sft/yaya_instruct.jsonl"
    if os.path.exists(instruct_path):
        with open(instruct_path, "a", encoding="utf-8") as f:
            for ex in EXAMPLES:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Appended {len(EXAMPLES)} CoT examples -> {instruct_path}")

    return len(EXAMPLES)


if __name__ == "__main__":
    main()
