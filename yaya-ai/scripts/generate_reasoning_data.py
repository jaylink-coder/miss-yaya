"""Generate reasoning + calculator-tool SFT data for Yaya.

Produces two kinds of examples:
1. Chain-of-Thought (CoT) reasoning with <|think|>...</|think|> blocks
2. Calculator tool-call examples using <|calc|>EXPR<|/calc|>=RESULT

Output: data/sft/yaya_reasoning.jsonl
"""

import json
import math
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

THINK_OPEN  = "<|think|>"
THINK_CLOSE = "<|/think|>"
CALC_OPEN   = "<|calc|>"
CALC_CLOSE  = "<|/calc|>"

SYSTEM_REASON = (
    "You are Yaya, a brilliant and friendly AI assistant. "
    "You think carefully before answering, showing your reasoning step by step. "
    "You have access to a calculator: write <|calc|>EXPRESSION<|/calc|> and the result will appear as =RESULT."
)

SYSTEM_MATH = (
    "You are Yaya, a helpful math assistant. "
    "Answer clearly and show your work. "
    "Use <|calc|>EXPRESSION<|/calc|> for any arithmetic."
)


def cot(thought: str, answer: str) -> str:
    return f"{THINK_OPEN}\n{thought.strip()}\n{THINK_CLOSE}\n\n{answer.strip()}"


def calc(expr: str) -> str:
    """Return a calc-call string with the real result injected."""
    from src.tools.calculator import _safe_eval
    try:
        result = _safe_eval(expr)
    except Exception:
        result = "?"
    return f"{CALC_OPEN}{expr}{CALC_CLOSE}={result}"


def ex(user: str, thought: str, answer: str, system: str = SYSTEM_REASON) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": cot(thought, answer)},
        ]
    }


def ex_calc(user: str, answer_with_calcs: str, system: str = SYSTEM_MATH) -> dict:
    """Example where the answer already contains <|calc|> calls with results."""
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": answer_with_calcs},
        ]
    }


# ─── CoT Reasoning Examples ───────────────────────────────────────────────────

COT_EXAMPLES = [

# ── Arithmetic ──
ex(
    "What is 47 × 83?",
    f"I need to multiply 47 by 83.\n"
    f"47 × 83 = 47 × 80 + 47 × 3\n"
    f"47 × 80 = {calc('47*80')}\n"
    f"47 × 3 = {calc('47*3')}\n"
    f"Total = {calc('3760+141')}",
    f"47 × 83 = **3901**",
),
ex(
    "What is the square root of 169?",
    "I need to find a number that, when multiplied by itself, gives 169.\n"
    "13 × 13 = 169. Let me verify: " + calc("13*13"),
    "The square root of 169 is **13**.",
),
ex(
    "Calculate 15 + 8 × 3.",
    "Order of operations (PEMDAS/BODMAS): multiplication before addition.\n"
    f"Step 1: 8 × 3 = {calc('8*3')}\n"
    f"Step 2: 15 + 24 = {calc('15+24')}",
    "15 + 8 × 3 = **39** (multiplication is done before addition).",
),
ex(
    "What is 2^10?",
    f"2^10 = 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2\n"
    f"= {calc('2**10')}",
    f"2^10 = **1024**.",
),

# ── Fractions & Percentages ──
ex(
    "What is 3/4 + 1/6?",
    "To add fractions, I need a common denominator.\n"
    "LCM(4, 6) = 12\n"
    "3/4 = 9/12\n"
    "1/6 = 2/12\n"
    f"9/12 + 2/12 = 11/12",
    "3/4 + 1/6 = **11/12**.",
),
ex(
    "What is 35% of 240?",
    f"35% of 240 = (35/100) × 240\n"
    f"= {calc('35/100 * 240')}",
    "35% of 240 = **84**.",
),
ex(
    "Convert 0.625 to a fraction.",
    "0.625 = 625/1000\n"
    "Simplify: GCD(625, 1000) = 125\n"
    "625/125 = 5, 1000/125 = 8\n"
    "So 0.625 = 5/8",
    "0.625 = **5/8**.",
),

# ── Algebra ──
ex(
    "Solve for x: 3x - 9 = 12.",
    "3x - 9 = 12\n"
    "3x = 12 + 9\n"
    f"3x = {calc('12+9')}\n"
    "x = 21 / 3\n"
    f"x = {calc('21/3')}",
    "x = **7**. Check: 3(7) - 9 = 21 - 9 = 12 ✓",
),
ex(
    "If x = 5, what is 4x + 7?",
    f"Substitute x = 5:\n"
    f"4(5) + 7 = {calc('4*5')} + 7 = {calc('4*5+7')}",
    "4x + 7 = **27** when x = 5.",
),
ex(
    "What are the solutions to x² - 5x + 6 = 0?",
    "Factor the quadratic:\n"
    "x² - 5x + 6 = (x - 2)(x - 3)\n"
    "Set each factor to zero:\n"
    "x - 2 = 0 → x = 2\n"
    "x - 3 = 0 → x = 3\n"
    "Verify: (2)² - 5(2) + 6 = 4 - 10 + 6 = 0 ✓\n"
    "        (3)² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓",
    "The solutions are **x = 2 and x = 3**.",
),

# ── Geometry ──
ex(
    "What is the area of a circle with radius 7? Use π ≈ 3.14159.",
    f"Area = π × r²\n"
    f"= 3.14159 × 7²\n"
    f"= 3.14159 × {calc('7**2')}\n"
    f"= {calc('3.14159 * 49')}",
    "The area ≈ **153.94 square units**.",
),
ex(
    "A right triangle has legs 5 and 12. What is the hypotenuse?",
    f"By the Pythagorean theorem: c² = a² + b²\n"
    f"c² = 5² + 12² = {calc('5**2')} + {calc('12**2')} = {calc('25+144')}\n"
    f"c = √169 = {calc('sqrt(169)')}",
    "The hypotenuse is **13**.",
),
ex(
    "Two angles are supplementary. One is 65°. What is the other?",
    "Supplementary angles sum to 180°.\n"
    f"Other angle = 180 - 65 = {calc('180-65')}°",
    "The other angle is **115°**.",
),

# ── Statistics ──
ex(
    "Find the mean of: 4, 7, 2, 9, 13.",
    f"Mean = sum / count\n"
    f"Sum = {calc('4+7+2+9+13')}\n"
    f"Count = 5\n"
    f"Mean = {calc('(4+7+2+9+13)/5')}",
    "The mean is **7**.",
),
ex(
    "What is the probability of rolling a 4 on a fair six-sided die?",
    "A fair die has 6 equally likely outcomes: 1, 2, 3, 4, 5, 6.\n"
    "Only one outcome (4) is favorable.\n"
    "Probability = 1/6 ≈ 0.1667",
    "The probability is **1/6** (approximately 16.67%).",
),

# ── Word Problems ──
ex(
    "A car travels at 90 km/h for 3 hours. How far does it travel?",
    f"Distance = speed × time\n"
    f"= 90 km/h × 3 h\n"
    f"= {calc('90*3')} km",
    "The car travels **270 km**.",
),
ex(
    "What is the simple interest on $1000 at 8% per year for 2 years?",
    "Simple Interest = Principal × Rate × Time\n"
    f"= 1000 × 0.08 × 2\n"
    f"= {calc('1000 * 0.08 * 2')}",
    "The simple interest is **$160**. Total amount owed = $1160.",
),
ex(
    "A train leaves at 60 km/h. Another leaves 1 hour later at 90 km/h. When does the second catch up?",
    "After 1 hour, the first train is 60 km ahead.\n"
    "Relative speed of second train = 90 - 60 = 30 km/h.\n"
    f"Time to close gap = 60 / 30 = {calc('60/30')} hours.\n"
    "So the second train catches up 2 hours after it departs.",
    "The second train catches up **2 hours** after departure.",
),
ex(
    "If you invest KSh 50,000 at 12% per year for 3 years (compound interest), what is the total?",
    "Compound Interest formula: A = P(1 + r)^t\n"
    f"A = 50000 × (1 + 0.12)^3\n"
    f"1.12^3 = {calc('1.12**3')}\n"
    f"A = 50000 × 1.404928 = {calc('50000 * 1.12**3')}",
    f"The total after 3 years is approximately **KSh {round(50000 * 1.12**3):,}**.",
),

# ── Logic & Reasoning ──
ex(
    "All mammals are warm-blooded. Dolphins are mammals. Are dolphins warm-blooded?",
    "This is a syllogism:\n"
    "Premise 1: All mammals → warm-blooded\n"
    "Premise 2: Dolphins → mammals\n"
    "Conclusion: Dolphins → warm-blooded (by transitivity)",
    "Yes. **Dolphins are warm-blooded** because all mammals are warm-blooded and dolphins are mammals.",
),
ex(
    "If it rains, the ground gets wet. The ground is not wet. Did it rain?",
    "P → Q (rain → wet ground)\n"
    "We know ¬Q (ground is NOT wet).\n"
    "By contrapositive: ¬Q → ¬P\n"
    "Therefore: ¬P — it did NOT rain.",
    "**No, it did not rain.** This is modus tollens: if rain implies wet ground, and the ground is not wet, then it did not rain.",
),
ex(
    "You have a 3-litre jug and a 5-litre jug. How do you measure exactly 4 litres?",
    "Steps:\n"
    "1. Fill the 5L jug.\n"
    "2. Pour from 5L into 3L until 3L is full → 5L has 2L left.\n"
    "3. Empty the 3L jug.\n"
    "4. Pour the 2L from 5L into the 3L jug.\n"
    "5. Fill the 5L jug again.\n"
    "6. Pour from 5L into 3L (which has 2L) until 3L is full → 3L needs 1L more, so 5L now has 4L.",
    "Fill the 5L, pour into 3L (5L has 2L), empty 3L, move 2L to 3L, refill 5L, pour 1L into 3L → **5L jug has exactly 4 litres**.",
),
ex(
    "What is the difference between correlation and causation?",
    "Correlation: two variables move together (when A increases, B tends to increase/decrease).\n"
    "Causation: A directly causes B.\n"
    "Example of correlation without causation: ice cream sales and drowning rates both rise in summer — but ice cream doesn't cause drowning (both are caused by hot weather).\n"
    "To establish causation, you need a controlled experiment (randomized controlled trial) or strong causal reasoning.",
    "**Correlation** means two variables are statistically related. **Causation** means one variable directly causes the other. Correlation does not imply causation — both could be caused by a third variable (confound). Example: shoe size correlates with reading ability in children (both caused by age, not each other).",
),

# ── Science ──
ex(
    "Why do objects fall at the same rate regardless of mass?",
    "Newton's 2nd law: F = ma → a = F/m\n"
    "Gravity on an object: F = mg\n"
    "Substituting: a = mg/m = g\n"
    "The mass cancels out → acceleration = g for all objects.\n"
    "Air resistance differs, but in a vacuum all objects fall identically.",
    "Because gravity pulls proportionally to mass but acceleration = Force/Mass. The mass cancels, leaving **a = g ≈ 9.8 m/s²** for all objects. Galileo proved this; Apollo 15 confirmed it on the Moon (feather and hammer fell together).",
),
ex(
    "What is the speed of light, and why can't anything travel faster?",
    "c ≈ 299,792,458 m/s (about 300,000 km/s).\n"
    "Special Relativity (Einstein, 1905): as an object accelerates, its relativistic mass increases.\n"
    "At v → c, the energy required → infinity.\n"
    "Only massless particles (photons) can travel at c.",
    "The speed of light is **≈ 3 × 10⁸ m/s**. Nothing with mass can reach it because the energy required becomes infinite as you approach c — a consequence of Einstein's Special Relativity.",
),

# ── Coding ──
ex(
    "Write a Python function that checks if a number is prime.",
    "A prime is only divisible by 1 and itself.\n"
    "Edge cases: ≤ 1 not prime; 2 is prime.\n"
    "Optimization: only check divisors up to √n.",
    "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n```\nRuns in O(√n) time.",
),
ex(
    "What is Big O notation?",
    "Big O describes how algorithm runtime (or memory) scales with input size n.\n"
    "Examples:\n"
    "O(1) — constant (dict lookup)\n"
    "O(log n) — logarithmic (binary search)\n"
    "O(n) — linear (scan a list)\n"
    "O(n²) — quadratic (nested loops)\n"
    "O(2^n) — exponential (brute-force subsets)\n"
    "We care about the dominant term as n → ∞.",
    "**Big O notation** describes the worst-case growth rate of an algorithm as input size grows:\n- O(1): constant — same time regardless of n\n- O(log n): logarithmic — binary search\n- O(n): linear — scanning an array\n- O(n²): quadratic — nested loops\n\nWe drop constants and lower-order terms: 3n² + 5n + 2 → **O(n²)**.",
),

# ── Kenyan Context ──
ex(
    "A matatu charges KSh 80 per person. If 12 passengers board, how much does the driver collect?",
    f"Total = price per person × number of passengers\n"
    f"= 80 × 12\n"
    f"= {calc('80*12')}",
    f"The driver collects **KSh {calc('80*12').split('=')[-1].strip()}**.",
),
ex(
    "A shopkeeper buys goods for KSh 4,500 and sells for KSh 6,000. What is the profit percentage?",
    f"Profit = selling price - cost price = {calc('6000-4500')}\n"
    f"Profit % = (profit / cost) × 100\n"
    f"= ({calc('6000-4500')} / 4500) × 100\n"
    f"= {calc('(6000-4500)/4500*100')}%",
    f"The profit percentage is approximately **{round((6000-4500)/4500*100, 1)}%**.",
),
]

# ─── Calculator Tool-Call Examples ───────────────────────────────────────────
# These teach the model to emit <|calc|>...<|/calc|> in its output

CALC_EXAMPLES = [
ex_calc(
    "What is 47 × 83?",
    f"I'll use the calculator:\n{calc('47*83')}\n\n47 × 83 = **3901**.",
),
ex_calc(
    "What is 17% of 4,500?",
    f"17% of 4,500 = 17/100 × 4,500\n{calc('17/100 * 4500')}\n\nThe answer is **765**.",
),
ex_calc(
    "Calculate 2^12.",
    f"{calc('2**12')}\n\n2¹² = **4096**.",
),
ex_calc(
    "What is the area of a rectangle 34 cm by 27 cm?",
    f"Area = length × width\n{calc('34*27')}\n\nThe area is **918 cm²**.",
),
ex_calc(
    "A school has 1,248 students. If they are split into classes of 32, how many classes are there?",
    f"Number of classes = 1248 ÷ 32\n{calc('1248/32')}\n\nThere are **39 classes**.",
),
ex_calc(
    "What is the perimeter of a square with side 15 m?",
    f"Perimeter = 4 × side\n{calc('4*15')}\n\nThe perimeter is **60 m**.",
),
ex_calc(
    "Convert 72°F to Celsius. Formula: C = (F - 32) × 5/9.",
    f"C = (72 - 32) × 5/9\n= {calc('72-32')} × 5/9\n{calc('(72-32)*5/9')}\n\n72°F = **22.2°C**.",
),
ex_calc(
    "A worker earns KSh 1,200 per day and works 22 days in a month. What is her monthly income?",
    f"Monthly income = daily rate × days worked\n{calc('1200*22')}\n\nShe earns **KSh 26,400** per month.",
),
ex_calc(
    "If a 2 kg bag of flour costs KSh 170, what is the price per kilogram?",
    f"Price per kg = total cost ÷ weight\n{calc('170/2')}\n\nThe price is **KSh 85 per kilogram**.",
),
ex_calc(
    "Solve: (3 + 5) × (10 - 4)",
    f"Following order of operations:\n{calc('(3+5)*(10-4)')}\n\n(3 + 5) × (10 - 4) = **48**.",
),
ex_calc(
    "What is the hypotenuse of a right triangle with legs 9 and 40?",
    f"By the Pythagorean theorem: c = √(a² + b²)\n"
    f"a² = {calc('9**2')}, b² = {calc('40**2')}\n"
    f"a² + b² = {calc('9**2 + 40**2')}\n"
    f"c = √{calc('9**2 + 40**2')} = {calc('sqrt(9**2 + 40**2)')}\n\n"
    "The hypotenuse is **41**.",
),
ex_calc(
    "A laptop costs KSh 65,000. There is a 15% discount. What is the final price?",
    f"Discount = 15% of 65,000\n{calc('0.15 * 65000')}\n"
    f"Final price = 65,000 - discount\n{calc('65000 - 0.15*65000')}\n\n"
    "The final price is **KSh 55,250**.",
),
]


def main():
    out_path = Path("data/sft/yaya_reasoning.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    all_examples = COT_EXAMPLES + CALC_EXAMPLES

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_examples)} reasoning examples -> {out_path}")
    print(f"  CoT examples:  {len(COT_EXAMPLES)}")
    print(f"  Calc examples: {len(CALC_EXAMPLES)}")


if __name__ == "__main__":
    main()
