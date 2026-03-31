"""
generate_math_data.py  — v2 (Computational Edition)
=====================================================
Redesigned to teach Yaya to COMPUTE, not just FORMAT.

Core principles:
  1. EXHAUSTIVE small-number facts — model memorizes lookup tables
  2. RIGID algorithmic CoT — every problem of the same type uses the IDENTICAL template
  3. GRADUAL difficulty — never skip steps; 1-digit before 2-digit before 3-digit
  4. CONSISTENT final answer format — always "Answer: X" at the end

Stage 1: Arithmetic Facts & Algorithms
Stage 2: Fractions & Decimals (algorithmic)
Stage 3: Pre-Algebra (equation solving, step-by-step)
Stage 4: Algebra (linear, quadratic, systems — with explicit steps)
Stage 5: Geometry (formulas → substitution → calculation)
Stage 6: Statistics & Probability (algorithmic)
Stage 7: Word Problems (identify → set up equation → solve)
Stage 8: Calculus (rules → apply → simplify)

Usage:
    python scripts/generate_math_data.py
    python scripts/generate_math_data.py --stage 1
    python scripts/generate_math_data.py --stage 1 --preview 3
"""

import sys, json, random, argparse, math
from pathlib import Path
from fractions import Fraction
from itertools import product

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

SYSTEM_MSG = (
    "You are Yaya, a helpful, honest, and friendly AI assistant. "
    "You answer questions clearly and thoughtfully."
)
OUTPUT_DIR = Path("data/sft/math")


def sample(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def shuffled(lst):
    random.shuffle(lst)
    return lst


# ══════════════════════════════════════════════════════════════
# STAGE 1 — Arithmetic: exhaustive facts + algorithms
# ══════════════════════════════════════════════════════════════

def gen_stage1(target: int = 800) -> list:
    out = []

    # ── 1a. ALL single-digit addition facts (0-9) + (0-9) = 100 facts × 2 formats
    for a in range(0, 10):
        for b in range(0, 10):
            r = a + b
            out.append(sample(f"What is {a} + {b}?", f"{a} + {b} = {r}"))
            if a != b:
                out.append(sample(f"What is {b} + {a}?", f"{b} + {a} = {r}"))

    # ── 1b. ALL single-digit multiplication facts (1-9) × (1-9) = 81 facts
    for a in range(1, 10):
        for b in range(1, 10):
            r = a * b
            out.append(sample(f"What is {a} × {b}?", f"{a} × {b} = {r}"))

    # ── 1c. Two-digit + one-digit (no carry) — rigid algorithm
    for _ in range(60):
        tens = random.randint(1, 9)
        a_ones = random.randint(0, 4)
        b = random.randint(0, 9 - a_ones)   # no carry
        a = tens * 10 + a_ones
        r = a + b
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a_ones} + {b} = {a_ones + b}\n"
            f"Tens: {tens}\n"
            f"Answer: {r}"
        ))

    # ── 1d. Two-digit + one-digit (WITH carry) — rigid algorithm
    for _ in range(60):
        tens = random.randint(1, 8)
        a_ones = random.randint(5, 9)
        b = random.randint(10 - a_ones, 9)  # force carry
        a = tens * 10 + a_ones
        r = a + b
        ones_sum = a_ones + b
        new_ones = ones_sum % 10
        carry = ones_sum // 10
        new_tens = tens + carry
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a_ones} + {b} = {ones_sum} → write {new_ones}, carry {carry}\n"
            f"Tens: {tens} + {carry}(carry) = {new_tens}\n"
            f"Answer: {r}"
        ))

    # ── 1e. Two-digit + two-digit (no carry) — rigid algorithm
    for _ in range(80):
        a1, a2 = random.randint(1, 9), random.randint(0, 4)
        b1, b2 = random.randint(1, 9), random.randint(0, 9 - a2)
        a, b = a1 * 10 + a2, b1 * 10 + b2
        r = a + b
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a2} + {b2} = {a2 + b2}\n"
            f"Tens: {a1} + {b1} = {a1 + b1}\n"
            f"Answer: {r}"
        ))

    # ── 1f. Two-digit + two-digit (WITH carry) — rigid algorithm
    for _ in range(80):
        a1, a2 = random.randint(1, 8), random.randint(5, 9)
        b1, b2 = random.randint(1, 8), random.randint(10 - a2, 9)
        a, b = a1 * 10 + a2, b1 * 10 + b2
        r = a + b
        ones_sum = a2 + b2
        new_ones = ones_sum % 10
        carry = 1
        tens_sum = a1 + b1 + carry
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a2} + {b2} = {ones_sum} → write {new_ones}, carry 1\n"
            f"Tens: {a1} + {b1} + 1(carry) = {tens_sum}\n"
            f"Answer: {r}"
        ))

    # ── 1g. Subtraction (two-digit, no borrow)
    for _ in range(80):
        r = random.randint(10, 89)
        b = random.randint(1, 9)
        a = r + b  # a - b = r
        if a > 99:
            continue
        a1, a2 = divmod(a, 10)
        b2 = b
        out.append(sample(
            f"What is {a} - {b}?",
            f"Ones: {a2} - {b2} = {a2 - b2}\n"
            f"Tens: {a1}\n"
            f"Answer: {r}"
        ))

    # ── 1h. Multiplication: two-digit × one-digit — rigid algorithm
    for _ in range(100):
        a1 = random.randint(1, 9)
        a2 = random.randint(0, 9)
        b = random.randint(2, 9)
        a = a1 * 10 + a2
        r = a * b
        ones_prod = a2 * b
        ones_digit = ones_prod % 10
        carry = ones_prod // 10
        tens_prod = a1 * b + carry
        out.append(sample(
            f"What is {a} × {b}?",
            f"Ones: {a2} × {b} = {ones_prod}"
            + (f" → write {ones_digit}, carry {carry}" if carry else "") + "\n"
            f"Tens: {a1} × {b}" + (f" + {carry}(carry)" if carry else "") + f" = {tens_prod}\n"
            f"Answer: {r}"
        ))

    # ── 1i. Division (exact, single-digit divisor)
    for _ in range(80):
        b = random.randint(2, 9)
        q = random.randint(2, 20)
        a = b * q
        out.append(sample(
            f"What is {a} ÷ {b}?",
            f"{b} × {q} = {a}, so {a} ÷ {b} = {q}\n"
            f"Answer: {q}"
        ))

    # ── 1j. Squares and square roots (1-15)
    for n in range(1, 16):
        sq = n * n
        out.append(sample(f"What is {n} squared?", f"{n} × {n} = {sq}\nAnswer: {sq}"))
        out.append(sample(f"What is {n}²?", f"{n} × {n} = {sq}\nAnswer: {sq}"))
        out.append(sample(
            f"What is the square root of {sq}?",
            f"{n} × {n} = {sq}, so √{sq} = {n}\nAnswer: {n}"
        ))

    # ── 1k. Order of operations — rigid two-step
    for _ in range(60):
        a = random.randint(1, 20)
        b = random.randint(1, 9)
        c = random.randint(1, 9)
        r = a + b * c
        out.append(sample(
            f"What is {a} + {b} × {c}?",
            f"Step 1 (multiplication first): {b} × {c} = {b * c}\n"
            f"Step 2 (addition): {a} + {b * c} = {r}\n"
            f"Answer: {r}"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 2 — Fractions & Decimals (algorithmic)
# ══════════════════════════════════════════════════════════════

def gen_stage2(target: int = 600) -> list:
    out = []

    # ── 2a. Fraction simplification — explicit GCD steps
    pairs = [(n, d) for d in range(2, 13) for n in range(1, d)
             if math.gcd(n, d) > 1]
    for n, d in random.choices(pairs, k=80):
        g = math.gcd(n, d)
        sn, sd = n // g, d // g
        out.append(sample(
            f"Simplify {n}/{d}.",
            f"Step 1: Find GCD({n}, {d}) = {g}\n"
            f"Step 2: Divide top and bottom by {g}:\n"
            f"  {n} ÷ {g} = {sn}\n"
            f"  {d} ÷ {g} = {sd}\n"
            f"Answer: {sn}/{sd}"
        ))

    # ── 2b. Fraction addition — LCD method
    for _ in range(100):
        d1 = random.choice([2, 3, 4, 5, 6, 8, 10])
        d2 = random.choice([2, 3, 4, 5, 6, 8, 10])
        n1 = random.randint(1, d1 - 1)
        n2 = random.randint(1, d2 - 1)
        f1, f2 = Fraction(n1, d1), Fraction(n2, d2)
        result = f1 + f2
        lcd = result.denominator * (result.numerator // result.numerator) if result.numerator else d1 * d2 // math.gcd(d1, d2)
        lcd = d1 * d2 // math.gcd(d1, d2)
        m1 = lcd // d1
        m2 = lcd // d2
        rn = n1 * m1 + n2 * m2
        g = math.gcd(rn, lcd)
        out.append(sample(
            f"What is {n1}/{d1} + {n2}/{d2}?",
            f"Step 1: LCD({d1}, {d2}) = {lcd}\n"
            f"Step 2: Convert fractions:\n"
            f"  {n1}/{d1} = {n1*m1}/{lcd}\n"
            f"  {n2}/{d2} = {n2*m2}/{lcd}\n"
            f"Step 3: Add numerators: {n1*m1} + {n2*m2} = {rn}\n"
            f"Step 4: Simplify {rn}/{lcd}: GCD = {g} → {rn//g}/{lcd//g}\n"
            f"Answer: {result.numerator}/{result.denominator}"
        ))

    # ── 2c. Fraction subtraction
    for _ in range(60):
        d1 = random.choice([2, 3, 4, 5, 6, 8])
        d2 = random.choice([2, 3, 4, 5, 6, 8])
        n1 = random.randint(1, d1)
        n2 = random.randint(1, d2)
        f1, f2 = Fraction(n1, d1), Fraction(n2, d2)
        if f1 <= f2:
            f1, f2 = f2, f1
            n1, d1, n2, d2 = f1.numerator, f1.denominator, f2.numerator, f2.denominator
        result = f1 - f2
        lcd = d1 * d2 // math.gcd(d1, d2)
        m1, m2 = lcd // d1, lcd // d2
        diff_n = n1 * m1 - n2 * m2
        g = math.gcd(abs(diff_n), lcd)
        out.append(sample(
            f"What is {f1.numerator}/{f1.denominator} - {f2.numerator}/{f2.denominator}?",
            f"Step 1: LCD({f1.denominator}, {f2.denominator}) = {lcd}\n"
            f"Step 2: Convert:\n"
            f"  {f1.numerator}/{f1.denominator} = {f1.numerator*m1}/{lcd}\n"
            f"  {f2.numerator}/{f2.denominator} = {f2.numerator*m2}/{lcd}\n"
            f"Step 3: Subtract: {f1.numerator*m1} - {f2.numerator*m2} = {diff_n}\n"
            f"Answer: {result.numerator}/{result.denominator}"
        ))

    # ── 2d. Fraction × fraction
    for _ in range(60):
        n1, d1 = random.randint(1, 5), random.randint(2, 8)
        n2, d2 = random.randint(1, 5), random.randint(2, 8)
        result = Fraction(n1, d1) * Fraction(n2, d2)
        g = math.gcd(n1 * n2, d1 * d2)
        out.append(sample(
            f"What is {n1}/{d1} × {n2}/{d2}?",
            f"Step 1: Multiply numerators: {n1} × {n2} = {n1*n2}\n"
            f"Step 2: Multiply denominators: {d1} × {d2} = {d1*d2}\n"
            f"Step 3: Simplify {n1*n2}/{d1*d2}: GCD = {g} → {result.numerator}/{result.denominator}\n"
            f"Answer: {result.numerator}/{result.denominator}"
        ))

    # ── 2e. Decimal addition with aligned columns
    for _ in range(80):
        a = round(random.uniform(1.1, 49.9), 1)
        b = round(random.uniform(1.1, 49.9), 1)
        r = round(a + b, 1)
        a1, a2 = str(a).split(".")
        b1, b2 = str(b).split(".")
        ones_sum = int(a2) + int(b2)
        carry = ones_sum // 10
        dec_digit = ones_sum % 10
        int_sum = int(a1) + int(b1) + carry
        out.append(sample(
            f"What is {a} + {b}?",
            f"Tenths: {a2} + {b2} = {ones_sum}"
            + (f" → write {dec_digit}, carry {carry}" if carry else "") + "\n"
            f"Ones: {a1} + {b1}" + (f" + {carry}" if carry else "") + f" = {int_sum}\n"
            f"Answer: {r}"
        ))

    # ── 2f. Percentage calculations — explicit formula
    for _ in range(80):
        pct = random.choice([10, 20, 25, 50, 75, 5, 15, 30, 40, 60])
        whole = random.choice([20, 40, 50, 80, 100, 120, 150, 200, 250, 300, 400, 500])
        result = pct * whole // 100
        out.append(sample(
            f"What is {pct}% of {whole}?",
            f"Step 1: Convert % to decimal: {pct}% = {pct}/100 = {pct/100}\n"
            f"Step 2: Multiply: {pct/100} × {whole} = {result}\n"
            f"Answer: {result}"
        ))

    # ── 2g. Convert fraction ↔ decimal ↔ percent — exhaustive common set
    conversions = [
        (1, 2, "0.5", "50%"), (1, 4, "0.25", "25%"), (3, 4, "0.75", "75%"),
        (1, 5, "0.2", "20%"), (2, 5, "0.4", "40%"), (3, 5, "0.6", "60%"),
        (4, 5, "0.8", "80%"), (1, 8, "0.125", "12.5%"), (3, 8, "0.375", "37.5%"),
        (1, 10, "0.1", "10%"), (1, 3, "0.333...", "33.3%"), (2, 3, "0.667...", "66.7%"),
        (1, 6, "0.167...", "16.7%"), (1, 100, "0.01", "1%"), (1, 20, "0.05", "5%"),
    ]
    for n, d, dec_str, pct_str in conversions:
        out.append(sample(
            f"Convert {n}/{d} to a decimal and a percentage.",
            f"Decimal: {n} ÷ {d} = {dec_str}\n"
            f"Percentage: {dec_str} × 100 = {pct_str}\n"
            f"Answer: {dec_str} = {pct_str}"
        ))
        out.append(sample(
            f"What is {dec_str} as a fraction?",
            f"{dec_str} = {n}/{d}\n"
            f"Answer: {n}/{d}"
        ))
        out.append(sample(
            f"What is {pct_str} as a decimal?",
            f"{pct_str} means {pct_str.replace('%','')} per 100 = {dec_str}\n"
            f"Answer: {dec_str}"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 3 — Pre-Algebra (equation solving, step-by-step)
# ══════════════════════════════════════════════════════════════

def gen_stage3(target: int = 500) -> list:
    out = []

    # ── 3a. Evaluate expression — substitute then compute
    for _ in range(100):
        x = random.randint(-5, 10)
        a = random.randint(1, 6)
        b = random.randint(0, 10)
        r = a * x + b
        out.append(sample(
            f"Evaluate {a}x + {b} when x = {x}.",
            f"Step 1: Substitute x = {x}:\n"
            f"  {a}({x}) + {b}\n"
            f"Step 2: Multiply: {a} × {x} = {a*x}\n"
            f"Step 3: Add: {a*x} + {b} = {r}\n"
            f"Answer: {r}"
        ))

    # ── 3b. One-step equations — inverse operation
    for _ in range(100):
        x = random.randint(1, 20)
        a = random.randint(2, 12)
        b = a * x
        out.append(sample(
            f"Solve for x: {a}x = {b}",
            f"Operation: divide both sides by {a}\n"
            f"  {a}x ÷ {a} = {b} ÷ {a}\n"
            f"  x = {x}\n"
            f"Answer: x = {x}"
        ))

    # ── 3c. Two-step equations
    for _ in range(100):
        x = random.randint(1, 15)
        a = random.randint(2, 8)
        b = random.randint(1, 10)
        c = a * x + b
        out.append(sample(
            f"Solve for x: {a}x + {b} = {c}",
            f"Step 1: Subtract {b} from both sides:\n"
            f"  {a}x + {b} - {b} = {c} - {b}\n"
            f"  {a}x = {c - b}\n"
            f"Step 2: Divide both sides by {a}:\n"
            f"  {a}x ÷ {a} = {c - b} ÷ {a}\n"
            f"  x = {x}\n"
            f"Answer: x = {x}"
        ))

    # ── 3d. Negative x two-step equations
    for _ in range(60):
        x = random.randint(-15, -1)
        a = random.randint(2, 8)
        b = random.randint(1, 10)
        c = a * x + b
        out.append(sample(
            f"Solve for x: {a}x + {b} = {c}",
            f"Step 1: Subtract {b} from both sides:\n"
            f"  {a}x = {c} - {b} = {c - b}\n"
            f"Step 2: Divide both sides by {a}:\n"
            f"  x = {c - b} ÷ {a} = {x}\n"
            f"Answer: x = {x}"
        ))

    # ── 3e. Inequalities
    for _ in range(80):
        x_bound = random.randint(1, 15)
        a = random.randint(2, 6)
        b = a * x_bound
        sign, flip = random.choice([("<", ">"), (">", "<"), ("≤", "≥"), ("≥", "≤")])
        out.append(sample(
            f"Solve: {a}x {sign} {b}",
            f"Divide both sides by {a} (positive, inequality direction stays):\n"
            f"  x {sign} {b} ÷ {a}\n"
            f"  x {sign} {x_bound}\n"
            f"Answer: x {sign} {x_bound}"
        ))

    # ── 3f. Substitution check — verify a solution
    for _ in range(60):
        x = random.randint(1, 10)
        a = random.randint(2, 6)
        b = random.randint(1, 8)
        c = a * x + b
        out.append(sample(
            f"Is x = {x} a solution to {a}x + {b} = {c}?",
            f"Step 1: Substitute x = {x}:\n"
            f"  {a}({x}) + {b} = {a*x} + {b} = {c}\n"
            f"Step 2: Compare to right side: {c} = {c} ✓\n"
            f"Answer: Yes, x = {x} is a solution."
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 4 — Algebra (explicit algorithmic steps)
# ══════════════════════════════════════════════════════════════

def gen_stage4(target: int = 500) -> list:
    out = []

    # ── 4a. Slope from two points — rigid formula
    for _ in range(80):
        x1, y1 = random.randint(-5, 4), random.randint(-5, 5)
        dx = random.randint(1, 5)
        m = random.randint(-4, 4) or 1
        x2, y2 = x1 + dx, y1 + m * dx
        out.append(sample(
            f"Find the slope of the line through ({x1}, {y1}) and ({x2}, {y2}).",
            f"Formula: m = (y₂ - y₁) / (x₂ - x₁)\n"
            f"Step 1: y₂ - y₁ = {y2} - {y1} = {y2 - y1}\n"
            f"Step 2: x₂ - x₁ = {x2} - {x1} = {x2 - x1}\n"
            f"Step 3: m = {y2 - y1} / {x2 - x1} = {m}\n"
            f"Answer: slope = {m}"
        ))

    # ── 4b. Slope-intercept from slope + point
    for _ in range(60):
        m = random.randint(-4, 4) or 1
        x1, b = random.randint(-4, 4), random.randint(-5, 5)
        y1 = m * x1 + b
        out.append(sample(
            f"A line has slope {m} and passes through ({x1}, {y1}). Find its equation.",
            f"Formula: y = mx + b\n"
            f"Step 1: Substitute slope m = {m} and point ({x1}, {y1}):\n"
            f"  {y1} = {m}({x1}) + b\n"
            f"Step 2: Solve for b:\n"
            f"  {y1} = {m * x1} + b\n"
            f"  b = {y1} - {m * x1} = {b}\n"
            f"Step 3: Write equation:\n"
            f"Answer: y = {m}x + {b}"
        ))

    # ── 4c. Quadratic — solve by factoring (rigid format)
    for _ in range(80):
        r1 = random.randint(-6, 6) or 1
        r2 = random.randint(-6, 6) or 2
        b_val = -(r1 + r2)
        c_val = r1 * r2
        b_str = f"{b_val:+d}x" if b_val != 0 else ""
        c_str = f"{c_val:+d}" if c_val != 0 else ""
        eq = f"x²{b_str}{c_str} = 0"
        out.append(sample(
            f"Solve: {eq}",
            f"Step 1: Find two numbers that multiply to {c_val} and add to {b_val}:\n"
            f"  {-r1} × {-r2} = {c_val}  and  {-r1} + {-r2} = {b_val} ✓\n"
            f"Step 2: Factor:\n"
            f"  (x {-r1:+d})(x {-r2:+d}) = 0\n"
            f"Step 3: Set each factor to zero:\n"
            f"  x {-r1:+d} = 0  →  x = {r1}\n"
            f"  x {-r2:+d} = 0  →  x = {r2}\n"
            f"Answer: x = {r1} and x = {r2}"
        ))

    # ── 4d. Quadratic formula
    for _ in range(60):
        r1 = random.randint(-4, 4) or 1
        r2 = random.randint(-4, 4) or -1
        a_coef = 1
        b_val = -(r1 + r2)
        c_val = r1 * r2
        disc = b_val**2 - 4 * a_coef * c_val
        disc_sqrt = int(math.isqrt(disc)) if disc >= 0 else 0
        if disc_sqrt * disc_sqrt != disc or disc < 0:
            continue
        b_str = f"{b_val:+d}x" if b_val != 0 else ""
        c_str = f"{c_val:+d}" if c_val != 0 else ""
        out.append(sample(
            f"Use the quadratic formula to solve x²{b_str}{c_str} = 0.",
            f"Quadratic formula: x = (-b ± √(b²-4ac)) / 2a\n"
            f"a = 1, b = {b_val}, c = {c_val}\n"
            f"Step 1: discriminant = {b_val}² - 4(1)({c_val}) = {disc}\n"
            f"Step 2: √{disc} = {disc_sqrt}\n"
            f"Step 3: x = ({-b_val} ± {disc_sqrt}) / 2\n"
            f"  x₁ = ({-b_val} + {disc_sqrt}) / 2 = {(-b_val + disc_sqrt) // 2} = {r1}\n"
            f"  x₂ = ({-b_val} - {disc_sqrt}) / 2 = {(-b_val - disc_sqrt) // 2} = {r2}\n"
            f"Answer: x = {r1} and x = {r2}"
        ))

    # ── 4e. Systems of equations — substitution method
    for _ in range(80):
        x, y = random.randint(1, 6), random.randint(1, 6)
        a1 = random.randint(1, 4)
        b1 = random.randint(1, 4)
        c1 = a1 * x + b1 * y
        a2 = random.randint(1, 4)
        b2_opt = [b for b in range(1, 5) if b != b1]
        b2 = random.choice(b2_opt)
        c2 = a2 * x + b2 * y
        out.append(sample(
            f"Solve the system:\n  {a1}x + {b1}y = {c1}\n  {a2}x + {b2}y = {c2}",
            f"Step 1: From equation 1, isolate x:\n"
            f"  {a1}x = {c1} - {b1}y\n"
            f"  x = ({c1} - {b1}y) / {a1}\n"
            f"Step 2: Substitute into equation 2:\n"
            f"  {a2}·({c1} - {b1}y)/{a1} + {b2}y = {c2}\n"
            f"Step 3: Solve for y → y = {y}\n"
            f"Step 4: Back-substitute → x = {x}\n"
            f"Answer: x = {x}, y = {y}"
        ))

    # ── 4f. Logarithm facts — exhaustive common values
    log_facts = [
        (2, 1, 2), (2, 2, 4), (2, 3, 8), (2, 4, 16), (2, 5, 32), (2, 6, 64),
        (3, 1, 3), (3, 2, 9), (3, 3, 27), (3, 4, 81),
        (5, 1, 5), (5, 2, 25), (5, 3, 125),
        (10, 1, 10), (10, 2, 100), (10, 3, 1000),
    ]
    for base, exp, val in log_facts:
        out.append(sample(
            f"What is log base {base} of {val}?",
            f"Ask: {base} to what power equals {val}?\n"
            f"{base}^{exp} = {val}\n"
            f"Answer: {exp}"
        ))
        out.append(sample(
            f"What is {base}^{exp}?",
            f"{' × '.join([str(base)] * exp)} = {val}\n"
            f"Answer: {val}"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 5 — Geometry (formula → substitution → compute)
# ══════════════════════════════════════════════════════════════

def gen_stage5(target: int = 500) -> list:
    out = []

    # ── 5a. Rectangle area
    for _ in range(80):
        l = random.randint(2, 30)
        w = random.randint(2, 30)
        out.append(sample(
            f"Find the area of a rectangle with length {l} and width {w}.",
            f"Formula: Area = length × width\n"
            f"Substitute: Area = {l} × {w}\n"
            f"Compute: {l} × {w} = {l*w}\n"
            f"Answer: {l*w} square units"
        ))

    # ── 5b. Rectangle perimeter
    for _ in range(60):
        l, w = random.randint(2, 30), random.randint(2, 30)
        out.append(sample(
            f"Find the perimeter of a rectangle with length {l} and width {w}.",
            f"Formula: Perimeter = 2 × (length + width)\n"
            f"Substitute: P = 2 × ({l} + {w})\n"
            f"Step 1: {l} + {w} = {l + w}\n"
            f"Step 2: 2 × {l + w} = {2*(l+w)}\n"
            f"Answer: {2*(l+w)} units"
        ))

    # ── 5c. Circle area and circumference
    for r in range(1, 16):
        area = round(math.pi * r * r, 2)
        circ = round(2 * math.pi * r, 2)
        out.append(sample(
            f"Find the area of a circle with radius {r}.",
            f"Formula: Area = π × r²\n"
            f"Substitute: Area = π × {r}²\n"
            f"Compute: π × {r*r} ≈ {area}\n"
            f"Answer: {area} square units"
        ))
        out.append(sample(
            f"Find the circumference of a circle with radius {r}.",
            f"Formula: Circumference = 2 × π × r\n"
            f"Substitute: C = 2 × π × {r}\n"
            f"Compute: 2 × {r} × π ≈ {circ}\n"
            f"Answer: {circ} units"
        ))

    # ── 5d. Pythagorean theorem — exhaustive triples
    triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(6,8,10),(9,12,15),
               (12,16,20),(15,20,25),(20,21,29),(9,40,41),(10,24,26)]
    for a, b, c in triples:
        out.append(sample(
            f"A right triangle has legs {a} and {b}. Find the hypotenuse.",
            f"Formula: c² = a² + b²\n"
            f"Step 1: a² = {a}² = {a**2}\n"
            f"Step 2: b² = {b}² = {b**2}\n"
            f"Step 3: c² = {a**2} + {b**2} = {a**2+b**2}\n"
            f"Step 4: c = √{a**2+b**2} = {c}\n"
            f"Answer: {c}"
        ))
        out.append(sample(
            f"A right triangle has hypotenuse {c} and one leg {a}. Find the other leg.",
            f"Formula: b² = c² - a²\n"
            f"Step 1: c² = {c}² = {c**2}\n"
            f"Step 2: a² = {a}² = {a**2}\n"
            f"Step 3: b² = {c**2} - {a**2} = {b**2}\n"
            f"Step 4: b = √{b**2} = {b}\n"
            f"Answer: {b}"
        ))

    # ── 5e. Triangle area
    for _ in range(60):
        base, height = random.randint(2, 20), random.randint(2, 20)
        area = base * height / 2
        out.append(sample(
            f"Find the area of a triangle with base {base} and height {height}.",
            f"Formula: Area = (1/2) × base × height\n"
            f"Substitute: Area = (1/2) × {base} × {height}\n"
            f"Compute: 0.5 × {base*height} = {area}\n"
            f"Answer: {area} square units"
        ))

    # ── 5f. Supplementary and complementary angles
    for deg in range(10, 90, 5):
        comp = 90 - deg
        supp = 180 - deg
        out.append(sample(
            f"If one angle is {deg}°, what is its complement?",
            f"Complementary angles add to 90°.\n"
            f"{deg}° + x = 90°\n"
            f"x = 90° - {deg}° = {comp}°\n"
            f"Answer: {comp}°"
        ))
        out.append(sample(
            f"If one angle is {deg}°, what is its supplement?",
            f"Supplementary angles add to 180°.\n"
            f"{deg}° + x = 180°\n"
            f"x = 180° - {deg}° = {supp}°\n"
            f"Answer: {supp}°"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 6 — Statistics & Probability (algorithmic)
# ══════════════════════════════════════════════════════════════

def gen_stage6(target: int = 400) -> list:
    out = []

    # ── 6a. Mean — exhaustive small lists
    for _ in range(100):
        nums = [random.randint(1, 30) for _ in range(random.randint(3, 7))]
        total = sum(nums)
        mean = round(total / len(nums), 2)
        out.append(sample(
            f"Find the mean of: {', '.join(map(str, nums))}",
            f"Step 1: Add all values:\n"
            f"  {' + '.join(map(str, nums))} = {total}\n"
            f"Step 2: Divide by count ({len(nums)}):\n"
            f"  {total} ÷ {len(nums)} = {mean}\n"
            f"Answer: {mean}"
        ))

    # ── 6b. Median — explicit sorting + middle
    for _ in range(80):
        nums = [random.randint(1, 30) for _ in range(random.randint(3, 7))]
        s = sorted(nums)
        n = len(s)
        if n % 2 == 1:
            med = s[n // 2]
            med_explanation = f"Middle value (position {n//2 + 1} of {n}): {med}"
        else:
            med = (s[n//2 - 1] + s[n//2]) / 2
            med_explanation = f"Average of middle two: ({s[n//2-1]} + {s[n//2]}) / 2 = {med}"
        out.append(sample(
            f"Find the median of: {', '.join(map(str, nums))}",
            f"Step 1: Sort: {', '.join(map(str, s))}\n"
            f"Step 2: {med_explanation}\n"
            f"Answer: {med}"
        ))

    # ── 6c. Basic probability — simple fractions
    for total in range(4, 13):
        for fav in range(1, total):
            f = Fraction(fav, total)
            pct = round(fav / total * 100, 1)
            out.append(sample(
                f"A bag has {total} balls. {fav} are red. What is the probability of picking a red ball?",
                f"Formula: P = favorable outcomes / total outcomes\n"
                f"P(red) = {fav} / {total}"
                + (f" = {f.numerator}/{f.denominator}" if f.numerator != fav else "") + "\n"
                f"As a percentage: {pct}%\n"
                f"Answer: {f.numerator}/{f.denominator}"
            ))

    # ── 6d. Permutations — small n, r
    for n in range(3, 7):
        for r in range(2, n):
            perm = math.perm(n, r)
            out.append(sample(
                f"How many ways can you arrange {r} items chosen from {n}? (Order matters)",
                f"Formula: P(n,r) = n! / (n-r)!\n"
                f"P({n},{r}) = {n}! / ({n}-{r})! = {n}! / {n-r}!\n"
                f"= {' × '.join(str(n-i) for i in range(r))}\n"
                f"= {perm}\n"
                f"Answer: {perm} ways"
            ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 7 — Word Problems (identify type → equation → solve)
# ══════════════════════════════════════════════════════════════

def gen_stage7(target: int = 400) -> list:
    out = []

    # ── 7a. Distance / speed / time
    for _ in range(100):
        speed = random.randint(20, 120)
        time = random.randint(1, 8)
        dist = speed * time
        choice = random.randint(0, 2)
        if choice == 0:
            out.append(sample(
                f"A car travels at {speed} km/h for {time} hours. How far does it travel?",
                f"Identify: distance problem\n"
                f"Formula: distance = speed × time\n"
                f"Substitute: d = {speed} × {time}\n"
                f"Compute: {speed} × {time} = {dist}\n"
                f"Answer: {dist} km"
            ))
        elif choice == 1:
            out.append(sample(
                f"A car travels {dist} km in {time} hours. What is its speed?",
                f"Identify: speed problem\n"
                f"Formula: speed = distance ÷ time\n"
                f"Substitute: s = {dist} ÷ {time}\n"
                f"Compute: {dist} ÷ {time} = {speed}\n"
                f"Answer: {speed} km/h"
            ))
        else:
            out.append(sample(
                f"A car travels {dist} km at {speed} km/h. How long does it take?",
                f"Identify: time problem\n"
                f"Formula: time = distance ÷ speed\n"
                f"Substitute: t = {dist} ÷ {speed}\n"
                f"Compute: {dist} ÷ {speed} = {time}\n"
                f"Answer: {time} hour{'s' if time != 1 else ''}"
            ))

    # ── 7b. Simple interest
    for _ in range(80):
        P = random.choice([100, 200, 500, 1000, 2000])
        r = random.choice([5, 8, 10, 12, 15])
        t = random.randint(1, 5)
        SI = P * r * t // 100
        out.append(sample(
            f"Find the simple interest on ${P} at {r}% per year for {t} year{'s' if t>1 else ''}.",
            f"Identify: simple interest problem\n"
            f"Formula: SI = (P × r × t) / 100\n"
            f"Substitute: SI = ({P} × {r} × {t}) / 100\n"
            f"Compute: {P*r*t} / 100 = {SI}\n"
            f"Answer: ${SI}"
        ))

    # ── 7c. Ratio problems
    for _ in range(80):
        a, b = random.randint(1, 5), random.randint(1, 5)
        total = (a + b) * random.randint(2, 8)
        part_a = total * a // (a + b)
        out.append(sample(
            f"Two quantities are in ratio {a}:{b}. The total is {total}. Find each part.",
            f"Identify: ratio problem\n"
            f"Step 1: Total parts = {a} + {b} = {a+b}\n"
            f"Step 2: Value of 1 part = {total} ÷ {a+b} = {total//(a+b)}\n"
            f"Step 3: First part = {a} × {total//(a+b)} = {part_a}\n"
            f"Step 4: Second part = {total} - {part_a} = {total - part_a}\n"
            f"Answer: {part_a} and {total - part_a}"
        ))

    # ── 7d. Work rate
    for _ in range(60):
        da = random.randint(2, 8)
        db = random.randint(2, 8)
        combined = round(da * db / (da + db), 2)
        out.append(sample(
            f"A can finish a job in {da} days. B can finish it in {db} days. How long working together?",
            f"Identify: work rate problem\n"
            f"Step 1: Rate of A = 1/{da} job/day\n"
            f"Step 2: Rate of B = 1/{db} job/day\n"
            f"Step 3: Combined rate = 1/{da} + 1/{db} = {da+db}/{da*db}\n"
            f"Step 4: Time = 1 ÷ ({da+db}/{da*db}) = {da*db}/{da+db} = {combined} days\n"
            f"Answer: {combined} days"
        ))

    # ── 7e. Consecutive integers
    for _ in range(80):
        n = random.randint(1, 20)
        s = n + (n+1) + (n+2)
        out.append(sample(
            f"Three consecutive integers sum to {s}. Find them.",
            f"Let the integers be n, n+1, n+2.\n"
            f"Equation: n + (n+1) + (n+2) = {s}\n"
            f"3n + 3 = {s}\n"
            f"3n = {s-3}\n"
            f"n = {(s-3)//3}\n"
            f"Answer: {n}, {n+1}, {n+2}"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 8 — Calculus (rule → apply → simplify)
# ══════════════════════════════════════════════════════════════

def gen_stage8(target: int = 350) -> list:
    out = []

    # ── 8a. Derivatives — power rule (exhaustive for n=1..8, a=1..5)
    for n in range(1, 9):
        for a in range(1, 6):
            coeff = a * n
            new_exp = n - 1
            if new_exp == 0:
                deriv = str(coeff)
            elif new_exp == 1:
                deriv = f"{coeff}x" if coeff != 1 else "x"
            else:
                deriv = f"{coeff}x^{new_exp}" if coeff != 1 else f"x^{new_exp}"
            out.append(sample(
                f"Find the derivative of f(x) = {a if a!=1 else ''}x^{n}.",
                f"Rule: d/dx[xⁿ] = n·xⁿ⁻¹\n"
                f"Step 1: Bring down the power: {a} × {n} = {coeff}\n"
                f"Step 2: Reduce the exponent: {n} - 1 = {new_exp}\n"
                f"Answer: f'(x) = {deriv}"
            ))

    # ── 8b. Derivatives — sum rule
    for _ in range(60):
        n1, a1 = random.randint(2, 6), random.randint(1, 4)
        n2, a2 = random.randint(1, 5), random.randint(1, 4)
        c1, c2 = a1*n1, a2*n2
        e1, e2 = n1-1, n2-1
        d1 = f"{c1}x^{e1}" if e1 > 1 else (f"{c1}x" if e1 == 1 else str(c1))
        d2 = f"{c2}x^{e2}" if e2 > 1 else (f"{c2}x" if e2 == 1 else str(c2))
        out.append(sample(
            f"Find the derivative of f(x) = {a1}x^{n1} + {a2}x^{n2}.",
            f"Rule: differentiate each term separately (sum rule)\n"
            f"d/dx[{a1}x^{n1}] = {a1} × {n1} · x^{n1-1} = {d1}\n"
            f"d/dx[{a2}x^{n2}] = {a2} × {n2} · x^{n2-1} = {d2}\n"
            f"Answer: f'(x) = {d1} + {d2}"
        ))

    # ── 8c. Integrals — power rule (exhaustive)
    for n in range(1, 8):
        for a in range(1, 5):
            new_exp = n + 1
            from fractions import Fraction as F
            coeff = F(a, new_exp)
            coeff_str = f"{a}/{new_exp}" if coeff.denominator != 1 else str(a)
            out.append(sample(
                f"Find ∫{a}x^{n} dx.",
                f"Rule: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C\n"
                f"Step 1: Raise exponent: {n} + 1 = {new_exp}\n"
                f"Step 2: Divide coefficient: {a} / {new_exp} = {coeff_str}\n"
                f"Answer: {coeff_str}x^{new_exp} + C"
            ))

    # ── 8d. Limits — exhaustive common patterns
    limits = [
        ("lim(x→0) of x²",
         "Direct substitution: x² at x=0 is 0² = 0\nAnswer: 0"),
        ("lim(x→2) of x² - 4",
         "Direct substitution: 2² - 4 = 4 - 4 = 0\nAnswer: 0"),
        ("lim(x→3) of (x² - 9)/(x - 3)",
         "Factor numerator: x² - 9 = (x+3)(x-3)\n"
         "Cancel (x-3): lim(x→3) (x+3) = 3+3 = 6\nAnswer: 6"),
        ("lim(x→∞) of 1/x",
         "As x → ∞, 1/x → 0\nAnswer: 0"),
        ("lim(x→∞) of 5/x²",
         "As x → ∞, 5/x² → 0\nAnswer: 0"),
        ("lim(x→0) of (1+x)^(1/x)",
         "This is the definition of e.\nAnswer: e ≈ 2.71828"),
        ("lim(x→1) of (x² - 1)/(x - 1)",
         "Factor: x² - 1 = (x+1)(x-1)\n"
         "Cancel (x-1): lim(x→1) (x+1) = 1+1 = 2\nAnswer: 2"),
        ("lim(x→0) of sin(x)/x",
         "This is a fundamental limit.\nAnswer: 1"),
    ]
    for q_text, a_text in limits:
        out.append(sample(f"Find the {q_text}.", a_text))

    # ── 8e. Conceptual — definitions the model CAN learn
    concepts = [
        ("What is a derivative?",
         "A derivative measures the instantaneous rate of change of a function.\n"
         "f'(x) = lim(h→0) [f(x+h) - f(x)] / h\n"
         "Geometrically, it equals the slope of the tangent line at x."),
        ("What is an integral?",
         "An integral represents the area under a curve.\n"
         "∫f(x)dx gives the family of antiderivatives F(x) + C.\n"
         "The definite integral ∫ₐᵇf(x)dx = F(b) - F(a)."),
        ("What is the power rule for derivatives?",
         "d/dx[xⁿ] = n·xⁿ⁻¹\n"
         "Example: d/dx[x³] = 3x²\n"
         "Bring down the exponent, then reduce it by 1."),
        ("What is the power rule for integration?",
         "∫xⁿ dx = xⁿ⁺¹/(n+1) + C  (when n ≠ -1)\n"
         "Example: ∫x² dx = x³/3 + C\n"
         "Raise the exponent by 1, then divide by the new exponent."),
        ("What is the Fundamental Theorem of Calculus?",
         "It connects differentiation and integration:\n"
         "If F'(x) = f(x), then ∫ₐᵇf(x)dx = F(b) - F(a)\n"
         "Differentiation and integration are inverse operations."),
    ]
    for q_text, a_text in concepts:
        out.append(sample(q_text, a_text))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

STAGES = {
    1: ("stage1_arithmetic",    gen_stage1),
    2: ("stage2_fractions",     gen_stage2),
    3: ("stage3_prealgebra",    gen_stage3),
    4: ("stage4_algebra",       gen_stage4),
    5: ("stage5_geometry",      gen_stage5),
    6: ("stage6_statistics",    gen_stage6),
    7: ("stage7_wordproblems",  gen_stage7),
    8: ("stage8_calculus",      gen_stage8),
}


def write_jsonl(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data):>5} samples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", type=int, default=list(STAGES.keys()))
    parser.add_argument("--stage",  type=int, default=None)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    to_run = [args.stage] if args.stage else args.stages
    all_samples = []

    for num in to_run:
        if num not in STAGES:
            print(f"Unknown stage {num}")
            continue
        name, fn = STAGES[num]
        print(f"\n[Stage {num}] Generating {name}...")
        data = fn()
        if args.preview:
            for s in data[:args.preview]:
                print(json.dumps(s, indent=2, ensure_ascii=False))
            return
        write_jsonl(out_dir / f"yaya_math_{name}.jsonl", data)
        all_samples.extend(data)

    if len(to_run) > 1 and not args.preview:
        random.shuffle(all_samples)
        write_jsonl(out_dir / "yaya_math_combined.jsonl", all_samples)
        print(f"\nTotal: {len(all_samples)} samples across {len(to_run)} stages")


if __name__ == "__main__":
    main()
