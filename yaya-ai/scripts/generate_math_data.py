"""
generate_math_data.py
=====================
Generates Yaya math curriculum training data — 8 progressive stages.

Stage 1: Arithmetic        (counting, +, -, ×, ÷)
Stage 2: Fractions/Decimals/Percentages
Stage 3: Pre-Algebra       (expressions, simple equations)
Stage 4: Algebra           (linear, quadratic, systems)
Stage 5: Geometry          (shapes, area, volume, angles)
Stage 6: Statistics & Probability
Stage 7: Word Problems     (applied multi-step)
Stage 8: Calculus & Beyond (limits, derivatives, integrals — conceptual)

Usage:
    python scripts/generate_math_data.py
    python scripts/generate_math_data.py --stages 1 2 3
    python scripts/generate_math_data.py --stage 1 --preview 5
"""

import sys
import json
import random
import argparse
import math
from pathlib import Path
from fractions import Fraction

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

SYSTEM_MSG = (
    "You are Yaya, a helpful, honest, and friendly AI assistant. "
    "You answer questions clearly and thoughtfully."
)

OUTPUT_DIR = Path("data/sft/math")

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def make_sample(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def shuffle(lst: list) -> list:
    random.shuffle(lst)
    return lst


# ──────────────────────────────────────────────────────────
# STAGE 1 — Arithmetic
# ──────────────────────────────────────────────────────────

def gen_stage1(n: int = 600) -> list:
    samples = []

    # Addition
    for _ in range(n // 6):
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        result = a + b
        templates = [
            (f"What is {a} + {b}?", f"{a} + {b} = {result}"),
            (f"Calculate {a} plus {b}.", f"{a} + {b} = {result}"),
            (f"Add {a} and {b}.", f"Adding {a} and {b}:\n{a} + {b} = **{result}**"),
            (f"What is the sum of {a} and {b}?",
             f"Sum = {a} + {b} = **{result}**"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    # Subtraction
    for _ in range(n // 6):
        a = random.randint(1, 999)
        b = random.randint(1, a)
        result = a - b
        templates = [
            (f"What is {a} - {b}?", f"{a} - {b} = {result}"),
            (f"Subtract {b} from {a}.", f"{a} - {b} = **{result}**"),
            (f"What is the difference between {a} and {b}?",
             f"Difference = {a} - {b} = **{result}**"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    # Multiplication
    for _ in range(n // 6):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        result = a * b
        templates = [
            (f"What is {a} × {b}?", f"{a} × {b} = {result}"),
            (f"Multiply {a} by {b}.", f"{a} × {b} = **{result}**"),
            (f"What is the product of {a} and {b}?",
             f"Product = {a} × {b} = **{result}**"),
            (f"Calculate {a} times {b}.", f"{a} × {b} = {result}"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    # Division
    for _ in range(n // 6):
        b = random.randint(1, 20)
        result = random.randint(1, 50)
        a = b * result
        templates = [
            (f"What is {a} ÷ {b}?", f"{a} ÷ {b} = {result}"),
            (f"Divide {a} by {b}.", f"{a} ÷ {b} = **{result}**"),
            (f"What is {a} divided by {b}?",
             f"{a} ÷ {b} = **{result}**"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    # Mixed operations
    for _ in range(n // 6):
        a, b, c = random.randint(1, 50), random.randint(1, 20), random.randint(1, 10)
        r = a + b * c
        samples.append(make_sample(
            f"Calculate {a} + {b} × {c}.",
            f"Following order of operations (multiplication before addition):\n"
            f"Step 1: {b} × {c} = {b*c}\n"
            f"Step 2: {a} + {b*c} = **{r}**"
        ))

    # Exponents / squares / square roots
    for _ in range(n // 6):
        n_val = random.randint(2, 20)
        sq = n_val ** 2
        choice = random.randint(0, 2)
        if choice == 0:
            samples.append(make_sample(
                f"What is {n_val} squared?",
                f"{n_val}² = {n_val} × {n_val} = **{sq}**"
            ))
        elif choice == 1:
            samples.append(make_sample(
                f"What is the square root of {sq}?",
                f"√{sq} = **{n_val}** (because {n_val} × {n_val} = {sq})"
            ))
        else:
            e = random.randint(2, 4)
            result = n_val ** e
            samples.append(make_sample(
                f"What is {n_val}^{e}?",
                f"{n_val}^{e} = " + " × ".join([str(n_val)] * e) + f" = **{result}**"
            ))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# STAGE 2 — Fractions, Decimals, Percentages
# ──────────────────────────────────────────────────────────

def gen_stage2(n: int = 500) -> list:
    samples = []

    # Fraction simplification
    for _ in range(n // 5):
        d = random.randint(2, 12)
        factor = random.randint(2, 6)
        num = d * factor
        den = random.randint(d + 1, d * 3) * factor
        f = Fraction(num, den)
        samples.append(make_sample(
            f"Simplify the fraction {num}/{den}.",
            f"Find the GCD of {num} and {den}.\n"
            f"GCD({num}, {den}) = {math.gcd(num, den)}\n"
            f"Divide both by {math.gcd(num, den)}:\n"
            f"{num}/{den} = **{f.numerator}/{f.denominator}**"
        ))

    # Fraction addition
    for _ in range(n // 5):
        a = Fraction(random.randint(1, 5), random.randint(2, 8))
        b = Fraction(random.randint(1, 5), random.randint(2, 8))
        result = a + b
        samples.append(make_sample(
            f"What is {a.numerator}/{a.denominator} + {b.numerator}/{b.denominator}?",
            f"Find a common denominator. LCD({a.denominator}, {b.denominator}) = {result.denominator}\n"
            f"{a.numerator}/{a.denominator} = {a.numerator * (result.denominator // a.denominator)}/{result.denominator}\n"
            f"{b.numerator}/{b.denominator} = {b.numerator * (result.denominator // b.denominator)}/{result.denominator}\n"
            f"Sum = **{result.numerator}/{result.denominator}**"
        ))

    # Decimal operations
    for _ in range(n // 5):
        a = round(random.uniform(0.1, 99.9), 2)
        b = round(random.uniform(0.1, 99.9), 2)
        op = random.choice(["+", "-", "×"])
        if op == "+":
            r = round(a + b, 4)
            samples.append(make_sample(
                f"What is {a} + {b}?",
                f"Align decimal points:\n  {a}\n+ {b}\n= **{r}**"
            ))
        elif op == "-" and a >= b:
            r = round(a - b, 4)
            samples.append(make_sample(
                f"What is {a} - {b}?",
                f"{a} - {b} = **{r}**"
            ))
        else:
            r = round(a * b, 4)
            samples.append(make_sample(
                f"What is {a} × {b}?",
                f"{a} × {b} = **{r}**"
            ))

    # Percentage calculations
    for _ in range(n // 5):
        pct = random.choice([10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
        whole = random.randint(20, 500)
        result = round(pct / 100 * whole, 2)
        templates = [
            (f"What is {pct}% of {whole}?",
             f"{pct}% of {whole} = ({pct}/100) × {whole} = **{result}**"),
            (f"Calculate {pct} percent of {whole}.",
             f"({pct} ÷ 100) × {whole} = {pct/100} × {whole} = **{result}**"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    # Fraction ↔ Decimal ↔ Percent conversion
    conversions = [
        ("1/2", 0.5, 50),
        ("1/4", 0.25, 25),
        ("3/4", 0.75, 75),
        ("1/5", 0.2, 20),
        ("2/5", 0.4, 40),
        ("3/5", 0.6, 60),
        ("1/3", round(1/3, 4), round(100/3, 2)),
        ("2/3", round(2/3, 4), round(200/3, 2)),
        ("1/8", 0.125, 12.5),
        ("3/8", 0.375, 37.5),
    ]
    for _ in range(n // 5):
        frac_str, dec, pct = random.choice(conversions)
        choice = random.randint(0, 2)
        if choice == 0:
            samples.append(make_sample(
                f"Convert {frac_str} to a decimal.",
                f"{frac_str} = **{dec}**"
            ))
        elif choice == 1:
            samples.append(make_sample(
                f"Convert {frac_str} to a percentage.",
                f"{frac_str} = {dec} = **{pct}%**"
            ))
        else:
            samples.append(make_sample(
                f"Convert {dec} to a fraction.",
                f"{dec} = **{frac_str}**"
            ))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# STAGE 3 — Pre-Algebra
# ──────────────────────────────────────────────────────────

def gen_stage3(n: int = 400) -> list:
    samples = []

    # Evaluate expressions
    for _ in range(n // 4):
        x = random.randint(-10, 10)
        a = random.randint(1, 5)
        b = random.randint(0, 10)
        result = a * x + b
        templates = [
            (f"If x = {x}, what is {a}x + {b}?",
             f"Substitute x = {x}:\n{a}({x}) + {b} = {a*x} + {b} = **{result}**"),
            (f"Evaluate {a}x + {b} when x = {x}.",
             f"{a}({x}) + {b} = {a*x} + {b} = **{result}**"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    # Solve one-step equations
    for _ in range(n // 4):
        x = random.randint(-20, 20)
        a = random.randint(1, 12)
        b = a * x  # ax = b
        samples.append(make_sample(
            f"Solve for x: {a}x = {b}",
            f"Divide both sides by {a}:\n"
            f"x = {b} ÷ {a} = **{x}**"
        ))

    # Solve two-step equations
    for _ in range(n // 4):
        x = random.randint(-15, 15)
        a = random.randint(1, 8)
        b = random.randint(-10, 10)
        c = a * x + b
        samples.append(make_sample(
            f"Solve for x: {a}x + {b} = {c}",
            f"Step 1: Subtract {b} from both sides:\n"
            f"  {a}x = {c} - {b} = {c - b}\n"
            f"Step 2: Divide both sides by {a}:\n"
            f"  x = {c - b} ÷ {a} = **{x}**"
        ))

    # Inequalities
    for _ in range(n // 4):
        x_min = random.randint(1, 10)
        a = random.randint(1, 5)
        b = a * x_min
        sign = random.choice([">", "≥", "<", "≤"])
        if sign in [">", "≥"]:
            answer = f"x {sign} {x_min}" if sign == "≥" else f"x > {x_min}"
        else:
            answer = f"x {sign} {x_min}" if sign == "≤" else f"x < {x_min}"
        samples.append(make_sample(
            f"Solve the inequality: {a}x {sign} {b}",
            f"Divide both sides by {a}:\n"
            f"**{answer}**\n\n"
            f"This means x can be any number that is {sign} {x_min}."
        ))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# STAGE 4 — Algebra
# ──────────────────────────────────────────────────────────

def gen_stage4(n: int = 400) -> list:
    samples = []

    # Linear equation from two points
    for _ in range(n // 5):
        x1, x2 = random.randint(-5, 0), random.randint(1, 6)
        m = random.randint(-4, 4) or 1
        b = random.randint(-5, 5)
        y1, y2 = m * x1 + b, m * x2 + b
        samples.append(make_sample(
            f"Find the slope-intercept equation of the line through ({x1}, {y1}) and ({x2}, {y2}).",
            f"Step 1: Find the slope:\n"
            f"  m = (y₂ - y₁)/(x₂ - x₁) = ({y2} - {y1})/({x2} - {x1}) = {y2-y1}/{x2-x1} = {m}\n"
            f"Step 2: Use point-slope form with ({x1}, {y1}):\n"
            f"  y - {y1} = {m}(x - {x1})\n"
            f"  y = {m}x + {b}\n\n"
            f"**Equation: y = {m}x + {b}**"
        ))

    # Quadratic — solve by factoring
    factoring_pairs = [
        (2, 3, 6, 5),   # (x+2)(x+3) = x²+5x+6
        (1, 4, 4, 5),
        (3, 2, 6, 5),
        (-1, -3, 3, -4),
        (2, -5, -10, -3),
        (-2, 3, -6, 1),
        (1, 6, 6, 7),
        (4, 1, 4, 5),
    ]
    for _ in range(n // 5):
        r1 = random.randint(-8, 8) or 1
        r2 = random.randint(-8, 8) or 2
        b_val = -(r1 + r2)
        c_val = r1 * r2
        b_str = f"{b_val:+}x" if b_val != 0 else ""
        c_str = f"{c_val:+}" if c_val != 0 else ""
        eq = f"x² {b_str} {c_str} = 0".replace("  ", " ").strip()
        samples.append(make_sample(
            f"Solve: {eq}",
            f"Factor the quadratic:\n"
            f"(x - {r1})(x - {r2}) = 0\n\n"
            f"Set each factor to zero:\n"
            f"x - {r1} = 0  →  x = {r1}\n"
            f"x - {r2} = 0  →  x = {r2}\n\n"
            f"**Solutions: x = {r1} and x = {r2}**"
        ))

    # System of equations
    for _ in range(n // 5):
        x, y = random.randint(-5, 5), random.randint(-5, 5)
        a1, b1 = random.randint(1, 4), random.randint(1, 4)
        a2, b2 = random.randint(1, 4), random.randint(1, 4)
        c1, c2 = a1*x + b1*y, a2*x + b2*y
        samples.append(make_sample(
            f"Solve the system:\n{a1}x + {b1}y = {c1}\n{a2}x + {b2}y = {c2}",
            f"From equation 1: x = ({c1} - {b1}y) / {a1}\n"
            f"Substitute into equation 2:\n"
            f"{a2}(({c1} - {b1}y)/{a1}) + {b2}y = {c2}\n"
            f"Solving: y = {y}\n"
            f"Then: x = ({c1} - {b1}×{y}) / {a1} = {x}\n\n"
            f"**Solution: x = {x}, y = {y}**"
        ))

    # Polynomial expansion
    for _ in range(n // 5):
        a, b = random.randint(1, 5), random.randint(1, 5)
        # (ax + b)^2 = a²x² + 2abx + b²
        samples.append(make_sample(
            f"Expand ({a}x + {b})².",
            f"Use the formula (A + B)² = A² + 2AB + B²\n"
            f"A = {a}x, B = {b}\n"
            f"= ({a}x)² + 2({a}x)({b}) + ({b})²\n"
            f"= {a**2}x² + {2*a*b}x + {b**2}\n\n"
            f"**= {a**2}x² + {2*a*b}x + {b**2}**"
        ))

    # Logarithms
    log_examples = [
        (2, 8, 3, "log₂(8) = 3 because 2³ = 8"),
        (3, 9, 2, "log₃(9) = 2 because 3² = 9"),
        (10, 100, 2, "log₁₀(100) = 2 because 10² = 100"),
        (10, 1000, 3, "log₁₀(1000) = 3 because 10³ = 1000"),
        (2, 16, 4, "log₂(16) = 4 because 2⁴ = 16"),
        (2, 32, 5, "log₂(32) = 5 because 2⁵ = 32"),
        (5, 25, 2, "log₅(25) = 2 because 5² = 25"),
        (5, 125, 3, "log₅(125) = 3 because 5³ = 125"),
        (4, 64, 3, "log₄(64) = 3 because 4³ = 64"),
        (3, 81, 4, "log₃(81) = 4 because 3⁴ = 81"),
    ]
    for _ in range(n // 5):
        base, val, exp, expl = random.choice(log_examples)
        templates = [
            (f"What is log base {base} of {val}?", f"**Answer: {exp}**\n{expl}"),
            (f"Evaluate log_{base}({val}).", f"{expl}\n\n**log_{base}({val}) = {exp}**"),
            (f"If log_{base}(x) = {exp}, what is x?",
             f"log_{base}(x) = {exp}  means  {base}^{exp} = x\n"
             f"**x = {val}**"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# STAGE 5 — Geometry
# ──────────────────────────────────────────────────────────

def gen_stage5(n: int = 400) -> list:
    samples = []

    # Area & perimeter — basic shapes
    shapes = []
    for _ in range(n // 5):
        l, w = random.randint(2, 30), random.randint(2, 30)
        shapes.append(make_sample(
            f"What is the area and perimeter of a rectangle with length {l} and width {w}?",
            f"**Area** = length × width = {l} × {w} = **{l*w} square units**\n"
            f"**Perimeter** = 2(length + width) = 2({l} + {w}) = 2 × {l+w} = **{2*(l+w)} units**"
        ))
    samples.extend(shapes)

    # Circle
    for _ in range(n // 5):
        r = random.randint(1, 20)
        area = round(math.pi * r**2, 4)
        circ = round(2 * math.pi * r, 4)
        samples.append(make_sample(
            f"Find the area and circumference of a circle with radius {r}.",
            f"**Area** = πr² = π × {r}² = π × {r**2} ≈ **{area:.2f} square units**\n"
            f"**Circumference** = 2πr = 2π × {r} ≈ **{circ:.2f} units**"
        ))

    # Triangle — Pythagorean theorem
    pythagorean_triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(6,8,10),(9,12,15),(12,16,20)]
    for _ in range(n // 5):
        a, b, c = random.choice(pythagorean_triples)
        choice = random.randint(0, 1)
        if choice == 0:
            samples.append(make_sample(
                f"A right triangle has legs of {a} and {b}. What is the hypotenuse?",
                f"Using the Pythagorean theorem: c² = a² + b²\n"
                f"c² = {a}² + {b}² = {a**2} + {b**2} = {a**2 + b**2}\n"
                f"c = √{a**2 + b**2} = **{c}**"
            ))
        else:
            samples.append(make_sample(
                f"A right triangle has hypotenuse {c} and one leg {a}. What is the other leg?",
                f"Using the Pythagorean theorem: b² = c² - a²\n"
                f"b² = {c}² - {a}² = {c**2} - {a**2} = {b**2}\n"
                f"b = √{b**2} = **{b}**"
            ))

    # Area of triangle
    for _ in range(n // 5):
        base = random.randint(2, 20)
        height = random.randint(2, 20)
        area = (base * height) / 2
        samples.append(make_sample(
            f"What is the area of a triangle with base {base} and height {height}?",
            f"Area of triangle = ½ × base × height\n"
            f"= ½ × {base} × {height}\n"
            f"= **{area:.1f} square units**"
        ))

    # Angles
    for _ in range(n // 5):
        a1 = random.randint(10, 160)
        a2 = 180 - a1
        choice = random.randint(0, 2)
        if choice == 0:
            samples.append(make_sample(
                f"Two angles are supplementary. One angle is {a1}°. What is the other?",
                f"Supplementary angles sum to 180°.\n"
                f"{a1}° + x = 180°\n"
                f"x = 180° - {a1}° = **{a2}°**"
            ))
        elif choice == 1:
            c1 = random.randint(5, 85)
            c2 = 90 - c1
            samples.append(make_sample(
                f"Two angles are complementary. One angle is {c1}°. What is the other?",
                f"Complementary angles sum to 90°.\n"
                f"{c1}° + x = 90°\n"
                f"x = 90° - {c1}° = **{c2}°**"
            ))
        else:
            ang = random.choice([30, 45, 60, 90, 120, 135, 150])
            samples.append(make_sample(
                f"What type of angle is {ang}°?",
                {
                    30: "**Acute angle** — less than 90°",
                    45: "**Acute angle** — less than 90°",
                    60: "**Acute angle** — less than 90°",
                    90: "**Right angle** — exactly 90°",
                    120: "**Obtuse angle** — greater than 90° but less than 180°",
                    135: "**Obtuse angle** — greater than 90° but less than 180°",
                    150: "**Obtuse angle** — greater than 90° but less than 180°",
                }[ang]
            ))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# STAGE 6 — Statistics & Probability
# ──────────────────────────────────────────────────────────

def gen_stage6(n: int = 350) -> list:
    samples = []

    # Mean, median, mode, range
    for _ in range(n // 4):
        data = sorted(random.sample(range(1, 50), random.randint(5, 9)))
        mean_val = round(sum(data) / len(data), 4)
        median_idx = len(data) // 2
        median_val = data[median_idx] if len(data) % 2 == 1 else (data[median_idx-1] + data[median_idx]) / 2
        rng = max(data) - min(data)
        data_str = ", ".join(map(str, data))
        samples.append(make_sample(
            f"Find the mean, median, and range of: {data_str}",
            f"**Mean** = sum / count = ({' + '.join(map(str, data))}) / {len(data)} = {sum(data)} / {len(data)} = **{mean_val}**\n\n"
            f"**Median** (middle value of sorted list): **{median_val}**\n\n"
            f"**Range** = max - min = {max(data)} - {min(data)} = **{rng}**"
        ))

    # Basic probability
    for _ in range(n // 4):
        total = random.randint(5, 20)
        fav = random.randint(1, total - 1)
        f = Fraction(fav, total)
        pct = round(fav / total * 100, 2)
        templates = [
            (f"A bag has {total} balls, of which {fav} are red. What is the probability of picking a red ball?",
             f"P(red) = favorable outcomes / total outcomes = {fav}/{total} = **{f.numerator}/{f.denominator}** ≈ {pct}%"),
            (f"You roll a fair die. What is the probability of rolling a number greater than {total // 2}?",
             f"Numbers 1–6 greater than {total // 2}: {[x for x in range(1,7) if x > total//2]}\n"
             f"P = {len([x for x in range(1,7) if x > total//2])}/6 = **{Fraction(len([x for x in range(1,7) if x > total//2]), 6)}**"),
        ]
        q, a_text = random.choice(templates)
        samples.append(make_sample(q, a_text))

    # Permutations & combinations
    for _ in range(n // 4):
        n_val = random.randint(4, 8)
        r_val = random.randint(2, n_val - 1)
        perm = math.perm(n_val, r_val)
        comb = math.comb(n_val, r_val)
        choice = random.randint(0, 1)
        if choice == 0:
            samples.append(make_sample(
                f"How many ways can you arrange {r_val} items from a set of {n_val}? (Order matters)",
                f"This is a permutation: P({n_val},{r_val}) = {n_val}! / ({n_val}-{r_val})!\n"
                f"= {n_val}! / {n_val-r_val}!\n"
                f"= **{perm}** ways"
            ))
        else:
            samples.append(make_sample(
                f"How many ways can you choose {r_val} items from {n_val}? (Order doesn't matter)",
                f"This is a combination: C({n_val},{r_val}) = {n_val}! / ({r_val}! × ({n_val}-{r_val})!)\n"
                f"= **{comb}** ways"
            ))

    # Standard deviation concept
    for _ in range(n // 4):
        data = [random.randint(1, 20) for _ in range(6)]
        mean = sum(data) / len(data)
        variance = sum((x - mean)**2 for x in data) / len(data)
        std = round(math.sqrt(variance), 4)
        samples.append(make_sample(
            f"Find the standard deviation of: {', '.join(map(str, data))}",
            f"Step 1: Mean = ({' + '.join(map(str, data))}) / {len(data)} = {round(mean,4)}\n"
            f"Step 2: Squared deviations from mean:\n"
            + "\n".join(f"  ({x} - {round(mean,2)})² = {round((x-mean)**2,4)}" for x in data) +
            f"\nStep 3: Variance = sum / n = {round(variance,4)}\n"
            f"Step 4: Std dev = √{round(variance,4)} = **{std}**"
        ))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# STAGE 7 — Word Problems
# ──────────────────────────────────────────────────────────

def gen_stage7(n: int = 350) -> list:
    samples = []

    # Distance/speed/time
    for _ in range(n // 5):
        speed = random.randint(20, 120)
        time = random.randint(1, 8)
        distance = speed * time
        choice = random.randint(0, 2)
        if choice == 0:
            samples.append(make_sample(
                f"A car travels at {speed} km/h for {time} hours. How far does it travel?",
                f"Distance = Speed × Time\n"
                f"= {speed} × {time}\n"
                f"= **{distance} km**"
            ))
        elif choice == 1:
            samples.append(make_sample(
                f"A train travels {distance} km in {time} hours. What is its average speed?",
                f"Speed = Distance ÷ Time\n"
                f"= {distance} ÷ {time}\n"
                f"= **{speed} km/h**"
            ))
        else:
            samples.append(make_sample(
                f"A cyclist travels {distance} km at {speed} km/h. How long does the journey take?",
                f"Time = Distance ÷ Speed\n"
                f"= {distance} ÷ {speed}\n"
                f"= **{time} hour{'s' if time != 1 else ''}**"
            ))

    # Profit/loss/interest
    for _ in range(n // 5):
        principal = random.choice([100, 200, 500, 1000, 2000, 5000])
        rate = random.choice([5, 8, 10, 12, 15, 20])
        years = random.randint(1, 5)
        simple_interest = principal * rate * years / 100
        total = principal + simple_interest
        samples.append(make_sample(
            f"What is the simple interest on ${principal} at {rate}% per year for {years} year{'s' if years != 1 else ''}?",
            f"Simple Interest = Principal × Rate × Time / 100\n"
            f"= {principal} × {rate} × {years} / 100\n"
            f"= **${simple_interest:.2f}**\n\n"
            f"Total amount after {years} year{'s' if years != 1 else ''} = ${principal} + ${simple_interest:.2f} = **${total:.2f}**"
        ))

    # Mixture/ratio problems
    for _ in range(n // 5):
        ratio_a, ratio_b = random.randint(1, 5), random.randint(1, 5)
        total = random.randint(20, 100)
        part_a = round(total * ratio_a / (ratio_a + ratio_b))
        part_b = total - part_a
        names = random.choice([("boys", "girls"), ("apples", "oranges"), ("red balls", "blue balls")])
        samples.append(make_sample(
            f"The ratio of {names[0]} to {names[1]} is {ratio_a}:{ratio_b}. "
            f"If there are {total} total, how many {names[0]} are there?",
            f"Total ratio parts = {ratio_a} + {ratio_b} = {ratio_a+ratio_b}\n"
            f"{names[0].capitalize()} = ({ratio_a}/{ratio_a+ratio_b}) × {total} = **{part_a}**\n"
            f"{names[1].capitalize()} = {total} - {part_a} = {part_b}"
        ))

    # Age problems
    for _ in range(n // 5):
        current_age = random.randint(5, 40)
        years_later = random.randint(3, 20)
        future_age = current_age + years_later
        samples.append(make_sample(
            f"Alice is {current_age} years old. How old will she be in {years_later} years?",
            f"Future age = Current age + Years\n"
            f"= {current_age} + {years_later}\n"
            f"= **{future_age} years old**"
        ))

    # Work rate problems
    for _ in range(n // 5):
        days_a = random.randint(2, 10)
        days_b = random.randint(2, 10)
        combined_days = (days_a * days_b) / (days_a + days_b)
        samples.append(make_sample(
            f"Person A can finish a job in {days_a} days. Person B can finish it in {days_b} days. "
            f"How long will it take if they work together?",
            f"Rate A = 1/{days_a} job/day\n"
            f"Rate B = 1/{days_b} job/day\n"
            f"Combined rate = 1/{days_a} + 1/{days_b} = {days_b + days_a}/{days_a*days_b}\n"
            f"Time = 1 ÷ (combined rate) = {days_a}×{days_b} ÷ ({days_a}+{days_b}) = **{round(combined_days, 2)} days**"
        ))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# STAGE 8 — Calculus & Beyond (Conceptual + Computations)
# ──────────────────────────────────────────────────────────

def gen_stage8(n: int = 300) -> list:
    samples = []

    # Limits
    limit_examples = [
        ("lim(x→2) of (x² - 4)/(x - 2)",
         "Factor the numerator: x² - 4 = (x+2)(x-2)\n"
         "lim(x→2) (x+2)(x-2)/(x-2) = lim(x→2) (x+2) = 2+2 = **4**"),
        ("lim(x→0) of sin(x)/x",
         "This is a fundamental limit.\n"
         "lim(x→0) sin(x)/x = **1**\n\n"
         "This result is foundational in calculus and can be proved using the squeeze theorem."),
        ("lim(x→∞) of 1/x",
         "As x grows larger and larger, 1/x approaches zero.\n"
         "lim(x→∞) 1/x = **0**"),
        ("lim(x→3) of (x² - 9)/(x - 3)",
         "Factor: x² - 9 = (x+3)(x-3)\n"
         "lim(x→3) (x+3)(x-3)/(x-3) = lim(x→3) (x+3) = **6**"),
        ("lim(x→0) of (1 + x)^(1/x)",
         "This is the definition of e.\n"
         "lim(x→0) (1 + x)^(1/x) = **e ≈ 2.71828**"),
    ]
    for _ in range(n // 4):
        expr, sol = random.choice(limit_examples)
        samples.append(make_sample(f"Find the {expr}.", sol))

    # Derivatives — power rule
    for _ in range(n // 4):
        n_val = random.randint(2, 8)
        a = random.randint(1, 6)
        # d/dx [a * x^n] = a*n * x^(n-1)
        coeff = a * n_val
        new_exp = n_val - 1
        power_str = "x" if new_exp == 1 else (f"x^{new_exp}" if new_exp > 1 else "")
        deriv = f"{coeff}{power_str}" if coeff != 1 or new_exp == 0 else power_str
        samples.append(make_sample(
            f"Find the derivative of f(x) = {a if a != 1 else ''}x^{n_val}.",
            f"Using the power rule: d/dx [xⁿ] = n·xⁿ⁻¹\n"
            f"d/dx [{a}x^{n_val}] = {a} × {n_val} × x^{n_val - 1}\n"
            f"= **{coeff}x^{n_val-1}**"
        ))

    # Integrals — basic
    for _ in range(n // 4):
        n_val = random.randint(1, 6)
        a = random.randint(1, 5)
        new_exp = n_val + 1
        samples.append(make_sample(
            f"Find the integral of {a}x^{n_val}.",
            f"Using the power rule for integration: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C\n"
            f"∫{a}x^{n_val} dx = {a} × x^{new_exp}/{new_exp} + C\n"
            f"= **{a}/{new_exp} x^{new_exp} + C**"
        ))

    # Conceptual questions
    conceptual = [
        ("What is a derivative?",
         "A **derivative** measures how a function changes as its input changes — it's the instantaneous rate of change.\n\n"
         "Formally: f'(x) = lim(h→0) [f(x+h) - f(x)] / h\n\n"
         "Geometrically, the derivative at a point equals the slope of the tangent line to the curve at that point."),
        ("What is an integral?",
         "An **integral** represents the area under a curve.\n\n"
         "The definite integral ∫ₐᵇ f(x) dx gives the signed area between the curve y=f(x) and the x-axis from x=a to x=b.\n\n"
         "The indefinite integral ∫f(x) dx gives the family of antiderivatives F(x) + C."),
        ("What is the Fundamental Theorem of Calculus?",
         "The **Fundamental Theorem of Calculus** connects differentiation and integration:\n\n"
         "**Part 1:** If F(x) = ∫ₐˣ f(t) dt, then F'(x) = f(x)\n\n"
         "**Part 2:** ∫ₐᵇ f(x) dx = F(b) - F(a), where F is any antiderivative of f.\n\n"
         "In short: integration and differentiation are inverse operations."),
        ("What is the chain rule in calculus?",
         "The **chain rule** is used to differentiate composite functions.\n\n"
         "If h(x) = f(g(x)), then:\n"
         "h'(x) = f'(g(x)) · g'(x)\n\n"
         "Example: d/dx [sin(x²)] = cos(x²) · 2x = **2x·cos(x²)**"),
        ("What is a limit in calculus?",
         "A **limit** describes the value a function approaches as the input approaches a given value.\n\n"
         "lim(x→a) f(x) = L means f(x) gets arbitrarily close to L as x gets close to a.\n\n"
         "Limits are the foundation of calculus — derivatives and integrals are both defined using limits."),
        ("Explain the product rule for derivatives.",
         "The **product rule** states: if h(x) = f(x)·g(x), then\n"
         "h'(x) = f'(x)·g(x) + f(x)·g'(x)\n\n"
         "Example: d/dx [x² · sin(x)]\n"
         "= 2x · sin(x) + x² · cos(x)\n"
         "= **2x·sin(x) + x²·cos(x)**"),
    ]
    for _ in range(n // 4):
        q, a_text = random.choice(conceptual)
        samples.append(make_sample(q, a_text))

    return shuffle(samples)


# ──────────────────────────────────────────────────────────
# Build combined dataset
# ──────────────────────────────────────────────────────────

STAGES = {
    1: ("stage1_arithmetic",      gen_stage1),
    2: ("stage2_fractions",       gen_stage2),
    3: ("stage3_prealgebra",      gen_stage3),
    4: ("stage4_algebra",         gen_stage4),
    5: ("stage5_geometry",        gen_stage5),
    6: ("stage6_statistics",      gen_stage6),
    7: ("stage7_wordproblems",    gen_stage7),
    8: ("stage8_calculus",        gen_stage8),
}


def write_jsonl(path: Path, samples: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples):>5} samples → {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Yaya math curriculum data")
    parser.add_argument("--stages", nargs="*", type=int, default=list(STAGES.keys()),
                        help="Which stages to generate (default: all)")
    parser.add_argument("--stage", type=int, default=None,
                        help="Generate a single stage")
    parser.add_argument("--preview", type=int, default=0,
                        help="Print N samples and exit without writing")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    stages_to_run = [args.stage] if args.stage else args.stages

    all_samples = []

    for stage_num in stages_to_run:
        if stage_num not in STAGES:
            print(f"  Unknown stage {stage_num}, skipping")
            continue
        name, gen_fn = STAGES[stage_num]
        print(f"\n[Stage {stage_num}] Generating {name}...")
        samples = gen_fn()

        if args.preview > 0:
            for s in samples[:args.preview]:
                print(json.dumps(s, indent=2, ensure_ascii=False))
            return

        write_jsonl(out_dir / f"yaya_math_{name}.jsonl", samples)
        all_samples.extend(samples)

    if len(stages_to_run) > 1 and not args.preview:
        random.shuffle(all_samples)
        write_jsonl(out_dir / "yaya_math_combined.jsonl", all_samples)
        print(f"\nTotal: {len(all_samples)} math samples across {len(stages_to_run)} stages")


if __name__ == "__main__":
    main()
