"""
generate_math_data.py — v3 (Force-Compute Edition)
====================================================
Core fix: tiny models learn by REPETITION of IDENTICAL patterns.

Key principles:
  1. Every single-digit fact repeated 30x with different question phrasings
  2. ZERO noise in answers — just the number, no "Ones: ... Tens: ..."
  3. Multi-digit uses same rigid 3-line algorithm every single time
  4. High step counts in configs to force memorization through many epochs
  5. Never mix unrelated numbers in one sample (no division inside addition)
"""

import sys, json, random, argparse, math
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
# STAGE 1 — Arithmetic: Force memorization via massive repetition
# ══════════════════════════════════════════════════════════════

ADD_TEMPLATES = [
    lambda a, b, r: (f"What is {a} + {b}?",             f"{a} + {b} = {r}"),
    lambda a, b, r: (f"Calculate {a} + {b}.",            f"{a} + {b} = {r}"),
    lambda a, b, r: (f"Add {a} and {b}.",                f"{a} + {b} = {r}"),
    lambda a, b, r: (f"What does {a} plus {b} equal?",   f"{a} + {b} = {r}"),
    lambda a, b, r: (f"{a} + {b} = ?",                   f"{a} + {b} = {r}"),
]

MUL_TEMPLATES = [
    lambda a, b, r: (f"What is {a} × {b}?",              f"{a} × {b} = {r}"),
    lambda a, b, r: (f"Calculate {a} × {b}.",             f"{a} × {b} = {r}"),
    lambda a, b, r: (f"What is {a} times {b}?",           f"{a} × {b} = {r}"),
    lambda a, b, r: (f"Multiply {a} by {b}.",             f"{a} × {b} = {r}"),
    lambda a, b, r: (f"{a} × {b} = ?",                    f"{a} × {b} = {r}"),
]

SUB_TEMPLATES = [
    lambda a, b, r: (f"What is {a} - {b}?",              f"{a} - {b} = {r}"),
    lambda a, b, r: (f"Calculate {a} - {b}.",             f"{a} - {b} = {r}"),
    lambda a, b, r: (f"Subtract {b} from {a}.",           f"{a} - {b} = {r}"),
    lambda a, b, r: (f"{a} - {b} = ?",                    f"{a} - {b} = {r}"),
]

DIV_TEMPLATES = [
    lambda a, b, r: (f"What is {a} ÷ {b}?",              f"{a} ÷ {b} = {r}"),
    lambda a, b, r: (f"Calculate {a} ÷ {b}.",             f"{a} ÷ {b} = {r}"),
    lambda a, b, r: (f"Divide {a} by {b}.",               f"{a} ÷ {b} = {r}"),
    lambda a, b, r: (f"{a} ÷ {b} = ?",                    f"{a} ÷ {b} = {r}"),
]


def gen_stage1(reps: int = 30) -> list:
    """
    Generate massive repetition of every arithmetic fact.
    reps = how many times each unique (a,b) fact appears.
    """
    out = []

    # ── ALL single-digit addition: 0+0 to 9+9 (100 facts × reps × templates)
    for a in range(0, 10):
        for b in range(0, 10):
            r = a + b
            templates = ADD_TEMPLATES
            for i in range(reps):
                fn = templates[i % len(templates)]
                q, ans = fn(a, b, r)
                out.append(sample(q, ans))
            # Commutativity (if different)
            if a != b:
                for i in range(reps // 2):
                    fn = templates[i % len(templates)]
                    q, ans = fn(b, a, r)
                    out.append(sample(q, ans))

    # ── ALL single-digit multiplication: 1×1 to 9×9 (81 facts × reps × templates)
    for a in range(1, 10):
        for b in range(1, 10):
            r = a * b
            for i in range(reps):
                fn = MUL_TEMPLATES[i % len(MUL_TEMPLATES)]
                q, ans = fn(a, b, r)
                out.append(sample(q, ans))
            if a != b:
                for i in range(reps // 2):
                    fn = MUL_TEMPLATES[i % len(MUL_TEMPLATES)]
                    q, ans = fn(b, a, r)
                    out.append(sample(q, ans))

    # ── ALL single-digit subtraction: a - b where a >= b (55 facts × reps)
    for a in range(0, 10):
        for b in range(0, a + 1):
            r = a - b
            for i in range(reps // 2):
                fn = SUB_TEMPLATES[i % len(SUB_TEMPLATES)]
                q, ans = fn(a, b, r)
                out.append(sample(q, ans))

    # ── ALL single-digit division: a÷b = exact (from multiplication table)
    for b in range(1, 10):
        for q_val in range(1, 10):
            a = b * q_val
            for i in range(reps // 2):
                fn = DIV_TEMPLATES[i % len(DIV_TEMPLATES)]
                q, ans = fn(a, b, q_val)
                out.append(sample(q, ans))

    # ── Squares 1²-15² (each × reps)
    sq_templates = [
        lambda n, sq: (f"What is {n} squared?",         f"{n}² = {sq}"),
        lambda n, sq: (f"What is {n}²?",                f"{n}² = {sq}"),
        lambda n, sq: (f"Calculate {n}².",               f"{n}² = {sq}"),
        lambda n, sq: (f"What is {n} × {n}?",           f"{n} × {n} = {sq}"),
        lambda n, sq: (f"{n}² = ?",                      f"{n}² = {sq}"),
    ]
    sqrt_templates = [
        lambda n, sq: (f"What is the square root of {sq}?",   f"√{sq} = {n}"),
        lambda n, sq: (f"√{sq} = ?",                           f"√{sq} = {n}"),
        lambda n, sq: (f"Calculate √{sq}.",                    f"√{sq} = {n}"),
    ]
    for n in range(1, 16):
        sq = n * n
        for i in range(reps):
            fn = sq_templates[i % len(sq_templates)]
            q, ans = fn(n, sq)
            out.append(sample(q, ans))
        for i in range(reps):
            fn = sqrt_templates[i % len(sqrt_templates)]
            q, ans = fn(n, sq)
            out.append(sample(q, ans))

    # ── Two-digit + one-digit (NO carry) — rigid 3-line algorithm
    for _ in range(200):
        t = random.randint(1, 9)
        a2 = random.randint(0, 4)
        b  = random.randint(0, 9 - a2)
        a  = t * 10 + a2
        r  = a + b
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a2} + {b} = {a2+b}\n"
            f"Tens: {t}\n"
            f"Answer: {r}"
        ))

    # ── Two-digit + one-digit (WITH carry) — rigid 3-line algorithm
    for _ in range(200):
        t  = random.randint(1, 8)
        a2 = random.randint(5, 9)
        b  = random.randint(10 - a2, 9)
        a  = t * 10 + a2
        r  = a + b
        s  = a2 + b
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a2} + {b} = {s} → write {s%10}, carry 1\n"
            f"Tens: {t} + 1 = {t+1}\n"
            f"Answer: {r}"
        ))

    # ── Two-digit + two-digit (no carry) — rigid algorithm
    for _ in range(300):
        a1, a2 = random.randint(1,9), random.randint(0,4)
        b1, b2 = random.randint(1,9), random.randint(0, 9-a2)
        a, b = a1*10+a2, b1*10+b2
        r = a + b
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a2} + {b2} = {a2+b2}\n"
            f"Tens: {a1} + {b1} = {a1+b1}\n"
            f"Answer: {r}"
        ))

    # ── Two-digit + two-digit (WITH carry) — rigid algorithm
    for _ in range(300):
        a1, a2 = random.randint(1,8), random.randint(5,9)
        b1, b2 = random.randint(1,8), random.randint(10-a2, 9)
        a, b = a1*10+a2, b1*10+b2
        r = a + b
        s = a2 + b2
        out.append(sample(
            f"What is {a} + {b}?",
            f"Ones: {a2} + {b2} = {s} → write {s%10}, carry 1\n"
            f"Tens: {a1} + {b1} + 1 = {a1+b1+1}\n"
            f"Answer: {r}"
        ))

    # ── Two-digit × one-digit — rigid algorithm
    for _ in range(400):
        a1 = random.randint(1, 9)
        a2 = random.randint(0, 9)
        b  = random.randint(2, 9)
        a  = a1*10 + a2
        r  = a * b
        p2 = a2 * b
        d2 = p2 % 10
        c  = p2 // 10
        p1 = a1 * b + c
        ans = (
            f"Ones: {a2} × {b} = {p2}"
            + (f" → write {d2}, carry {c}" if c else "") + "\n"
            f"Tens: {a1} × {b}" + (f" + {c}" if c else "") + f" = {p1}\n"
            f"Answer: {r}"
        )
        out.append(sample(f"What is {a} × {b}?", ans))

    # ── Order of operations (multiplication before addition)
    for _ in range(200):
        a = random.randint(1, 20)
        b = random.randint(1, 9)
        c = random.randint(1, 9)
        r = a + b * c
        out.append(sample(
            f"What is {a} + {b} × {c}?",
            f"Multiplication first: {b} × {c} = {b*c}\n"
            f"Then addition: {a} + {b*c} = {r}\n"
            f"Answer: {r}"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 2 — Fractions & Decimals
# ══════════════════════════════════════════════════════════════

def gen_stage2(reps: int = 20) -> list:
    out = []

    # ── Fraction simplification — repeated with GCD steps
    pairs = [(n, d) for d in range(2, 13) for n in range(1, d) if math.gcd(n, d) > 1]
    for n, d in pairs:
        g = math.gcd(n, d)
        sn, sd = n // g, d // g
        for _ in range(reps):
            out.append(sample(
                f"Simplify {n}/{d}.",
                f"GCD({n}, {d}) = {g}\n"
                f"{n} ÷ {g} = {sn}\n"
                f"{d} ÷ {g} = {sd}\n"
                f"Answer: {sn}/{sd}"
            ))

    # ── Fraction addition — LCD method, repeated
    frac_pairs = [(n1, d1, n2, d2)
                  for d1 in [2,3,4,5,6,8]
                  for d2 in [2,3,4,5,6,8]
                  for n1 in range(1, d1)
                  for n2 in range(1, d2)]
    random.shuffle(frac_pairs)
    for n1, d1, n2, d2 in frac_pairs[:120]:
        f1, f2 = Fraction(n1, d1), Fraction(n2, d2)
        result = f1 + f2
        lcd = d1 * d2 // math.gcd(d1, d2)
        m1, m2 = lcd // d1, lcd // d2
        rn = n1*m1 + n2*m2
        g = math.gcd(rn, lcd)
        for _ in range(reps // 2):
            out.append(sample(
                f"What is {n1}/{d1} + {n2}/{d2}?",
                f"LCD({d1}, {d2}) = {lcd}\n"
                f"{n1}/{d1} = {n1*m1}/{lcd}\n"
                f"{n2}/{d2} = {n2*m2}/{lcd}\n"
                f"{n1*m1} + {n2*m2} = {rn}\n"
                f"Simplify: {rn}/{lcd} ÷ {g} = {result.numerator}/{result.denominator}\n"
                f"Answer: {result.numerator}/{result.denominator}"
            ))

    # ── Fraction × fraction
    for _ in range(150):
        n1, d1 = random.randint(1,5), random.randint(2,8)
        n2, d2 = random.randint(1,5), random.randint(2,8)
        result = Fraction(n1,d1) * Fraction(n2,d2)
        out.append(sample(
            f"What is {n1}/{d1} × {n2}/{d2}?",
            f"Numerator: {n1} × {n2} = {n1*n2}\n"
            f"Denominator: {d1} × {d2} = {d1*d2}\n"
            f"Simplify: {n1*n2}/{d1*d2} = {result.numerator}/{result.denominator}\n"
            f"Answer: {result.numerator}/{result.denominator}"
        ))

    # ── Percentage — exhaustive small table, repeated
    pcts   = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90]
    wholes = [10, 20, 40, 50, 80, 100, 120, 150, 200, 250, 300, 400, 500]
    for p in pcts:
        for w in wholes:
            r = p * w // 100
            for _ in range(3):
                out.append(sample(
                    f"What is {p}% of {w}?",
                    f"{p}% = {p}/100\n"
                    f"{p}/100 × {w} = {r}\n"
                    f"Answer: {r}"
                ))

    # ── Fraction ↔ decimal ↔ percent — exhaustive common set, many reps
    conversions = [
        (1,2,"0.5","50%"),(1,4,"0.25","25%"),(3,4,"0.75","75%"),
        (1,5,"0.2","20%"),(2,5,"0.4","40%"),(3,5,"0.6","60%"),(4,5,"0.8","80%"),
        (1,8,"0.125","12.5%"),(3,8,"0.375","37.5%"),
        (1,10,"0.1","10%"),(1,20,"0.05","5%"),(1,100,"0.01","1%"),
    ]
    for n, d, dec, pct in conversions:
        for _ in range(reps):
            out.append(sample(f"Convert {n}/{d} to a decimal.",
                              f"{n} ÷ {d} = {dec}\nAnswer: {dec}"))
            out.append(sample(f"Convert {n}/{d} to a percentage.",
                              f"{n}/{d} = {dec} = {pct}\nAnswer: {pct}"))
            out.append(sample(f"What is {dec} as a fraction?",
                              f"{dec} = {n}/{d}\nAnswer: {n}/{d}"))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 3 — Pre-Algebra
# ══════════════════════════════════════════════════════════════

def gen_stage3(reps: int = 15) -> list:
    out = []

    # ── Evaluate expressions
    for x in range(-5, 11):
        for a in range(1, 7):
            for b in range(0, 11):
                r = a * x + b
                for _ in range(reps // 5 + 1):
                    out.append(sample(
                        f"Evaluate {a}x + {b} when x = {x}.",
                        f"Substitute x = {x}:\n"
                        f"{a} × {x} = {a*x}\n"
                        f"{a*x} + {b} = {r}\n"
                        f"Answer: {r}"
                    ))

    # ── One-step equations: ax = b
    for x in range(1, 21):
        for a in range(2, 13):
            b = a * x
            for _ in range(reps // 3):
                out.append(sample(
                    f"Solve for x: {a}x = {b}",
                    f"Divide both sides by {a}:\n"
                    f"{b} ÷ {a} = {x}\n"
                    f"Answer: x = {x}"
                ))

    # ── Two-step equations: ax + b = c
    for x in range(1, 16):
        for a in range(2, 9):
            for b in range(1, 11):
                c = a * x + b
                for _ in range(reps // 5):
                    out.append(sample(
                        f"Solve for x: {a}x + {b} = {c}",
                        f"Subtract {b} from both sides: {c} - {b} = {c-b}\n"
                        f"Divide both sides by {a}: {c-b} ÷ {a} = {x}\n"
                        f"Answer: x = {x}"
                    ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 4 — Algebra
# ══════════════════════════════════════════════════════════════

def gen_stage4(reps: int = 10) -> list:
    out = []

    # ── Slope from two points
    for _ in range(200):
        x1, y1 = random.randint(-5, 4), random.randint(-5, 5)
        dx = random.randint(1, 5)
        m  = random.randint(-4, 4) or 1
        x2, y2 = x1 + dx, y1 + m * dx
        for _ in range(reps // 2):
            out.append(sample(
                f"Find the slope of the line through ({x1}, {y1}) and ({x2}, {y2}).",
                f"m = (y₂ - y₁) / (x₂ - x₁)\n"
                f"y₂ - y₁ = {y2} - {y1} = {y2-y1}\n"
                f"x₂ - x₁ = {x2} - {x1} = {x2-x1}\n"
                f"m = {y2-y1} / {x2-x1} = {m}\n"
                f"Answer: {m}"
            ))

    # ── Quadratic factoring
    for r1 in range(-6, 7):
        for r2 in range(-6, 7):
            if r1 == 0 or r2 == 0:
                continue
            b = -(r1 + r2)
            c = r1 * r2
            b_str = f"{b:+d}x" if b != 0 else ""
            c_str = f"{c:+d}"  if c != 0 else ""
            for _ in range(reps // 3):
                out.append(sample(
                    f"Solve: x²{b_str}{c_str} = 0",
                    f"Find numbers that multiply to {c} and add to {b}:\n"
                    f"{-r1} × {-r2} = {c}, {-r1} + {-r2} = {b} ✓\n"
                    f"(x {-r1:+d})(x {-r2:+d}) = 0\n"
                    f"x = {r1} or x = {r2}\n"
                    f"Answer: x = {r1} and x = {r2}"
                ))

    # ── Logarithm facts — exhaustive + repeated
    log_facts = [
        (2,1,2),(2,2,4),(2,3,8),(2,4,16),(2,5,32),(2,6,64),
        (3,1,3),(3,2,9),(3,3,27),(3,4,81),
        (5,1,5),(5,2,25),(5,3,125),
        (10,1,10),(10,2,100),(10,3,1000),
    ]
    for base, exp, val in log_facts:
        for _ in range(reps * 2):
            out.append(sample(
                f"What is log_{base}({val})?",
                f"{base}^{exp} = {val}\n"
                f"Answer: {exp}"
            ))
            out.append(sample(
                f"What is {base}^{exp}?",
                f"{' × '.join([str(base)]*exp)} = {val}\n"
                f"Answer: {val}"
            ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 5 — Geometry
# ══════════════════════════════════════════════════════════════

def gen_stage5(reps: int = 10) -> list:
    out = []

    # ── Rectangle area — exhaustive small dims
    for l in range(1, 21):
        for w in range(1, 21):
            for _ in range(reps // 5 + 1):
                out.append(sample(
                    f"Find the area of a rectangle with length {l} and width {w}.",
                    f"Area = length × width\n"
                    f"{l} × {w} = {l*w}\n"
                    f"Answer: {l*w} square units"
                ))

    # ── Circle area + circumference (r = 1-15, repeated)
    for r in range(1, 16):
        area = round(math.pi * r**2, 2)
        circ = round(2 * math.pi * r, 2)
        for _ in range(reps):
            out.append(sample(
                f"Find the area of a circle with radius {r}.",
                f"Area = π × r²\n"
                f"= π × {r}² = π × {r*r}\n"
                f"≈ {area}\n"
                f"Answer: {area} square units"
            ))
            out.append(sample(
                f"Find the circumference of a circle with radius {r}.",
                f"Circumference = 2 × π × r\n"
                f"= 2 × π × {r}\n"
                f"≈ {circ}\n"
                f"Answer: {circ} units"
            ))

    # ── Pythagorean triples — exhaustive + repeated
    triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(6,8,10),(9,12,15),
               (12,16,20),(15,20,25),(20,21,29),(9,40,41),(10,24,26)]
    for a, b, c in triples:
        for _ in range(reps * 2):
            out.append(sample(
                f"A right triangle has legs {a} and {b}. Find the hypotenuse.",
                f"c² = a² + b²\n"
                f"{a}² + {b}² = {a**2} + {b**2} = {a**2+b**2}\n"
                f"√{a**2+b**2} = {c}\n"
                f"Answer: {c}"
            ))

    # ── Supplementary/complementary — exhaustive 10° steps
    for deg in range(10, 90, 5):
        comp, supp = 90-deg, 180-deg
        for _ in range(reps):
            out.append(sample(
                f"If one angle is {deg}°, what is its complement?",
                f"Complementary angles add to 90°.\n"
                f"90 - {deg} = {comp}\n"
                f"Answer: {comp}°"
            ))
            out.append(sample(
                f"If one angle is {deg}°, what is its supplement?",
                f"Supplementary angles add to 180°.\n"
                f"180 - {deg} = {supp}\n"
                f"Answer: {supp}°"
            ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 6 — Statistics & Probability
# ══════════════════════════════════════════════════════════════

def gen_stage6(reps: int = 8) -> list:
    out = []

    # ── Mean of small lists
    for _ in range(300):
        nums = [random.randint(1, 30) for _ in range(random.randint(3, 6))]
        total = sum(nums)
        mean = round(total / len(nums), 2)
        for _ in range(reps // 2):
            out.append(sample(
                f"Find the mean of: {', '.join(map(str, nums))}",
                f"Sum: {' + '.join(map(str, nums))} = {total}\n"
                f"Count: {len(nums)}\n"
                f"{total} ÷ {len(nums)} = {mean}\n"
                f"Answer: {mean}"
            ))

    # ── Median
    for _ in range(200):
        nums = [random.randint(1, 30) for _ in range(random.randint(3, 7))]
        s = sorted(nums)
        n = len(s)
        med = s[n//2] if n%2==1 else (s[n//2-1]+s[n//2])/2
        out.append(sample(
            f"Find the median of: {', '.join(map(str, nums))}",
            f"Sorted: {', '.join(map(str, s))}\n"
            f"Middle: {med}\n"
            f"Answer: {med}"
        ))

    # ── Probability — exhaustive small cases
    for total in range(4, 13):
        for fav in range(1, total):
            f = Fraction(fav, total)
            for _ in range(reps // 2):
                out.append(sample(
                    f"A bag has {total} items. {fav} are red. Probability of red?",
                    f"P = {fav}/{total}"
                    + (f" = {f.numerator}/{f.denominator}" if f.numerator != fav else "") + "\n"
                    f"Answer: {f.numerator}/{f.denominator}"
                ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 7 — Word Problems
# ══════════════════════════════════════════════════════════════

def gen_stage7(reps: int = 8) -> list:
    out = []

    # ── Distance/speed/time — systematic grid
    for speed in range(10, 121, 10):
        for time in range(1, 9):
            dist = speed * time
            for _ in range(reps // 3):
                out.append(sample(
                    f"A car travels at {speed} km/h for {time} hours. How far?",
                    f"distance = speed × time\n"
                    f"= {speed} × {time} = {dist}\n"
                    f"Answer: {dist} km"
                ))
                out.append(sample(
                    f"A car travels {dist} km at {speed} km/h. How long?",
                    f"time = distance ÷ speed\n"
                    f"= {dist} ÷ {speed} = {time}\n"
                    f"Answer: {time} hour{'s' if time!=1 else ''}"
                ))

    # ── Simple interest — systematic grid
    for P in [100, 200, 500, 1000, 2000, 5000]:
        for r in [5, 8, 10, 12, 15]:
            for t in range(1, 6):
                SI = P * r * t // 100
                for _ in range(reps // 4):
                    out.append(sample(
                        f"Simple interest on ${P} at {r}% for {t} year{'s' if t>1 else ''}?",
                        f"SI = P × r × t / 100\n"
                        f"= {P} × {r} × {t} / 100\n"
                        f"= {P*r*t} / 100\n"
                        f"Answer: ${SI}"
                    ))

    # ── Ratio problems
    for a in range(1, 6):
        for b in range(1, 6):
            for mult in range(2, 9):
                total = (a + b) * mult
                pa = a * mult
                pb = b * mult
                out.append(sample(
                    f"Ratio {a}:{b}, total {total}. Find each part.",
                    f"Total parts = {a} + {b} = {a+b}\n"
                    f"One part = {total} ÷ {a+b} = {mult}\n"
                    f"First part = {a} × {mult} = {pa}\n"
                    f"Second part = {b} × {mult} = {pb}\n"
                    f"Answer: {pa} and {pb}"
                ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 8 — Calculus
# ══════════════════════════════════════════════════════════════

def gen_stage8(reps: int = 15) -> list:
    out = []

    # ── Derivatives — power rule: exhaustive n=1..8, a=1..6, repeated
    for n in range(1, 9):
        for a in range(1, 7):
            coeff = a * n
            e = n - 1
            deriv = (str(coeff) if e == 0
                     else (f"{coeff}x" if e == 1
                           else f"{coeff}x^{e}"))
            for _ in range(reps):
                out.append(sample(
                    f"Find the derivative of f(x) = {a}x^{n}.",
                    f"Power rule: bring down {n}, reduce exponent by 1\n"
                    f"{a} × {n} = {coeff}\n"
                    f"Exponent: {n} - 1 = {e}\n"
                    f"Answer: f'(x) = {deriv}"
                ))

    # ── Integrals — power rule: exhaustive n=1..7, a=1..5, repeated
    for n in range(1, 8):
        for a in range(1, 6):
            new_e = n + 1
            coeff = Fraction(a, new_e)
            c_str = f"{a}/{new_e}" if coeff.denominator != 1 else str(a)
            for _ in range(reps):
                out.append(sample(
                    f"Find ∫{a}x^{n} dx.",
                    f"Power rule: raise exponent by 1, divide by new exponent\n"
                    f"Exponent: {n} + 1 = {new_e}\n"
                    f"Coefficient: {a} ÷ {new_e} = {c_str}\n"
                    f"Answer: {c_str}x^{new_e} + C"
                ))

    # ── Limits — repeated
    limits = [
        ("lim(x→0) of x", "Direct substitution: 0\nAnswer: 0"),
        ("lim(x→2) of x²", "Direct substitution: 2² = 4\nAnswer: 4"),
        ("lim(x→3) of (x²-9)/(x-3)",
         "Factor: x²-9 = (x+3)(x-3)\nCancel (x-3): lim = x+3 at x=3 = 6\nAnswer: 6"),
        ("lim(x→∞) of 1/x", "As x→∞, 1/x→0\nAnswer: 0"),
        ("lim(x→1) of (x²-1)/(x-1)",
         "Factor: x²-1 = (x+1)(x-1)\nCancel (x-1): lim = x+1 at x=1 = 2\nAnswer: 2"),
        ("lim(x→0) of sin(x)/x",
         "Fundamental limit.\nAnswer: 1"),
        ("lim(x→∞) of 5/x", "As x→∞, 5/x→0\nAnswer: 0"),
        ("lim(x→4) of x²", "Direct substitution: 4² = 16\nAnswer: 16"),
    ]
    for q_text, a_text in limits:
        for _ in range(reps):
            out.append(sample(f"Find the {q_text}.", a_text))

    # ── Conceptual — definitions (repeated)
    concepts = [
        ("What is a derivative?",
         "A derivative is the instantaneous rate of change of a function.\n"
         "f'(x) = lim(h→0) [f(x+h) - f(x)] / h\n"
         "It equals the slope of the tangent line at x."),
        ("What is an integral?",
         "An integral is the area under a curve.\n"
         "∫f(x)dx gives antiderivatives F(x) + C.\n"
         "∫ₐᵇf(x)dx = F(b) - F(a)."),
        ("What does the power rule for derivatives say?",
         "d/dx[xⁿ] = n·xⁿ⁻¹\n"
         "Bring down the exponent, reduce it by 1.\n"
         "Example: d/dx[x³] = 3x²"),
        ("What does the power rule for integration say?",
         "∫xⁿ dx = xⁿ⁺¹/(n+1) + C\n"
         "Raise the exponent by 1, divide by the new exponent.\n"
         "Example: ∫x² dx = x³/3 + C"),
    ]
    for q_text, a_text in concepts:
        for _ in range(reps * 2):
            out.append(sample(q_text, a_text))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

STAGES = {
    1: ("stage1_arithmetic",   gen_stage1),
    2: ("stage2_fractions",    gen_stage2),
    3: ("stage3_prealgebra",   gen_stage3),
    4: ("stage4_algebra",      gen_stage4),
    5: ("stage5_geometry",     gen_stage5),
    6: ("stage6_statistics",   gen_stage6),
    7: ("stage7_wordproblems", gen_stage7),
    8: ("stage8_calculus",     gen_stage8),
}


def write_jsonl(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data):>6} samples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", type=int, default=list(STAGES.keys()))
    parser.add_argument("--stage",  type=int, default=None)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    to_run  = [args.stage] if args.stage else args.stages
    all_samples = []

    for num in to_run:
        if num not in STAGES:
            print(f"Unknown stage {num}"); continue
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
        print(f"\nTotal: {len(all_samples):,} samples across {len(to_run)} stages")


if __name__ == "__main__":
    main()
