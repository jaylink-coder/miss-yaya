"""
generate_math_data.py — v4 (Tool-Use + Reasoning Edition)
===========================================================
Core insight: don't make the 4.8M model STORE arithmetic in its weights.
Teach it to USE the calculator tool: <|calc|>EXPR<|/calc|>=RESULT

The model learns:
  1. When to call the calculator (any arithmetic)
  2. How to form the expression (copy numbers from question)
  3. How to reason step-by-step (<|think|> blocks for complex problems)

At inference, ToolAugmentedGenerator intercepts <|calc|>EXPR<|/calc|>
and evaluates it in Python — giving exact answers every time.

Data format:
  Simple:   <|calc|>6*7<|/calc|>=42\nAnswer: 42
  Stepped:  Step 1: <|calc|>23+48<|/calc|>=71\nAnswer: 71
  Reasoned: <|think|>reasoning...</|think|>\nStep 1: <|calc|>...<|/calc|>=X\nAnswer: X

Usage:
    python scripts/generate_math_data.py
    python scripts/generate_math_data.py --stage 1 --preview 3
"""

import sys, json, random, argparse, math
from pathlib import Path
from fractions import Fraction

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

SYSTEM_MSG = (
    "You are Yaya, a helpful AI assistant. "
    "When you need to compute, use <|calc|>EXPRESSION<|/calc|>. "
    "The result appears as =RESULT. Show your reasoning step by step."
)
OUTPUT_DIR = Path("data/sft/math")

CALC_OPEN  = "<|calc|>"
CALC_CLOSE = "<|/calc|>"
THINK_OPEN  = "<|think|>"
THINK_CLOSE = "<|/think|>"


def calc(expr: str, result) -> str:
    """Format a single calculator call."""
    return f"{CALC_OPEN}{expr}{CALC_CLOSE}={result}"


def think(text: str) -> str:
    return f"{THINK_OPEN}{text}{THINK_CLOSE}"


def sample(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def shuffled(lst):
    random.shuffle(lst)
    return lst


# ══════════════════════════════════════════════════════════════
# STAGE 1 — Arithmetic via calculator tool
# Teach: emit <|calc|>EXPR<|/calc|>=RESULT for any arithmetic
# ══════════════════════════════════════════════════════════════

def gen_stage1(reps: int = 20) -> list:
    out = []

    # ── Single-digit addition — all 100 facts × reps
    for a in range(0, 10):
        for b in range(0, 10):
            r = a + b
            for i in range(reps):
                q_forms = [
                    f"What is {a} + {b}?",
                    f"Calculate {a} + {b}.",
                    f"{a} + {b} = ?",
                    f"Add {a} and {b}.",
                ]
                q = q_forms[i % len(q_forms)]
                out.append(sample(q, f"{calc(f'{a}+{b}', r)}\nAnswer: {r}"))

    # ── Single-digit multiplication — all 81 facts × reps
    for a in range(1, 10):
        for b in range(1, 10):
            r = a * b
            for i in range(reps):
                q_forms = [
                    f"What is {a} × {b}?",
                    f"Calculate {a} × {b}.",
                    f"{a} × {b} = ?",
                    f"Multiply {a} by {b}.",
                    f"What is {a} times {b}?",
                ]
                q = q_forms[i % len(q_forms)]
                out.append(sample(q, f"{calc(f'{a}*{b}', r)}\nAnswer: {r}"))

    # ── Single-digit subtraction — all non-negative facts × reps
    for a in range(0, 10):
        for b in range(0, a + 1):
            r = a - b
            for i in range(reps // 2):
                q_forms = [
                    f"What is {a} - {b}?",
                    f"Subtract {b} from {a}.",
                    f"{a} - {b} = ?",
                ]
                q = q_forms[i % len(q_forms)]
                out.append(sample(q, f"{calc(f'{a}-{b}', r)}\nAnswer: {r}"))

    # ── Single-digit division — all exact facts × reps
    for b in range(1, 10):
        for q_val in range(1, 10):
            a = b * q_val
            for i in range(reps // 2):
                q_forms = [
                    f"What is {a} ÷ {b}?",
                    f"Divide {a} by {b}.",
                    f"{a} ÷ {b} = ?",
                ]
                q = q_forms[i % len(q_forms)]
                out.append(sample(q, f"{calc(f'{a}/{b}', q_val)}\nAnswer: {q_val}"))

    # ── Two-digit + one-digit (with and without carry) × reps
    for _ in range(400 * reps // 5):
        a = random.randint(10, 99)
        b = random.randint(1, 9)
        r = a + b
        out.append(sample(f"What is {a} + {b}?",
                          f"{calc(f'{a}+{b}', r)}\nAnswer: {r}"))

    # ── Two-digit + two-digit × reps
    for _ in range(400 * reps // 5):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        r = a + b
        out.append(sample(f"What is {a} + {b}?",
                          f"{calc(f'{a}+{b}', r)}\nAnswer: {r}"))

    # ── Two-digit × one-digit × reps
    for _ in range(400 * reps // 5):
        a = random.randint(10, 99)
        b = random.randint(2, 9)
        r = a * b
        out.append(sample(f"What is {a} × {b}?",
                          f"{calc(f'{a}*{b}', r)}\nAnswer: {r}"))

    # ── Two-digit × two-digit
    for _ in range(200 * reps // 5):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        r = a * b
        out.append(sample(f"What is {a} × {b}?",
                          f"{calc(f'{a}*{b}', r)}\nAnswer: {r}"))

    # ── Squares and square roots (1-20) × reps
    for n in range(1, 21):
        sq = n * n
        for i in range(reps):
            q_forms = [
                f"What is {n} squared?",
                f"What is {n}²?",
                f"Calculate {n}².",
            ]
            out.append(sample(q_forms[i % len(q_forms)],
                               f"{calc(f'{n}**2', sq)}\nAnswer: {sq}"))
            out.append(sample(f"What is the square root of {sq}?",
                               f"{calc(f'sqrt({sq})', n)}\nAnswer: {n}"))

    # ── Order of operations — with reasoning
    for _ in range(300):
        a = random.randint(1, 30)
        b = random.randint(1, 9)
        c = random.randint(1, 9)
        r = a + b * c
        out.append(sample(
            f"What is {a} + {b} × {c}?",
            f"{think('Multiplication before addition: do the × first.')}\n"
            f"Step 1: {calc(f'{b}*{c}', b*c)}\n"
            f"Step 2: {calc(f'{a}+{b*c}', r)}\n"
            f"Answer: {r}"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 2 — Fractions & Decimals (tool-assisted)
# ══════════════════════════════════════════════════════════════

def gen_stage2(reps: int = 10) -> list:
    out = []

    # ── Fraction simplification
    pairs = [(n, d) for d in range(2, 16) for n in range(1, d)
             if math.gcd(n, d) > 1]
    for n, d in random.choices(pairs, k=150):
        g = math.gcd(n, d)
        sn, sd = n // g, d // g
        for _ in range(reps):
            out.append(sample(
                f"Simplify {n}/{d}.",
                f"{think(f'Find GCD({n},{d})={g}. Divide top and bottom.')}\n"
                f"GCD: {calc(f'__import__(\"math\").gcd({n},{d})', g)}\n"
                f"{n} ÷ {g} = {sn},  {d} ÷ {g} = {sd}\n"
                f"Answer: {sn}/{sd}"
            ))

    # ── Fraction addition
    frac_pairs = [(n1,d1,n2,d2)
                  for d1 in [2,3,4,5,6,8,10]
                  for d2 in [2,3,4,5,6,8,10]
                  for n1 in range(1,d1)
                  for n2 in range(1,d2)]
    random.shuffle(frac_pairs)
    for n1,d1,n2,d2 in frac_pairs[:200]:
        f1, f2 = Fraction(n1,d1), Fraction(n2,d2)
        result = f1 + f2
        lcd = d1*d2 // math.gcd(d1,d2)
        m1, m2 = lcd//d1, lcd//d2
        rn = n1*m1 + n2*m2
        g = math.gcd(rn, lcd)
        for _ in range(reps // 2):
            out.append(sample(
                f"What is {n1}/{d1} + {n2}/{d2}?",
                f"{think(f'Find LCD({d1},{d2})={lcd}, convert fractions, then add.')}\n"
                f"LCD = {lcd}\n"
                f"{n1}/{d1} = {n1*m1}/{lcd}\n"
                f"{n2}/{d2} = {n2*m2}/{lcd}\n"
                f"Add: {calc(f'{n1*m1}+{n2*m2}', rn)}\n"
                f"Simplify: {rn}/{lcd} = {result.numerator}/{result.denominator}\n"
                f"Answer: {result.numerator}/{result.denominator}"
            ))

    # ── Percentage calculations
    pcts   = [5,10,15,20,25,30,40,50,60,75,80,90]
    wholes = [10,20,40,50,80,100,120,150,200,250,300,400,500,1000]
    for p in pcts:
        for w in wholes:
            r = round(p * w / 100, 2)
            r_int = int(r) if r == int(r) else r
            for _ in range(reps // 3):
                out.append(sample(
                    f"What is {p}% of {w}?",
                    f"{think(f'{p}% means {p}/100. Multiply.')}\n"
                    f"{calc(f'{p}/100*{w}', r_int)}\n"
                    f"Answer: {r_int}"
                ))

    # ── Decimal operations
    for _ in range(200):
        a = round(random.uniform(1.1, 49.9), 1)
        b = round(random.uniform(1.1, 49.9), 1)
        r = round(a + b, 1)
        out.append(sample(f"What is {a} + {b}?",
                          f"{calc(f'{a}+{b}', r)}\nAnswer: {r}"))

    # ── Fraction ↔ decimal ↔ percent conversions
    conversions = [
        (1,2,"0.5","50%"),(1,4,"0.25","25%"),(3,4,"0.75","75%"),
        (1,5,"0.2","20%"),(2,5,"0.4","40%"),(3,5,"0.6","60%"),(4,5,"0.8","80%"),
        (1,8,"0.125","12.5%"),(3,8,"0.375","37.5%"),
        (1,10,"0.1","10%"),(1,20,"0.05","5%"),
    ]
    for n, d, dec, pct in conversions:
        for _ in range(reps * 2):
            r = round(n/d, 4)
            out.append(sample(f"Convert {n}/{d} to a decimal.",
                               f"{calc(f'{n}/{d}', r)}\nAnswer: {dec}"))
            out.append(sample(f"What is {dec} as a percentage?",
                               f"{calc(f'{dec}*100', float(pct.replace('%','')))}\\nAnswer: {pct}"))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 3 — Pre-Algebra (reason + tool)
# ══════════════════════════════════════════════════════════════

def gen_stage3(reps: int = 8) -> list:
    out = []

    # ── Expression evaluation
    for x in range(-8, 11):
        for a in range(1, 7):
            for b in range(0, 11):
                r = a * x + b
                for _ in range(reps // 5 + 1):
                    out.append(sample(
                        f"Evaluate {a}x + {b} when x = {x}.",
                        f"{think(f'Substitute x={x}, then compute.')}\n"
                        f"Step 1: {calc(f'{a}*({x})', a*x)}\n"
                        f"Step 2: {calc(f'{a*x}+{b}', r)}\n"
                        f"Answer: {r}"
                    ))

    # ── One-step equations
    for x in range(1, 21):
        for a in range(2, 13):
            b_val = a * x
            for _ in range(reps // 3):
                out.append(sample(
                    f"Solve for x: {a}x = {b_val}",
                    f"{think(f'Divide both sides by {a}.')}\n"
                    f"{calc(f'{b_val}/{a}', x)}\n"
                    f"Answer: x = {x}"
                ))

    # ── Two-step equations
    for x in range(1, 13):
        for a in range(2, 8):
            for b in range(1, 10):
                c = a * x + b
                for _ in range(reps // 5):
                    out.append(sample(
                        f"Solve for x: {a}x + {b} = {c}",
                        f"{think(f'Subtract {b}, then divide by {a}.')}\n"
                        f"Step 1: {calc(f'{c}-{b}', c-b)}\n"
                        f"Step 2: {calc(f'{c-b}/{a}', x)}\n"
                        f"Answer: x = {x}"
                    ))

    # ── Inequalities
    for x_b in range(1, 16):
        for a in range(2, 7):
            b_val = a * x_b
            for sign in ["<", ">", "≤", "≥"]:
                out.append(sample(
                    f"Solve: {a}x {sign} {b_val}",
                    f"{think(f'Divide both sides by {a} (positive, keep direction).')}\n"
                    f"{calc(f'{b_val}/{a}', x_b)}\n"
                    f"Answer: x {sign} {x_b}"
                ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 4 — Algebra (reason + tool)
# ══════════════════════════════════════════════════════════════

def gen_stage4(reps: int = 6) -> list:
    out = []

    # ── Slope
    for _ in range(200):
        x1, y1 = random.randint(-5, 4), random.randint(-5, 5)
        dx = random.randint(1, 5)
        m  = random.randint(-4, 4) or 1
        x2, y2 = x1 + dx, y1 + m * dx
        for _ in range(reps // 2):
            out.append(sample(
                f"Find the slope through ({x1},{y1}) and ({x2},{y2}).",
                f"{think('m = (y2-y1)/(x2-x1)')}\n"
                f"Numerator: {calc(f'{y2}-({y1})', y2-y1)}\n"
                f"Denominator: {calc(f'{x2}-({x1})', x2-x1)}\n"
                f"Slope: {calc(f'({y2-y1})/({x2-x1})', m)}\n"
                f"Answer: {m}"
            ))

    # ── Quadratic factoring
    for r1 in range(-5, 6):
        for r2 in range(-5, 6):
            if r1 == 0 or r2 == 0:
                continue
            bv = -(r1+r2)
            cv = r1*r2
            b_str = f"{bv:+d}x" if bv != 0 else ""
            c_str = f"{cv:+d}"  if cv != 0 else ""
            for _ in range(reps // 3):
                out.append(sample(
                    f"Solve: x²{b_str}{c_str} = 0",
                    f"{think(f'Need two numbers: product={cv}, sum={bv}. That is {-r1} and {-r2}.')}\n"
                    f"Verify product: {calc(f'({-r1})*({-r2})', cv)}\n"
                    f"Verify sum: {calc(f'({-r1})+({-r2})', bv)}\n"
                    f"Factors: (x {-r1:+d})(x {-r2:+d}) = 0\n"
                    f"Answer: x = {r1} and x = {r2}"
                ))

    # ── Logarithm facts
    log_facts = [
        (2,1,2),(2,2,4),(2,3,8),(2,4,16),(2,5,32),(2,6,64),
        (3,1,3),(3,2,9),(3,3,27),(3,4,81),
        (5,1,5),(5,2,25),(5,3,125),
        (10,1,10),(10,2,100),(10,3,1000),
    ]
    for base, exp, val in log_facts:
        for _ in range(reps * 3):
            out.append(sample(
                f"What is log_{base}({val})?",
                f"{think(f'Ask: {base} to what power equals {val}?')}\n"
                f"{base}^{exp} = {calc(f'{base}**{exp}', val)}\n"
                f"Answer: {exp}"
            ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 5 — Geometry (formula → calc)
# ══════════════════════════════════════════════════════════════

def gen_stage5(reps: int = 6) -> list:
    out = []

    # ── Rectangle area
    for l in range(1, 26):
        for w in range(1, 26):
            for _ in range(reps // 5 + 1):
                out.append(sample(
                    f"Area of rectangle: length={l}, width={w}.",
                    f"Area = length × width\n"
                    f"{calc(f'{l}*{w}', l*w)}\n"
                    f"Answer: {l*w} square units"
                ))

    # ── Circle
    for r_val in range(1, 21):
        area = round(math.pi * r_val**2, 2)
        circ = round(2 * math.pi * r_val, 2)
        for _ in range(reps):
            out.append(sample(
                f"Area of a circle with radius {r_val}.",
                f"Area = π × r²\n"
                f"{calc(f'pi*{r_val}**2', area)}\n"
                f"Answer: ≈{area} square units"
            ))
            out.append(sample(
                f"Circumference of a circle with radius {r_val}.",
                f"C = 2πr\n"
                f"{calc(f'2*pi*{r_val}', circ)}\n"
                f"Answer: ≈{circ} units"
            ))

    # ── Pythagorean theorem
    triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(6,8,10),(9,12,15),
               (12,16,20),(15,20,25),(20,21,29),(9,40,41),(10,24,26)]
    for a, b, c in triples:
        for _ in range(reps * 2):
            out.append(sample(
                f"Right triangle legs {a} and {b}. Find hypotenuse.",
                f"c² = a² + b²\n"
                f"{calc(f'{a}**2+{b}**2', a**2+b**2)}\n"
                f"c = {calc(f'sqrt({a**2+b**2})', c)}\n"
                f"Answer: {c}"
            ))

    # ── Angles
    for deg in range(5, 90, 5):
        comp, supp = 90-deg, 180-deg
        for _ in range(reps):
            out.append(sample(
                f"One angle is {deg}°. Find its complement.",
                f"Complement = 90 - angle\n"
                f"{calc(f'90-{deg}', comp)}\n"
                f"Answer: {comp}°"
            ))
            out.append(sample(
                f"One angle is {deg}°. Find its supplement.",
                f"Supplement = 180 - angle\n"
                f"{calc(f'180-{deg}', supp)}\n"
                f"Answer: {supp}°"
            ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 6 — Statistics & Probability
# ══════════════════════════════════════════════════════════════

def gen_stage6(reps: int = 5) -> list:
    out = []

    # ── Mean
    for _ in range(400):
        nums = [random.randint(1, 30) for _ in range(random.randint(3, 7))]
        total = sum(nums)
        mean  = round(total / len(nums), 2)
        for _ in range(reps):
            sum_expr = "+".join(map(str, nums))
            out.append(sample(
                f"Find the mean of: {', '.join(map(str, nums))}",
                f"{think('Sum all values, then divide by count.')}\n"
                f"Sum: {calc(sum_expr, total)}\n"
                f"Mean: {calc(f'{total}/{len(nums)}', mean)}\n"
                f"Answer: {mean}"
            ))

    # ── Median
    for _ in range(300):
        nums = [random.randint(1, 30) for _ in range(random.randint(3, 7))]
        s = sorted(nums)
        n = len(s)
        med = s[n//2] if n%2==1 else (s[n//2-1]+s[n//2])/2
        out.append(sample(
            f"Find the median of: {', '.join(map(str, nums))}",
            f"{think('Sort the values, find the middle.')}\n"
            f"Sorted: {', '.join(map(str, s))}\n"
            f"Middle value: {med}\n"
            f"Answer: {med}"
        ))

    # ── Probability
    for total in range(4, 13):
        for fav in range(1, total):
            f = Fraction(fav, total)
            pct = round(fav/total*100, 1)
            for _ in range(reps):
                out.append(sample(
                    f"Bag has {total} items, {fav} are red. Probability of red?",
                    f"P = favorable/total\n"
                    f"{calc(f'{fav}/{total}', round(fav/total,4))}\n"
                    f"Answer: {f.numerator}/{f.denominator} ≈ {pct}%"
                ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 7 — Word Problems (identify → equation → compute)
# ══════════════════════════════════════════════════════════════

def gen_stage7(reps: int = 5) -> list:
    out = []

    # ── Distance/speed/time — full systematic grid
    for speed in range(10, 121, 10):
        for time in range(1, 9):
            dist = speed * time
            for _ in range(reps):
                out.append(sample(
                    f"A car travels at {speed} km/h for {time} hours. How far?",
                    f"{think('distance = speed × time')}\n"
                    f"{calc(f'{speed}*{time}', dist)}\n"
                    f"Answer: {dist} km"
                ))
                out.append(sample(
                    f"A car travels {dist} km at {speed} km/h. How long?",
                    f"{think('time = distance ÷ speed')}\n"
                    f"{calc(f'{dist}/{speed}', time)}\n"
                    f"Answer: {time} hour{'s' if time!=1 else ''}"
                ))

    # ── Simple interest
    for P in [100,200,500,1000,2000,5000]:
        for r_pct in [5,8,10,12,15]:
            for t in range(1, 6):
                SI = P * r_pct * t // 100
                total_amt = P + SI
                for _ in range(reps):
                    out.append(sample(
                        f"Simple interest on ${P} at {r_pct}% for {t} year{'s' if t>1 else ''}?",
                        f"{think('SI = P × r × t / 100')}\n"
                        f"SI = {calc(f'{P}*{r_pct}*{t}/100', SI)}\n"
                        f"Total = {calc(f'{P}+{SI}', total_amt)}\n"
                        f"Answer: SI = ${SI}, Total = ${total_amt}"
                    ))

    # ── Ratio problems — systematic grid
    for a in range(1, 6):
        for b in range(1, 6):
            for mult in range(2, 8):
                total = (a+b)*mult
                pa = a*mult
                pb = b*mult
                out.append(sample(
                    f"Ratio {a}:{b}, total {total}. Find each part.",
                    f"{think(f'Total parts = {a}+{b} = {a+b}. One part = {total}/{a+b}.')}\n"
                    f"One part: {calc(f'{total}/{a+b}', mult)}\n"
                    f"First part: {calc(f'{a}*{mult}', pa)}\n"
                    f"Second part: {calc(f'{b}*{mult}', pb)}\n"
                    f"Answer: {pa} and {pb}"
                ))

    # ── Percentage increase/decrease
    for _ in range(200):
        original = random.choice([50,100,200,400,500,1000])
        pct = random.choice([10,15,20,25,30,50])
        increase = original * pct // 100
        new_val = original + increase
        out.append(sample(
            f"A price of ${original} increased by {pct}%. What is the new price?",
            f"{think(f'Increase = {pct}% of {original}. Add to original.')}\n"
            f"Increase: {calc(f'{pct}/100*{original}', increase)}\n"
            f"New price: {calc(f'{original}+{increase}', new_val)}\n"
            f"Answer: ${new_val}"
        ))

    return shuffled(out)


# ══════════════════════════════════════════════════════════════
# STAGE 8 — Calculus (rule → apply → verify with tool)
# ══════════════════════════════════════════════════════════════

def gen_stage8(reps: int = 12) -> list:
    out = []

    # ── Derivatives — power rule: exhaustive n=1..8, a=1..6
    for n in range(1, 9):
        for a in range(1, 7):
            coeff = a * n
            e = n - 1
            deriv = (str(coeff) if e == 0
                     else (f"{coeff}x" if e == 1
                           else f"{coeff}x^{e}"))
            for _ in range(reps):
                out.append(sample(
                    f"Differentiate f(x) = {a}x^{n}.",
                    f"{think(f'Power rule: d/dx[xⁿ]=n·xⁿ⁻¹. Multiply {a} by {n}, reduce exponent.')}\n"
                    f"New coefficient: {calc(f'{a}*{n}', coeff)}\n"
                    f"New exponent: {calc(f'{n}-1', e)}\n"
                    f"Answer: f'(x) = {deriv}"
                ))

    # ── Integrals — power rule
    for n in range(1, 8):
        for a in range(1, 6):
            new_e = n + 1
            coeff = Fraction(a, new_e)
            c_str = f"{a}/{new_e}" if coeff.denominator != 1 else str(a)
            for _ in range(reps):
                out.append(sample(
                    f"Find ∫{a}x^{n} dx.",
                    f"{think(f'Power rule: ∫xⁿ dx = xⁿ⁺¹/(n+1)+C. Raise exponent, divide.')}\n"
                    f"New exponent: {calc(f'{n}+1', new_e)}\n"
                    f"Coefficient: {calc(f'{a}/{new_e}', round(a/new_e,4))}\n"
                    f"Answer: {c_str}x^{new_e} + C"
                ))

    # ── Limits — patterns
    limits = [
        ("lim(x→2) of x²",           "Direct substitution",   "2**2",   4),
        ("lim(x→3) of (x²-9)/(x-3)", "Factor x²-9=(x+3)(x-3), cancel (x-3), get x+3", "3+3", 6),
        ("lim(x→∞) of 1/x",          "As x→∞, numerator fixed, denominator→∞, so →0", "1/1e10", 0),
        ("lim(x→1) of (x²-1)/(x-1)", "Factor x²-1=(x+1)(x-1), cancel (x-1), get x+1", "1+1", 2),
        ("lim(x→4) of x²",           "Direct substitution",   "4**2",   16),
        ("lim(x→5) of 2x+1",         "Direct substitution",   "2*5+1",  11),
    ]
    for q_text, reasoning, expr, result in limits:
        for _ in range(reps * 2):
            out.append(sample(
                f"Find the {q_text}.",
                f"{think(reasoning)}\n"
                f"Evaluate: {calc(expr, result)}\n"
                f"Answer: {result}"
            ))

    # ── Conceptual questions
    concepts = [
        ("What is a derivative?",
         "A derivative is the instantaneous rate of change of a function.\n"
         "Formula: f'(x) = lim(h→0) [f(x+h)-f(x)]/h\n"
         "Geometrically: slope of the tangent line at x."),
        ("What is an integral?",
         "An integral measures the area under a curve.\n"
         "∫f(x)dx = F(x)+C where F'(x)=f(x).\n"
         "Definite integral: ∫ₐᵇf(x)dx = F(b)-F(a)."),
        ("What is the power rule for differentiation?",
         "d/dx[xⁿ] = n·xⁿ⁻¹\n"
         "Bring down the exponent, reduce it by 1.\n"
         "Example: d/dx[x⁵] = 5x⁴"),
        ("What is the power rule for integration?",
         "∫xⁿ dx = xⁿ⁺¹/(n+1) + C  (n ≠ -1)\n"
         "Raise the exponent by 1, divide by the new exponent.\n"
         "Example: ∫x³ dx = x⁴/4 + C"),
        ("What does the Fundamental Theorem of Calculus say?",
         "Differentiation and integration are inverse operations.\n"
         "If F'(x) = f(x), then ∫ₐᵇ f(x)dx = F(b) - F(a)."),
    ]
    for q_text, a_text in concepts:
        for _ in range(reps * 3):
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
    parser.add_argument("--stages",  nargs="*", type=int, default=list(STAGES.keys()))
    parser.add_argument("--stage",   type=int, default=None)
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
