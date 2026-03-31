"""
gen_cot_data.py — Generate chain-of-thought (CoT) reasoning training data.

Teaches Yaya to show its work: step-by-step arithmetic, algebra, logical
deductions, comparisons, and multi-step word problems.  Every answer uses
an explicit numbered-step format so the model learns to reason gradually
before reaching a conclusion.
"""
import sys, json, random, math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYSTEM = (
    "You are Yaya, a helpful, honest, and friendly AI assistant. "
    "You think step by step and show your reasoning before giving a final answer."
)

OUTFILE = "data/sft/yaya_cot_reasoning.jsonl"
random.seed(42)

samples = []

def add(user, assistant):
    samples.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user",  "content": user},
        {"role": "assistant", "content": assistant},
    ]})

# ─────────────────────────────────────────────────────────────────────────────
# 1. ARITHMETIC WITH EXPLICIT STEPS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_add(a, b):
    add(f"What is {a} + {b}?",
        f"Step 1: Start with {a}.\n"
        f"Step 2: Add {b} to get {a} + {b} = {a+b}.\n\n"
        f"**Answer: {a+b}**")

def fmt_sub(a, b):
    add(f"What is {a} - {b}?",
        f"Step 1: Start with {a}.\n"
        f"Step 2: Subtract {b}: {a} - {b} = {a-b}.\n\n"
        f"**Answer: {a-b}**")

def fmt_mul(a, b):
    # Show partial products for two-digit × one-digit
    if b < 10:
        parts = [(a * (b % 10), b % 10)]
        steps = f"Step 1: {a} × {b} = {a*b}.\n\n**Answer: {a*b}**"
    else:
        ones = b % 10
        tens = b // 10
        p1 = a * ones
        p2 = a * tens
        steps = (
            f"Step 1: Break {b} into {tens*10} + {ones}.\n"
            f"Step 2: {a} × {ones} = {p1}.\n"
            f"Step 3: {a} × {tens*10} = {p2*10}.\n"
            f"Step 4: Add partial products: {p1} + {p2*10} = {a*b}.\n\n"
            f"**Answer: {a*b}**"
        )
    add(f"What is {a} × {b}?", steps)

def fmt_div(a, b):
    q, r = divmod(a, b)
    if r == 0:
        add(f"What is {a} ÷ {b}?",
            f"Step 1: Ask how many times {b} fits into {a}.\n"
            f"Step 2: {b} × {q} = {b*q} = {a}, so it fits exactly {q} times.\n\n"
            f"**Answer: {a} ÷ {b} = {q}**")
    else:
        add(f"What is {a} ÷ {b}?",
            f"Step 1: Ask how many times {b} fits into {a}.\n"
            f"Step 2: {b} × {q} = {b*q}.\n"
            f"Step 3: Remainder = {a} - {b*q} = {r}.\n\n"
            f"**Answer: {a} ÷ {b} = {q} remainder {r}**")

for _ in range(120):
    a, b = random.randint(10, 999), random.randint(1, 999)
    fmt_add(a, b)
for _ in range(120):
    a, b = random.randint(10, 999), random.randint(1, min(a, 999))
    fmt_sub(a, b)
for _ in range(120):
    a, b = random.randint(2, 99), random.randint(2, 99)
    fmt_mul(a, b)
for _ in range(120):
    b = random.randint(2, 20)
    q = random.randint(2, 50)
    r = random.randint(0, b - 1)
    a = b * q + r
    fmt_div(a, b)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FRACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def gcd(a, b):
    while b: a, b = b, a % b
    return a

def simplify(n, d):
    g = gcd(abs(n), abs(d))
    return n // g, d // g

for _ in range(80):
    # Add/subtract fractions with different denominators
    a, b = random.randint(1, 9), random.randint(2, 12)
    c, d = random.randint(1, 9), random.randint(2, 12)
    lcd = b * d // gcd(b, d)
    an, cn = a * (lcd // b), c * (lcd // d)
    sn = an + cn
    sn2, sd2 = simplify(sn, lcd)
    op = "+"
    add(f"What is {a}/{b} + {c}/{d}?",
        f"Step 1: Find the LCD of {b} and {d}. LCD = {lcd}.\n"
        f"Step 2: Convert fractions: {a}/{b} = {an}/{lcd}, {c}/{d} = {cn}/{lcd}.\n"
        f"Step 3: Add numerators: {an} + {cn} = {sn}.\n"
        f"Step 4: Simplify {sn}/{lcd} → {sn2}/{sd2}.\n\n"
        f"**Answer: {sn2}/{sd2}**")

for _ in range(60):
    # Multiply fractions
    a, b = random.randint(1, 9), random.randint(2, 12)
    c, d = random.randint(1, 9), random.randint(2, 12)
    pn, pd = simplify(a * c, b * d)
    add(f"What is {a}/{b} × {c}/{d}?",
        f"Step 1: Multiply numerators: {a} × {c} = {a*c}.\n"
        f"Step 2: Multiply denominators: {b} × {d} = {b*d}.\n"
        f"Step 3: Simplify {a*c}/{b*d} → {pn}/{pd}.\n\n"
        f"**Answer: {pn}/{pd}**")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PERCENTAGES
# ─────────────────────────────────────────────────────────────────────────────

pct_templates = [
    (lambda p, w: (
        f"What is {p}% of {w}?",
        f"Step 1: Convert {p}% to a decimal: {p}/100 = {p/100}.\n"
        f"Step 2: Multiply: {p/100} × {w} = {p/100*w:.2f}.\n\n"
        f"**Answer: {p/100*w:.2f}**"
    )),
    (lambda p, w: (
        f"A shirt costs ${w}. After a {p}% discount, what is the new price?",
        f"Step 1: Calculate the discount: {p}% of ${w} = {p/100} × {w} = ${p/100*w:.2f}.\n"
        f"Step 2: Subtract from original price: ${w} - ${p/100*w:.2f} = ${w - p/100*w:.2f}.\n\n"
        f"**Answer: ${w - p/100*w:.2f}**"
    )),
]

for _ in range(80):
    p = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
    w = random.randint(20, 500)
    fn = random.choice(pct_templates)
    u, a = fn(p, w)
    add(u, a)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ALGEBRA — SOLVING EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────

for _ in range(100):
    # ax + b = c  →  x = (c - b) / a
    a = random.randint(1, 10)
    x = random.randint(-20, 20)
    b = random.randint(-30, 30)
    c = a * x + b
    add(f"Solve for x: {a}x + {b} = {c}",
        f"Step 1: Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}.\n"
        f"Step 2: Divide both sides by {a}: x = {c-b}/{a} = {x}.\n\n"
        f"**Answer: x = {x}**")

for _ in range(80):
    # Two-step: ax - b = cx + d  (ensure integer solution)
    a = random.randint(2, 8)
    c_coef = random.randint(1, a - 1)
    x = random.randint(-10, 10)
    b = random.randint(-15, 15)
    d = (a - c_coef) * x - b
    lhs = f"{a}x - {b}" if b >= 0 else f"{a}x + {abs(b)}"
    rhs = f"{c_coef}x + {d}" if d >= 0 else f"{c_coef}x - {abs(d)}"
    add(f"Solve for x: {a}x - {b} = {c_coef}x + {d}",
        f"Step 1: Move x terms to the left: {a}x - {c_coef}x = {d} + {b}.\n"
        f"Step 2: Simplify: {a - c_coef}x = {d + b}.\n"
        f"Step 3: Divide by {a - c_coef}: x = {d+b}/{a-c_coef} = {x}.\n\n"
        f"**Answer: x = {x}**")

# ─────────────────────────────────────────────────────────────────────────────
# 5. WORD PROBLEMS
# ─────────────────────────────────────────────────────────────────────────────

word_problems = [
    # distance/rate/time
    lambda: (
        (r := random.randint(40, 120)),
        (t := random.randint(1, 8)),
        (
            f"A car travels at {r} km/h for {t} hours. How far does it travel?",
            f"Step 1: Use the formula: Distance = Speed × Time.\n"
            f"Step 2: Distance = {r} km/h × {t} h = {r*t} km.\n\n"
            f"**Answer: {r*t} km**"
        )
    )[-1],
    # work rate
    lambda: (
        (a := random.randint(2, 10)),
        (b := random.randint(a+1, a+8)),
        (
            f"Person A can do a job in {a} days. Person B can do it in {b} days. "
            f"How long do they take working together?",
            f"Step 1: Person A's rate = 1/{a} job/day. Person B's rate = 1/{b} job/day.\n"
            f"Step 2: Combined rate = 1/{a} + 1/{b} = {b}/{a*b} + {a}/{a*b} = {a+b}/{a*b} job/day.\n"
            f"Step 3: Time = 1 ÷ ({a+b}/{a*b}) = {a*b}/{a+b} ≈ {a*b/(a+b):.2f} days.\n\n"
            f"**Answer: {a*b/(a+b):.2f} days**"
        )
    )[-1],
    # age problems
    lambda: (
        (now := random.randint(8, 30)),
        (older := random.randint(now+5, now+30)),
        (years := random.randint(2, 10)),
        (
            f"Alex is {now} years old and Jordan is {older} years old. "
            f"How old will they be in {years} years, and what will their age difference be?",
            f"Step 1: Alex in {years} years: {now} + {years} = {now+years}.\n"
            f"Step 2: Jordan in {years} years: {older} + {years} = {older+years}.\n"
            f"Step 3: Age difference: {older+years} - {now+years} = {older-now} (stays constant).\n\n"
            f"**Answer: Alex = {now+years}, Jordan = {older+years}, difference = {older-now}**"
        )
    )[-1],
    # mixture/ratio
    lambda: (
        (r1 := random.randint(1, 5)),
        (r2 := random.randint(1, 5)),
        (total := random.randint(10, 100)),
        (
            f"A mixture uses ingredients A and B in a ratio of {r1}:{r2}. "
            f"If the total mixture is {total} units, how much of each is needed?",
            f"Step 1: Total ratio parts = {r1} + {r2} = {r1+r2}.\n"
            f"Step 2: Amount of A = ({r1}/{r1+r2}) × {total} = {r1*total/(r1+r2):.1f} units.\n"
            f"Step 3: Amount of B = ({r2}/{r1+r2}) × {total} = {r2*total/(r1+r2):.1f} units.\n\n"
            f"**Answer: A = {r1*total/(r1+r2):.1f} units, B = {r2*total/(r1+r2):.1f} units**"
        )
    )[-1],
    # profit/loss
    lambda: (
        (cp := random.randint(50, 500)),
        (sp := random.randint(40, 600)),
        (
            f"An item was bought for ${cp} and sold for ${sp}. "
            f"What is the profit or loss, and what is the percentage?",
            (
                f"Step 1: {'Profit' if sp >= cp else 'Loss'} = |${sp} - ${cp}| = ${abs(sp-cp)}.\n"
                f"Step 2: Percentage = ({abs(sp-cp)}/{cp}) × 100 = {abs(sp-cp)/cp*100:.1f}%.\n\n"
                f"**Answer: {'Profit' if sp >= cp else 'Loss'} of ${abs(sp-cp)} ({abs(sp-cp)/cp*100:.1f}%)**"
            )
        )
    )[-1],
]

for _ in range(150):
    fn = random.choice(word_problems)
    u, a = fn()
    add(u, a)

# ─────────────────────────────────────────────────────────────────────────────
# 6. LOGICAL DEDUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

logic_templates = [
    # Syllogism
    ("All {A}s are {B}s. {X} is a {A}. Is {X} a {B}?",
     "Step 1: Premise 1 — All {A}s are {B}s.\n"
     "Step 2: Premise 2 — {X} is a {A}.\n"
     "Step 3: Applying Premise 1 to Premise 2: since {X} is a {A}, and all {A}s are {B}s, {X} must be a {B}.\n\n"
     "**Answer: Yes, {X} is a {B}.**"),

    # Conditional
    ("If it rains, the ground is wet. The ground is not wet. Did it rain?",
     "Step 1: Premise — If it rains → ground is wet.\n"
     "Step 2: Observation — ground is NOT wet.\n"
     "Step 3: By contrapositive: if ground is not wet → it did NOT rain.\n\n"
     "**Answer: No, it did not rain.**"),

    ("If {A} then {B}. {B} is false. Is {A} true?",
     "Step 1: Premise — If {A} then {B} (i.e., {A} → {B}).\n"
     "Step 2: Observation — {B} is false.\n"
     "Step 3: By modus tollens (contrapositive): NOT {B} → NOT {A}.\n\n"
     "**Answer: No, {A} cannot be true.**"),

    # Ordering
    ("{A} is taller than {B}. {B} is taller than {C}. Who is the shortest?",
     "Step 1: {A} > {B} (in height).\n"
     "Step 2: {B} > {C} (in height).\n"
     "Step 3: Combining: {A} > {B} > {C}.\n\n"
     "**Answer: {C} is the shortest.**"),

    # Parity / counting
    ("A bag has 3 red balls and 5 blue balls. If you draw one at random, "
     "what is the probability of drawing a red ball?",
     "Step 1: Total balls = 3 + 5 = 8.\n"
     "Step 2: Red balls = 3.\n"
     "Step 3: Probability = favorable / total = 3/8.\n\n"
     "**Answer: 3/8 (or 37.5%)**"),

    ("There are 4 people in a room. Each person shakes hands with every other person exactly once. "
     "How many handshakes are there?",
     "Step 1: Each of the 4 people shakes hands with 3 others → 4 × 3 = 12 individual handshakes.\n"
     "Step 2: But each handshake is counted twice (once per person), so divide by 2: 12 / 2 = 6.\n"
     "Step 3: Alternatively, use the formula C(n,2) = n(n−1)/2 = 4×3/2 = 6.\n\n"
     "**Answer: 6 handshakes**"),
]

names_A = ["mammal", "bird", "reptile", "vehicle", "fruit", "planet", "prime number"]
names_B = ["living thing", "animal", "organism", "object", "natural thing", "number"]
names_X = ["a dog", "an eagle", "a snake", "a car", "an apple", "Mars", "7"]
cond_A = ["it is sunny", "the alarm rings", "the power goes out", "the door is locked"]
cond_B = ["we go outside", "everyone wakes up", "the lights turn off", "no one can enter"]
person_names = ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona", "Grace", "Henry"]

for _ in range(150):
    tpl_u, tpl_a = random.choice(logic_templates[:1] + logic_templates[2:4])
    if "{A}" in tpl_u and "{B}" in tpl_u and "{X}" in tpl_u:
        idx = random.randint(0, len(names_A) - 1)
        A, B, X = names_A[idx], names_B[min(idx, len(names_B)-1)], names_X[min(idx, len(names_X)-1)]
        u = tpl_u.format(A=A, B=B, X=X)
        a = tpl_a.format(A=A, B=B, X=X)
    elif "{A}" in tpl_u and "{B}" in tpl_u and "{C}" in tpl_u:
        n1, n2, n3 = random.sample(person_names, 3)
        u = tpl_u.format(A=n1, B=n2, C=n3)
        a = tpl_a.format(A=n1, B=n2, C=n3)
    elif "{A}" in tpl_u:
        ca, cb = random.choice(cond_A), random.choice(cond_B)
        u = tpl_u.format(A=ca, B=cb)
        a = tpl_a.format(A=ca, B=cb)
    else:
        u, a = tpl_u, tpl_a
    add(u, a)

# Fixed logical puzzles (verbatim)
for _ in range(2):
    add(*logic_templates[1])  # rain/ground
    add(*logic_templates[4])  # probability
    add(*logic_templates[5])  # handshakes

# ─────────────────────────────────────────────────────────────────────────────
# 7. NUMBER PROPERTIES & SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

for _ in range(60):
    n = random.randint(10, 200)
    is_prime = all(n % i != 0 for i in range(2, int(n**0.5)+1)) and n > 1
    if is_prime:
        add(f"Is {n} a prime number?",
            f"Step 1: Check divisibility by primes up to √{n} ≈ {int(n**0.5)}.\n"
            f"Step 2: {n} is not divisible by any prime ≤ {int(n**0.5)}.\n\n"
            f"**Answer: Yes, {n} is a prime number.**")
    else:
        factors = [i for i in range(2, n) if n % i == 0]
        add(f"Is {n} a prime number?",
            f"Step 1: Check if {n} has any divisors between 2 and √{n} ≈ {int(n**0.5)}.\n"
            f"Step 2: {n} ÷ {factors[0]} = {n // factors[0]} — so {factors[0]} is a factor.\n\n"
            f"**Answer: No, {n} is not prime. It is divisible by {factors[0]}.**")

for _ in range(50):
    # Arithmetic sequences
    a0 = random.randint(1, 20)
    d  = random.randint(1, 10)
    seq = [a0 + d*i for i in range(5)]
    n_th = random.randint(6, 15)
    val = a0 + d * (n_th - 1)
    add(f"What is the next number in the sequence: {', '.join(map(str, seq))}, ...? Also find the {n_th}th term.",
        f"Step 1: Find the common difference: {seq[1]} - {seq[0]} = {d} (confirmed across all terms).\n"
        f"Step 2: Next number: {seq[-1]} + {d} = {seq[-1]+d}.\n"
        f"Step 3: {n_th}th term formula: a₁ + (n-1)d = {a0} + ({n_th}-1)×{d} = {a0} + {(n_th-1)*d} = {val}.\n\n"
        f"**Answer: Next = {seq[-1]+d}, {n_th}th term = {val}**")

# ─────────────────────────────────────────────────────────────────────────────
# 8. UNIT CONVERSIONS (step-by-step)
# ─────────────────────────────────────────────────────────────────────────────

conversions = [
    (lambda v: (
        f"Convert {v} kilometers to miles.",
        f"Step 1: Use the conversion factor: 1 km = 0.621371 miles.\n"
        f"Step 2: {v} km × 0.621371 = {v*0.621371:.3f} miles.\n\n"
        f"**Answer: {v*0.621371:.3f} miles**"
    )),
    (lambda v: (
        f"Convert {v}°C to Fahrenheit.",
        f"Step 1: Use the formula: F = (C × 9/5) + 32.\n"
        f"Step 2: F = ({v} × 9/5) + 32 = ({v*9/5:.1f}) + 32 = {v*9/5+32:.1f}°F.\n\n"
        f"**Answer: {v*9/5+32:.1f}°F**"
    )),
    (lambda v: (
        f"Convert {v} hours to seconds.",
        f"Step 1: 1 hour = 60 minutes.\n"
        f"Step 2: {v} hours × 60 = {v*60} minutes.\n"
        f"Step 3: 1 minute = 60 seconds → {v*60} minutes × 60 = {v*3600} seconds.\n\n"
        f"**Answer: {v*3600} seconds**"
    )),
    (lambda v: (
        f"Convert {v} kilograms to pounds.",
        f"Step 1: Use the conversion factor: 1 kg = 2.20462 lbs.\n"
        f"Step 2: {v} kg × 2.20462 = {v*2.20462:.3f} lbs.\n\n"
        f"**Answer: {v*2.20462:.3f} lbs**"
    )),
]

for _ in range(80):
    fn = random.choice(conversions)
    v  = random.randint(1, 150)
    u, a = fn(v)
    add(u, a)

# ─────────────────────────────────────────────────────────────────────────────
# 9. MULTI-STEP REASONING (cause → effect, planning)
# ─────────────────────────────────────────────────────────────────────────────

multistep = [
    ("You have $200. You spend 30% on food, 25% on transport, and save the rest. "
     "How much do you save?",
     "Step 1: Food cost = 30% of $200 = 0.30 × 200 = $60.\n"
     "Step 2: Transport cost = 25% of $200 = 0.25 × 200 = $50.\n"
     "Step 3: Total spent = $60 + $50 = $110.\n"
     "Step 4: Savings = $200 - $110 = $90.\n\n"
     "**Answer: You save $90.**"),

    ("A recipe calls for 2.5 cups of flour for 12 cookies. "
     "How much flour is needed for 30 cookies?",
     "Step 1: Find flour per cookie: 2.5 cups ÷ 12 = 0.2083 cups/cookie.\n"
     "Step 2: Scale up: 0.2083 × 30 = 6.25 cups.\n"
     "Alternatively: ratio method — 30/12 = 2.5×, so 2.5 × 2.5 = 6.25 cups.\n\n"
     "**Answer: 6.25 cups of flour**"),

    ("A train leaves Station A at 9:00 AM traveling at 80 km/h. "
     "Another train leaves Station B (240 km away) at 10:00 AM traveling at 120 km/h toward the first train. "
     "At what time do they meet?",
     "Step 1: By 10:00 AM, Train A has traveled 1 hour × 80 km/h = 80 km.\n"
     "Step 2: Remaining gap at 10:00 AM = 240 - 80 = 160 km.\n"
     "Step 3: Closing speed = 80 + 120 = 200 km/h.\n"
     "Step 4: Time to close gap = 160 / 200 = 0.8 hours = 48 minutes.\n"
     "Step 5: They meet at 10:00 AM + 48 min = 10:48 AM.\n\n"
     "**Answer: 10:48 AM**"),

    ("You invest $1000 at 5% annual compound interest. How much do you have after 3 years?",
     "Step 1: Use the formula: A = P(1 + r)^t.\n"
     "Step 2: A = 1000 × (1 + 0.05)^3 = 1000 × (1.05)^3.\n"
     "Step 3: (1.05)^3 = 1.05 × 1.05 × 1.05 = 1.1025 × 1.05 = 1.157625.\n"
     "Step 4: A = 1000 × 1.157625 = $1157.63.\n\n"
     "**Answer: $1157.63**"),

    ("If you read 40 pages per day, and a book has 320 pages, "
     "how many days will it take to finish? If you start on Monday, which day do you finish?",
     "Step 1: Days needed = 320 ÷ 40 = 8 days.\n"
     "Step 2: Starting Monday, count 8 days: Mon(1), Tue(2), Wed(3), Thu(4), Fri(5), Sat(6), Sun(7), Mon(8).\n\n"
     "**Answer: 8 days; you finish on Monday.**"),

    ("A store buys shirts for $15 each and sells them at a 60% markup. "
     "During a sale, they offer 20% off the selling price. "
     "What is the sale price and is it still profitable?",
     "Step 1: Markup price = $15 × (1 + 60/100) = $15 × 1.60 = $24.\n"
     "Step 2: Sale discount = 20% of $24 = $4.80.\n"
     "Step 3: Sale price = $24 - $4.80 = $19.20.\n"
     "Step 4: Profit per shirt = $19.20 - $15.00 = $4.20 > 0.\n\n"
     "**Answer: Sale price = $19.20; yes, still profitable ($4.20 profit per shirt).**"),

    ("A rectangle has a perimeter of 54 cm. Its length is twice its width. "
     "Find the dimensions and area.",
     "Step 1: Let width = w, length = 2w.\n"
     "Step 2: Perimeter = 2(length + width) = 2(2w + w) = 6w = 54.\n"
     "Step 3: w = 54/6 = 9 cm; length = 2 × 9 = 18 cm.\n"
     "Step 4: Area = length × width = 18 × 9 = 162 cm².\n\n"
     "**Answer: Width = 9 cm, Length = 18 cm, Area = 162 cm²**"),
]

for item in multistep:
    add(*item)
    add(*item)   # duplicate for higher weight

# ─────────────────────────────────────────────────────────────────────────────
# 10. GENERAL REASONING — "THINK THROUGH IT" PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

think_prompts = [
    ("Which is larger: 7/8 or 5/6?",
     "Step 1: Convert both fractions to a common denominator.\n"
     "  LCD of 8 and 6 = 24.\n"
     "  7/8 = 21/24\n"
     "  5/6 = 20/24\n"
     "Step 2: Compare: 21/24 > 20/24.\n\n"
     "**Answer: 7/8 is larger.**"),

    ("If a number is multiplied by 3 and then 12 is subtracted, the result is 33. What is the number?",
     "Step 1: Let the number be x.\n"
     "Step 2: Equation: 3x - 12 = 33.\n"
     "Step 3: Add 12 to both sides: 3x = 45.\n"
     "Step 4: Divide by 3: x = 15.\n\n"
     "**Answer: The number is 15.**"),

    ("How many times does the digit 3 appear in the numbers from 1 to 100?",
     "Step 1: Count 3 in the units place: 3, 13, 23, 33, 43, 53, 63, 73, 83, 93 → 10 times.\n"
     "Step 2: Count 3 in the tens place: 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 → 10 times.\n"
     "Step 3: Note 33 was counted in both steps (that's correct — it has two 3s).\n"
     "Step 4: Total = 10 + 10 = 20.\n\n"
     "**Answer: 20 times.**"),

    ("A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. "
     "How much does the ball cost?",
     "Step 1: Let ball = b, bat = b + 1.00.\n"
     "Step 2: b + (b + 1.00) = 1.10 → 2b = 0.10 → b = 0.05.\n"
     "Step 3: Check: bat = $1.05, ball = $0.05, total = $1.10. ✓\n\n"
     "**Answer: The ball costs $0.05 (5 cents).**\n"
     "*Note: The intuitive answer of $0.10 is a common error — always verify.*"),

    ("You have two ropes. Each takes exactly 60 minutes to burn, but burns unevenly. "
     "How can you measure exactly 45 minutes?",
     "Step 1: Light both ends of Rope 1, and one end of Rope 2 simultaneously.\n"
     "Step 2: Rope 1, burning from both ends, takes 30 minutes to finish.\n"
     "Step 3: At that moment (30 min elapsed), light the other end of Rope 2.\n"
     "Step 4: Rope 2 had 30 minutes of burn left; burning from both ends halves that to 15 minutes.\n"
     "Step 5: Total time = 30 + 15 = 45 minutes.\n\n"
     "**Answer: Light Rope 1 from both ends + Rope 2 from one end; when Rope 1 finishes (30 min), light the other end of Rope 2 (15 min more) = 45 min total.**"),

    ("What is the sum of the first 100 natural numbers?",
     "Step 1: Use Gauss's formula: Sum = n(n+1)/2.\n"
     "Step 2: For n = 100: Sum = 100 × 101 / 2 = 10100 / 2 = 5050.\n"
     "Alternative: Pair up: (1+100) + (2+99) + ... = 101 × 50 pairs = 5050.\n\n"
     "**Answer: 5050**"),

    ("Is 0.999... (repeating) equal to 1?",
     "Step 1: Let x = 0.999...\n"
     "Step 2: Multiply both sides by 10: 10x = 9.999...\n"
     "Step 3: Subtract the original: 10x - x = 9.999... - 0.999... → 9x = 9.\n"
     "Step 4: Divide by 9: x = 1.\n\n"
     "**Answer: Yes, 0.999... = 1 exactly.** It's not approximately 1 — they are mathematically identical."),

    ("A clock shows 3:15. What is the angle between the hour hand and minute hand?",
     "Step 1: Minute hand at 15 min → 15 × 6° = 90° from 12.\n"
     "Step 2: Hour hand at 3:15 → 3 hours + 15/60 hour = 3.25 hours → 3.25 × 30° = 97.5°.\n"
     "Step 3: Angle between them = |97.5° - 90°| = 7.5°.\n\n"
     "**Answer: 7.5°**"),
]

for item in think_prompts:
    add(*item)
    add(*item)   # higher weight

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

random.shuffle(samples)

with open(OUTFILE, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Wrote {len(samples)} CoT samples to {OUTFILE}")
