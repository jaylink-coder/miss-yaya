"""Consolidation data generator.

The 16-phase sequential curriculum caused catastrophic forgetting:
  Kenya/Swahili: 0%  (phases 4+14 wiped by later phases)
  Open Knowledge: 0%  (not enough coverage)
  Language:      20%  (overwritten)

Fix: one unified training file mixing ALL phase data with smart weighting
so every capability is represented proportionally in every batch.

Weighting strategy (based on benchmark failures):
  Kenya & Swahili      3x  (scored 0% model-only — worst)
  Open Knowledge       3x  (scored 0%)
  Hard Reasoning       3x  (scored 0%)
  Instruction Follow   2x  (scored 20%)
  Language / Factual   2x  (scored 20-47%)
  All others           1x  (holding steady)

Output: data/sft/yaya_consolidation.jsonl
"""

import json, os, random, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE  = os.path.join(REPO_ROOT, "data/sft/yaya_consolidation.jsonl")
random.seed(2026)

SFT_DIR  = os.path.join(REPO_ROOT, "data/sft")
CURR_DIR = os.path.join(SFT_DIR, "curriculum")

# ── Helper: load a JSONL file ──────────────────────────────────────────────────
def load(path, max_lines=None):
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    if max_lines:
        rows = rows[:max_lines]
    return rows


def load_parsed(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


# ── Convert DPO pairs to SFT (use chosen response) ───────────────────────────
def dpo_to_sft(path):
    """Convert {prompt, chosen, rejected} → {messages:[sys,user,assistant]}."""
    SYS = ("You are Yaya, a helpful, honest, and friendly AI assistant. "
           "Answer questions clearly and concisely.")
    rows = load_parsed(path)
    out = []
    for r in rows:
        prompt  = r.get("prompt",  "")
        chosen  = r.get("chosen",  "")
        # Some DPO files use messages format already
        if "messages" in r:
            out.append(json.dumps(r, ensure_ascii=False))
            continue
        if prompt and chosen:
            obj = {"messages": [
                {"role": "system",    "content": SYS},
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": chosen},
            ]}
            out.append(json.dumps(obj, ensure_ascii=False))
    return out


# ── Load and weight all sources ───────────────────────────────────────────────
# Format: (label, path_or_lines, multiplier)
# multiplier = how many times to repeat (to fix the gaps in failing categories)

sources = []

# ── CRITICAL GAPS (0% model-only) — 3x ────────────────────────────────────────
# Kenya & East Africa
sources.append(("Kenya/Swahili curr",    load(os.path.join(CURR_DIR, "phase04_direct_qa.jsonl")),      3))
sources.append(("Kenya/Swahili phase14", load(os.path.join(CURR_DIR, "phase14_kenya_swahili.jsonl")),  3))
sources.append(("Kenya/Swahili sft",     load(os.path.join(SFT_DIR,  "yaya_phase4_swahili.jsonl")),    3))

# Open Knowledge (biology, science, world facts)
sources.append(("Open Knowledge",        load(os.path.join(CURR_DIR, "phase01_world_knowledge.jsonl")), 3))
sources.append(("Knowledge sft",         load(os.path.join(SFT_DIR,  "yaya_phase5_knowledge.jsonl")),  3))
sources.append(("Quick facts",           load(os.path.join(SFT_DIR,  "teach/quick_facts.jsonl")),       3))

# Hard Reasoning (analogies, temporal, negation)
sources.append(("Logical Reasoning",     load(os.path.join(CURR_DIR, "phase07_logical_reasoning.jsonl")), 3))
sources.append(("Chain of Thought",      load(os.path.join(CURR_DIR, "phase05_chain_of_thought.jsonl")),  3))
sources.append(("Reasoning sft",         load(os.path.join(SFT_DIR,  "yaya_phase3_reasoning.jsonl")),    3))

# ── WEAK AREAS (20-47% model-only) — 2x ──────────────────────────────────────
# Language & Instruction Following
sources.append(("Instruction Follow",    load(os.path.join(CURR_DIR, "phase03_instruction_follow.jsonl")), 2))
sources.append(("Concise SFT",           load(os.path.join(SFT_DIR,  "yaya_concise_sft.jsonl"), 1000),    2))
sources.append(("Short QA",              load(os.path.join(SFT_DIR,  "yaya_short_qa.jsonl")),             2))

# ── STABLE AREAS — 1x ─────────────────────────────────────────────────────────
sources.append(("Conversational",        load(os.path.join(CURR_DIR, "phase02_conversational.jsonl")),     1))
sources.append(("Multi-turn",            load(os.path.join(SFT_DIR,  "yaya_phase2_multiturn.jsonl")),      1))
sources.append(("Math Reasoning",        load(os.path.join(CURR_DIR, "phase06_math_reasoning.jsonl")),     1))
sources.append(("Self Reflection",       load(os.path.join(CURR_DIR, "phase08_self_reflection.jsonl")),    1))
sources.append(("Tool Basics",           load(os.path.join(CURR_DIR, "phase09_tool_basics.jsonl")),        1))
sources.append(("Multi-Step Tools",      load(os.path.join(CURR_DIR, "phase10_multi_tool.jsonl")),         1))
sources.append(("RAG Grounding",         load(os.path.join(CURR_DIR, "phase11_rag_grounding.jsonl")),      1))
sources.append(("Code",                  load(os.path.join(CURR_DIR, "phase12_code.jsonl")),               1))
sources.append(("Code sft",              load(os.path.join(SFT_DIR,  "yaya_phase6_code.jsonl")),           1))
sources.append(("Structured Output",     load(os.path.join(CURR_DIR, "phase13_structured_output.jsonl")),  1))
sources.append(("Safety",                load(os.path.join(CURR_DIR, "phase15_safety.jsonl")),             1))
sources.append(("Persona",               load(os.path.join(SFT_DIR,  "yaya_phase8_persona.jsonl")),        1))

# DPO pairs → convert chosen to SFT
sources.append(("DPO→SFT alignment",    dpo_to_sft(os.path.join(CURR_DIR, "phase16_dpo_pairs.jsonl")),   1))
sources.append(("DPO→SFT phase7",       dpo_to_sft(os.path.join(SFT_DIR,  "yaya_phase7_dpo.jsonl")),     2))


# ── Assemble ──────────────────────────────────────────────────────────────────
print("\nConsolidation data assembly:")
print(f"  {'Source':<28} {'Base':>6} {'×':>2} {'Total':>7}")
print(f"  {'-'*28}  {'-'*6}  {'-'*2}  {'-'*7}")

all_lines = []
for label, lines, mult in sources:
    count = len(lines) * mult
    print(f"  {label:<28} {len(lines):>6}  {mult:>2}  {count:>7}")
    all_lines.extend(lines * mult)

random.shuffle(all_lines)

print(f"\n  Total: {len(all_lines):,} examples")

os.makedirs(SFT_DIR, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")

print(f"  Written → {OUT_FILE}")
print("\n  Recommended training:")
print("    Steps:    3000 (enough to absorb all data ~2-3 epochs)")
print("    LR:       5e-6 (gentle — preserve arithmetic, blend in gaps)")
print("    Batch:    4, accum 8 → eff 32")
print("    Warmup:   200 steps")
