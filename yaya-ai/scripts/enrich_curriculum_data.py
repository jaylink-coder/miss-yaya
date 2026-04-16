"""Enrich curriculum SFT data with open-source HuggingFace datasets.

Downloads high-quality instruction datasets and maps them to milestones_v2
phases by content. Run this on Kaggle before curriculum training to beef up
each phase from ~4-16K to ~20-40K examples.

Usage:
    python scripts/enrich_curriculum_data.py [--phase N]

Requires: pip install datasets
"""

import json
import os
import sys
import random
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRICULUM_DIR = os.path.join(REPO_ROOT, 'data', 'sft', 'curriculum')

# ── Helpers ───────────────────────────────────────────────────────────────────

def msg(role, content):
    """Create a message dict."""
    return {"role": role, "content": content.strip()}


def save_jsonl(data, path):
    """Save list of dicts to JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'  Saved {len(data):,} examples → {os.path.basename(path)}')


def load_existing(path):
    """Load existing JSONL file."""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Phase enrichment functions ────────────────────────────────────────────────
# Each function downloads a specific HF dataset and formats it for Yaya.

def enrich_phase1_world_knowledge():
    """Phase 1: World Knowledge — add SciQ, OpenBookQA, ARC."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase03', 'hf_science_qa.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    # SciQ — science Q&A with support text
    print('  Downloading SciQ...')
    try:
        ds = load_dataset('allenai/sciq', split='train', trust_remote_code=True)
        for ex in ds:
            q = ex.get('question', '').strip()
            a = ex.get('correct_answer', '').strip()
            support = ex.get('support', '').strip()
            if q and a:
                answer = f"{a}. {support}" if support else a
                examples.append({"messages": [msg("user", q), msg("assistant", answer)]})
    except Exception as e:
        print(f'  SciQ failed: {e}')

    # ARC (AI2 Reasoning Challenge) — science multiple choice
    print('  Downloading ARC-Easy...')
    try:
        ds = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train', trust_remote_code=True)
        for ex in ds:
            q = ex.get('question', '').strip()
            choices = ex.get('choices', {})
            answer_key = ex.get('answerKey', '')
            if q and choices and answer_key:
                labels = choices.get('label', [])
                texts = choices.get('text', [])
                if answer_key in labels:
                    idx = labels.index(answer_key)
                    a = texts[idx]
                    examples.append({"messages": [msg("user", q), msg("assistant", a)]})
    except Exception as e:
        print(f'  ARC failed: {e}')

    # ARC Challenge
    print('  Downloading ARC-Challenge...')
    try:
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train', trust_remote_code=True)
        for ex in ds:
            q = ex.get('question', '').strip()
            choices = ex.get('choices', {})
            answer_key = ex.get('answerKey', '')
            if q and choices and answer_key:
                labels = choices.get('label', [])
                texts = choices.get('text', [])
                if answer_key in labels:
                    idx = labels.index(answer_key)
                    a = texts[idx]
                    examples.append({"messages": [msg("user", q), msg("assistant", a)]})
    except Exception as e:
        print(f'  ARC-Challenge failed: {e}')

    random.shuffle(examples)
    save_jsonl(examples, out_path)
    return examples


def enrich_phase2_conversational():
    """Phase 2: Conversational — add OpenAssistant conversations."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase01', 'hf_oasst_conversations.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    print('  Downloading OpenAssistant (oasst2)...')
    try:
        ds = load_dataset('OpenAssistant/oasst2', split='train', trust_remote_code=True)
        # Build message trees — extract root user + first assistant reply pairs
        by_parent = {}
        roots = []
        for row in ds:
            mid = row.get('message_id', '')
            pid = row.get('parent_id', None)
            role = row.get('role', '')
            text = row.get('text', '').strip()
            if not text or len(text) < 10:
                continue
            if pid is None:
                roots.append({'id': mid, 'role': role, 'text': text})
            else:
                by_parent.setdefault(pid, []).append({'id': mid, 'role': role, 'text': text})

        for root in roots:
            if root['role'] != 'prompter':
                continue
            children = by_parent.get(root['id'], [])
            assistants = [c for c in children if c['role'] == 'assistant']
            if assistants:
                # Pick highest quality reply (first one in dataset order)
                reply = assistants[0]['text']
                if len(reply) > 20:
                    examples.append({"messages": [
                        msg("user", root['text']),
                        msg("assistant", reply)
                    ]})
        # Cap at 10K to keep balanced
        if len(examples) > 10000:
            random.shuffle(examples)
            examples = examples[:10000]
    except Exception as e:
        print(f'  OpenAssistant failed: {e}')

    random.shuffle(examples)
    save_jsonl(examples, out_path)
    return examples


def enrich_phase3_instruction():
    """Phase 3: Instruction Following — add Alpaca."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase08', 'hf_alpaca.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    print('  Downloading Alpaca-cleaned...')
    try:
        ds = load_dataset('yahma/alpaca-cleaned', split='train', trust_remote_code=True)
        for ex in ds:
            instruction = ex.get('instruction', '').strip()
            inp = ex.get('input', '').strip()
            output = ex.get('output', '').strip()
            if not instruction or not output:
                continue
            q = f"{instruction}\n{inp}" if inp else instruction
            examples.append({"messages": [msg("user", q), msg("assistant", output)]})
    except Exception as e:
        print(f'  Alpaca failed: {e}')

    # Cap at 15K
    if len(examples) > 15000:
        random.shuffle(examples)
        examples = examples[:15000]

    save_jsonl(examples, out_path)
    return examples


def enrich_phase4_qa():
    """Phase 4: Direct Q&A — add TriviaQA, Natural Questions."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase02', 'hf_trivia_qa.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    # TriviaQA
    print('  Downloading TriviaQA (subset)...')
    try:
        ds = load_dataset('trivia_qa', 'rc.nocontext', split='train',
                          trust_remote_code=True)
        for i, ex in enumerate(ds):
            if i >= 10000:
                break
            q = ex.get('question', '').strip()
            answers = ex.get('answer', {}).get('aliases', [])
            if q and answers:
                a = answers[0]  # first alias
                examples.append({"messages": [msg("user", q), msg("assistant", a)]})
    except Exception as e:
        print(f'  TriviaQA failed: {e}')

    random.shuffle(examples)
    save_jsonl(examples, out_path)
    return examples


def enrich_phase5_cot():
    """Phase 5: Chain-of-Thought — add CoT examples from FLAN."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase07', 'hf_cot_reasoning.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    # StrategyQA — yes/no with reasoning
    print('  Downloading StrategyQA...')
    try:
        ds = load_dataset('wics/strategy-qa', split='train', trust_remote_code=True)
        for ex in ds:
            q = ex.get('question', '').strip()
            answer = ex.get('answer', '')
            facts = ex.get('facts', [])
            if q and answer is not None:
                reasoning = ' '.join(facts) if facts else ''
                ans_str = 'Yes' if answer else 'No'
                reply = f"<|think|>\n{reasoning}\n</|think|>\n{ans_str}." if reasoning else ans_str
                examples.append({"messages": [msg("user", q), msg("assistant", reply)]})
    except Exception as e:
        print(f'  StrategyQA failed: {e}')

    random.shuffle(examples)
    save_jsonl(examples, out_path)
    return examples


def enrich_phase6_math():
    """Phase 6: Math Reasoning — add GSM8K, MATH."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase05', 'hf_gsm8k.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    # GSM8K — grade school math
    print('  Downloading GSM8K...')
    try:
        ds = load_dataset('openai/gsm8k', 'main', split='train', trust_remote_code=True)
        for ex in ds:
            q = ex.get('question', '').strip()
            a = ex.get('answer', '').strip()
            if q and a:
                # Wrap reasoning in think tags
                parts = a.rsplit('####', 1)
                if len(parts) == 2:
                    reasoning = parts[0].strip()
                    final = parts[1].strip()
                    reply = f"<|think|>\n{reasoning}\n</|think|>\n{final}"
                else:
                    reply = a
                examples.append({"messages": [msg("user", q), msg("assistant", reply)]})
    except Exception as e:
        print(f'  GSM8K failed: {e}')

    # MathQA
    print('  Downloading MathQA (subset)...')
    try:
        ds = load_dataset('allenai/math_qa', split='train', trust_remote_code=True)
        for i, ex in enumerate(ds):
            if i >= 8000:
                break
            q = ex.get('Problem', '').strip()
            rationale = ex.get('Rationale', '').strip()
            answer = ex.get('correct', '').strip()
            options = ex.get('options', '').strip()
            if q and answer:
                reply = f"<|think|>\n{rationale}\n</|think|>\n{answer}" if rationale else answer
                examples.append({"messages": [msg("user", q), msg("assistant", reply)]})
    except Exception as e:
        print(f'  MathQA failed: {e}')

    random.shuffle(examples)
    save_jsonl(examples, out_path)
    return examples


def enrich_phase7_logic():
    """Phase 7: Logical Reasoning — add LogiQA, PIQA."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase06', 'hf_logic.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    # PIQA — physical intuition QA
    print('  Downloading PIQA...')
    try:
        ds = load_dataset('ybisk/piqa', split='train', trust_remote_code=True)
        for ex in ds:
            goal = ex.get('goal', '').strip()
            sol1 = ex.get('sol1', '').strip()
            sol2 = ex.get('sol2', '').strip()
            label = ex.get('label', -1)
            if goal and label in [0, 1]:
                answer = sol1 if label == 0 else sol2
                examples.append({"messages": [msg("user", goal), msg("assistant", answer)]})
    except Exception as e:
        print(f'  PIQA failed: {e}')

    # WinoGrande — commonsense reasoning
    print('  Downloading WinoGrande...')
    try:
        ds = load_dataset('allenai/winogrande', 'winogrande_xl', split='train',
                          trust_remote_code=True)
        for ex in ds:
            sentence = ex.get('sentence', '').strip()
            opt1 = ex.get('option1', '').strip()
            opt2 = ex.get('option2', '').strip()
            answer = ex.get('answer', '')
            if sentence and answer in ['1', '2']:
                correct = opt1 if answer == '1' else opt2
                q = sentence.replace('_', '____')
                examples.append({"messages": [
                    msg("user", f"Fill in the blank: {q}"),
                    msg("assistant", correct)
                ]})
    except Exception as e:
        print(f'  WinoGrande failed: {e}')

    # Cap at 15K
    if len(examples) > 15000:
        random.shuffle(examples)
        examples = examples[:15000]

    save_jsonl(examples, out_path)
    return examples


def enrich_phase12_code():
    """Phase 12: Code Understanding — add CodeAlpaca."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase12', 'hf_code_alpaca.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    print('  Downloading CodeAlpaca-20k...')
    try:
        ds = load_dataset('sahil2801/CodeAlpaca-20k', split='train', trust_remote_code=True)
        for ex in ds:
            instruction = ex.get('instruction', '').strip()
            inp = ex.get('input', '').strip()
            output = ex.get('output', '').strip()
            if not instruction or not output:
                continue
            q = f"{instruction}\n{inp}" if inp else instruction
            examples.append({"messages": [msg("user", q), msg("assistant", output)]})
    except Exception as e:
        print(f'  CodeAlpaca failed: {e}')

    random.shuffle(examples)
    save_jsonl(examples, out_path)
    return examples


def enrich_phase15_safety():
    """Phase 15: Safety — add safety/refusal examples."""
    out_path = os.path.join(CURRICULUM_DIR, 'phase15', 'hf_safety.jsonl')
    if os.path.exists(out_path):
        return load_existing(out_path)

    from datasets import load_dataset
    examples = []

    # Do-Not-Answer dataset
    print('  Downloading Do-Not-Answer...')
    try:
        ds = load_dataset('LibrAI/do-not-answer', split='train', trust_remote_code=True)
        for ex in ds:
            q = ex.get('question', '').strip()
            if q:
                examples.append({"messages": [
                    msg("user", q),
                    msg("assistant", "I'm sorry, but I can't help with that request. "
                        "It could cause harm or violate safety guidelines. "
                        "Is there something else I can help you with?")
                ]})
    except Exception as e:
        print(f'  Do-Not-Answer failed: {e}')

    random.shuffle(examples)
    save_jsonl(examples, out_path)
    return examples


# ── Phase enrichment registry ─────────────────────────────────────────────────
ENRICHERS = {
    1:  ("World Knowledge",       enrich_phase1_world_knowledge),
    2:  ("Conversational",        enrich_phase2_conversational),
    3:  ("Instruction Following", enrich_phase3_instruction),
    4:  ("Direct Q&A",            enrich_phase4_qa),
    5:  ("Chain-of-Thought",      enrich_phase5_cot),
    6:  ("Math Reasoning",        enrich_phase6_math),
    7:  ("Logical Reasoning",     enrich_phase7_logic),
    12: ("Code Understanding",    enrich_phase12_code),
    15: ("Safety & Refusals",     enrich_phase15_safety),
}


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Enrich curriculum data with HF datasets')
    parser.add_argument('--phase', type=int, default=None,
                        help='Enrich a specific phase (default: all)')
    args = parser.parse_args()

    print('=' * 60)
    print(' CURRICULUM DATA ENRICHMENT')
    print(' Downloading open-source datasets from HuggingFace')
    print('=' * 60)

    if args.phase:
        phases = [args.phase]
    else:
        phases = sorted(ENRICHERS.keys())

    total_new = 0
    for phase_id in phases:
        if phase_id not in ENRICHERS:
            print(f'\nPhase {phase_id}: no enricher available')
            continue
        name, func = ENRICHERS[phase_id]
        print(f'\nPhase {phase_id}: {name}')
        examples = func()
        total_new += len(examples)

    print(f'\n{"="*60}')
    print(f' ENRICHMENT COMPLETE: {total_new:,} new examples')
    print(f'{"="*60}')
