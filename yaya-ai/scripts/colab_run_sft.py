"""Single-script Colab SFT runner — download data, then fine-tune.

Run this AFTER pretraining is done (colab_run.py).

Usage (Colab cell):
    !python /content/miss-yaya/yaya-ai/scripts/colab_run_sft.py
"""

import os
import sys
import json
import glob
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

random.seed(42)

PRETRAIN_CKPT_DIR = '/content/drive/MyDrive/yaya-checkpoints'
SFT_CKPT_DIR      = '/content/drive/MyDrive/yaya-sft-checkpoints'
SFT_DATA_PATH     = 'data/sft/yaya_instruct.jsonl'

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly, tell jokes when asked, and are always honest."
)


def make_sample(user_msg, assistant_msg):
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ── 1. Load hand-crafted data ─────────────────────────────────────────────────
print('\n[1/4] Loading hand-crafted SFT examples...')
samples = []
if os.path.exists(SFT_DATA_PATH):
    with open(SFT_DATA_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
print(f'  Loaded {len(samples)} hand-crafted examples')


# ── 2. Download Alpaca ────────────────────────────────────────────────────────
print('\n[2/4] Downloading Alpaca (300 examples)...')
try:
    from datasets import load_dataset
    ds = load_dataset('tatsu-lab/alpaca', split='train')
    added = 0
    for row in ds:
        instruction = row.get('instruction', '').strip()
        inp         = row.get('input', '').strip()
        output      = row.get('output', '').strip()
        if not instruction or not output:
            continue
        user_msg = f'{instruction}\n\n{inp}' if inp else instruction
        samples.append(make_sample(user_msg, output))
        added += 1
        if added >= 300:
            break
    print(f'  Got {added} Alpaca examples')
except Exception as e:
    print(f'  Alpaca download failed: {e}')


# ── 3. Download Dolly ─────────────────────────────────────────────────────────
print('\n[3/4] Downloading Dolly (200 examples)...')
try:
    ds = load_dataset('databricks/databricks-dolly-15k', split='train')
    added = 0
    for row in ds:
        instruction = row.get('instruction', '').strip()
        context     = row.get('context', '').strip()
        response    = row.get('response', '').strip()
        if not instruction or not response:
            continue
        user_msg = f'{instruction}\n\nContext: {context}' if context else instruction
        samples.append(make_sample(user_msg, response))
        added += 1
        if added >= 200:
            break
    print(f'  Got {added} Dolly examples')
except Exception as e:
    print(f'  Dolly download failed: {e}')


# Deduplicate and shuffle
seen, deduped = set(), []
for s in samples:
    key = s['messages'][1]['content'][:80]
    if key not in seen:
        seen.add(key)
        deduped.append(s)
random.shuffle(deduped)

os.makedirs('data/sft', exist_ok=True)
with open(SFT_DATA_PATH, 'w', encoding='utf-8') as f:
    for s in deduped:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f'  Total SFT examples: {len(deduped)}')


# ── 4. Run SFT training ───────────────────────────────────────────────────────
print('\n[4/4] Starting SFT training...')

# Find the latest pretrain checkpoint
ckpts = sorted(glob.glob(f'{PRETRAIN_CKPT_DIR}/checkpoint-*'))
if not ckpts:
    print(f'ERROR: No pretrain checkpoint found in {PRETRAIN_CKPT_DIR}')
    print('Make sure pretraining has run and Drive is mounted.')
    sys.exit(1)

best_ckpt = ckpts[-1]
print(f'  Using pretrain checkpoint: {best_ckpt}')

# Update SFT config to point to Drive
import yaml
with open('configs/training/sft_125m.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['checkpointing']['save_dir'] = SFT_CKPT_DIR
with open('configs/training/sft_125m.yaml', 'w') as f:
    yaml.dump(cfg, f)

# Check for SFT resume checkpoint
sft_ckpts = sorted(glob.glob(f'{SFT_CKPT_DIR}/checkpoint-*'))
resume = f'--resume {sft_ckpts[-1]}' if sft_ckpts else ''
if resume:
    print(f'  Resuming SFT from: {sft_ckpts[-1]}')
    pretrain_flag = ''
else:
    pretrain_flag = f'--pretrain_checkpoint {best_ckpt}'

os.system(
    f'python scripts/train_sft.py '
    f'--model_config configs/model/yaya_125m.yaml '
    f'--train_config configs/training/sft_125m.yaml '
    f'{pretrain_flag} {resume}'
)
