"""Kaggle runner using Wikipedia dataset input — no downloads needed.

Attach the Wikipedia dataset in Kaggle:
  Add Data → Search "wikipedia" → add "Wikipedia 20220301 EN" (or similar)
  It mounts at /kaggle/input/wikipedia-20220301-en/

Then run this script instead of kaggle_run.py
"""

import os
import sys
import glob
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE']     = 'disabled'

CHECKPOINT_DIR = '/kaggle/working/yaya-checkpoints'
TRAIN_DIR = 'data/processed/openwebtext/train'
EVAL_DIR  = 'data/processed/openwebtext/eval'

# ── 1. Find Wikipedia input data ──────────────────────────────────────────────
print('\n[1/3] Finding Wikipedia data...')

# Try common Kaggle Wikipedia dataset locations
wiki_files = (
    glob.glob('/kaggle/input/wikipedia*/*/train*')          +
    glob.glob('/kaggle/input/wikipedia*/**/*.parquet')      +
    glob.glob('/kaggle/input/wikipedia*/**/*.jsonl')        +
    glob.glob('/kaggle/input/wikipedia*/**/*.json')         +
    glob.glob('/kaggle/input/*wiki*/**/*.parquet')
)

if not wiki_files:
    print('  No Wikipedia dataset found — searching all inputs...')
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f.endswith(('.parquet', '.jsonl', '.json', '.txt')):
                wiki_files.append(os.path.join(root, f))
                if len(wiki_files) >= 20:
                    break
        if len(wiki_files) >= 20:
            break

print(f'  Found {len(wiki_files)} data files')
for f in wiki_files[:5]:
    print(f'    {f}')

if not wiki_files:
    print('\nERROR: No input dataset found.')
    print('In Kaggle: click "+ Add Data" → search "wikipedia" → add dataset')
    sys.exit(1)

# ── 2. Tokenize ───────────────────────────────────────────────────────────────
print('\n[2/3] Tokenizing...')
from src.tokenizer.tokenizer import YayaTokenizer
tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
print(f'  Vocab size: {tokenizer.vocab_size}')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

TARGET_TRAIN = 500_000_000   # 500M tokens
TARGET_EVAL  =   2_000_000   #   2M tokens
CHUNK_SIZE   =   1_000_000

train_path = os.path.join(TRAIN_DIR, 'shard_00000.bin')
eval_path  = os.path.join(EVAL_DIR,  'eval.bin')

total_train = 0
total_eval  = 0
chunk       = []

def flush(f, tokens, target_remaining):
    """Write chunk to file, returns tokens written."""
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(f)
    return len(arr)

def iter_texts(files):
    """Yield text strings from parquet/jsonl/json/txt files."""
    for fpath in files:
        ext = fpath.split('.')[-1].lower()
        try:
            if ext == 'parquet':
                import pandas as pd
                df = pd.read_parquet(fpath)
                text_col = next((c for c in ['text','content','passage','abstract']
                                 if c in df.columns), df.columns[0])
                for text in df[text_col].dropna():
                    yield str(text)

            elif ext in ('jsonl', 'json'):
                import json
                with open(fpath, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        obj = json.loads(line)
                        text = obj.get('text') or obj.get('content') or obj.get('abstract','')
                        if text: yield text

            elif ext == 'txt':
                with open(fpath, encoding='utf-8', errors='ignore') as f:
                    yield f.read()
        except Exception as e:
            print(f'  Skipping {fpath}: {e}')
            continue

with open(train_path, 'wb') as ft, open(eval_path, 'wb') as fe:
    for i, text in enumerate(iter_texts(wiki_files)):
        if not text.strip():
            continue
        chunk.extend(tokenizer.encode(text))

        while len(chunk) >= CHUNK_SIZE:
            batch = chunk[:CHUNK_SIZE]
            chunk = chunk[CHUNK_SIZE:]
            if total_eval < TARGET_EVAL:
                total_eval += flush(fe, batch, TARGET_EVAL - total_eval)
            else:
                total_train += flush(ft, batch, TARGET_TRAIN - total_train)

        if (i + 1) % 50000 == 0:
            print(f'  {i+1:,} docs | train {total_train/1e6:.1f}M | eval {total_eval/1e6:.1f}M tokens')

        if total_train >= TARGET_TRAIN:
            break

    # flush remainder
    if chunk:
        if total_eval < TARGET_EVAL:
            total_eval += flush(fe, chunk, TARGET_EVAL - total_eval)
        else:
            total_train += flush(ft, chunk, TARGET_TRAIN - total_train)

print(f'\n  Train: {total_train/1e6:.1f}M tokens')
print(f'  Eval:  {total_eval/1e6:.1f}M tokens')

# ── 3. Train ──────────────────────────────────────────────────────────────────
print('\n[3/3] Starting training...')
import yaml

with open('configs/training/train_125m.yaml') as f:
    cfg = yaml.safe_load(f)

# Scale steps to match data: ~1.5 epochs of 500M tokens
steps_per_epoch = total_train // (8 * 1024 * 8)   # batch=8, seq=1024, grad_acc=8
max_steps = int(steps_per_epoch * 1.5)
cfg['checkpointing']['save_dir'] = CHECKPOINT_DIR
cfg['training']['max_steps']     = max_steps
cfg['training']['warmup_steps']  = min(2000, max_steps // 10)
cfg['training']['learning_rate'] = 3.0e-4

print(f'  Steps per epoch: {steps_per_epoch:,}')
print(f'  Training for: {max_steps:,} steps')

with open('configs/training/train_125m.yaml', 'w') as f:
    yaml.dump(cfg, f)

ckpts  = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint-*'))
resume = f'--resume {ckpts[-1]}' if ckpts else ''
if resume:
    print(f'  Resuming from: {ckpts[-1]}')

os.system(
    f'WANDB_DISABLED=true WANDB_MODE=disabled python scripts/train.py '
    f'--model_config configs/model/yaya_125m.yaml '
    f'--train_config configs/training/train_125m.yaml '
    f'{resume}'
)
