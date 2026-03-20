"""Kaggle runner — tokenize Wikipedia parquet files and train yaya-125M."""

import os
import sys
import glob
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

WIKI_DIR    = '/kaggle/input/notebooks/takashisomeya/wikipedia-plaintext-20230801'
CHECKPOINT_DIR = '/kaggle/working/yaya-checkpoints'
TRAIN_DIR   = 'data/processed/wikipedia/train'
EVAL_DIR    = 'data/processed/wikipedia/eval'
MAX_TRAIN_TOKENS = 500_000_000   # 500M tokens (~7,600 steps, no repetition)
EVAL_TOKENS      =   2_000_000   # 2M eval tokens

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── 1. Tokenize ────────────────────────────────────────────────────────────────
print('\n[1/2] Tokenizing Wikipedia...')
from src.tokenizer.tokenizer import YayaTokenizer
tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
print(f'  Vocab size: {tokenizer.vocab_size}')

import pandas as pd

parquet_files = sorted([
    os.path.join(WIKI_DIR, f)
    for f in os.listdir(WIKI_DIR)
    if f.endswith('.parquet') and len(f) == 9  # single-letter files: a.parquet etc
])
print(f'  Found {len(parquet_files)} parquet files')

CHUNK = 1_000_000  # write 1M tokens at a time

train_path = os.path.join(TRAIN_DIR, 'shard_00000.bin')
eval_path  = os.path.join(EVAL_DIR,  'eval.bin')

total_train = 0
total_eval  = 0
chunk_buf   = []
done        = False

ft = open(train_path, 'wb')
fe = open(eval_path,  'wb')

try:
    for pfile in parquet_files:
        if done:
            break
        letter = os.path.basename(pfile)
        print(f'  Processing {letter}...')
        df = pd.read_parquet(pfile, columns=['text'])

        for i, row in enumerate(df.itertuples(index=False)):
            text = str(row.text).strip()
            if not text or len(text) < 50:
                continue
            toks = tokenizer.encode(text)
            chunk_buf.extend(toks)

            while len(chunk_buf) >= CHUNK:
                arr = np.array(chunk_buf[:CHUNK], dtype=np.uint16)
                if total_eval < EVAL_TOKENS:
                    arr.tofile(fe)
                    total_eval += CHUNK
                else:
                    arr.tofile(ft)
                    total_train += CHUNK
                chunk_buf = chunk_buf[CHUNK:]

            if total_train >= MAX_TRAIN_TOKENS:
                done = True
                break

        print(f'    train {total_train/1e6:.1f}M | eval {total_eval/1e6:.1f}M tokens')

    # flush remainder
    if chunk_buf and not done:
        arr = np.array(chunk_buf, dtype=np.uint16)
        arr.tofile(ft)
        total_train += len(arr)

finally:
    ft.close()
    fe.close()

print(f'\n  Train: {total_train:,} tokens → {train_path}')
print(f'  Eval:  {total_eval:,} tokens  → {eval_path}')

# ── 2. Update config and train ─────────────────────────────────────────────────
print('\n[2/2] Starting training...')
import yaml

with open('configs/training/train_125m.yaml') as f:
    cfg = yaml.safe_load(f)

# Override only path fields — never touch learning_rate or other hyperparams
cfg['checkpointing']['save_dir'] = CHECKPOINT_DIR
cfg['data']['train_data'] = TRAIN_DIR
cfg['data']['eval_data']  = EVAL_DIR

# Safety: clamp LR to a stable value regardless of what's in the file
if cfg['training']['learning_rate'] > 3e-5:
    print(f"  [WARN] LR {cfg['training']['learning_rate']} too high — clamping to 2e-5")
    cfg['training']['learning_rate'] = 2e-5

with open('configs/training/train_125m.yaml', 'w') as f:
    yaml.dump(cfg, f)

print(f"  LR = {cfg['training']['learning_rate']}  warmup = {cfg['training']['warmup_steps']}")

ckpts  = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint-*'))
resume = f'--resume {ckpts[-1]}' if ckpts else ''
if resume:
    print(f'  Resuming from: {ckpts[-1]}')
else:
    print('  Starting from scratch')

os.system(
    f'WANDB_DISABLED=true WANDB_MODE=disabled python scripts/train.py '
    f'--model_config configs/model/yaya_125m.yaml '
    f'--train_config configs/training/train_125m.yaml '
    f'{resume}'
)
