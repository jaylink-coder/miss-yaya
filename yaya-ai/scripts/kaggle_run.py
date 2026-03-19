"""Single-script Kaggle runner — download, tokenize, and train."""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

CHECKPOINT_DIR = '/kaggle/working/yaya-checkpoints'
TRAIN_DIR = 'data/processed/openwebtext/train'
EVAL_DIR  = 'data/processed/openwebtext/eval'

# ── 1. Download data ──────────────────────────────────────────────────────────
print('\n[1/3] Downloading dataset...')
from datasets import load_dataset

# Use wikitext — downloads in seconds, no memory issues
ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
print(f'  Loaded {len(ds):,} documents')

# ── 2. Tokenize ───────────────────────────────────────────────────────────────
print('\n[2/3] Tokenizing...')
from src.tokenizer.tokenizer import YayaTokenizer
tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
print(f'  Vocab size: {tokenizer.vocab_size}')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

split    = ds.train_test_split(test_size=0.005, seed=42)
train_ds = split['train']
eval_ds  = split['test']

def tokenize_and_save(dataset, path, filename, max_tokens=30_000_000):
    """Stream tokens directly to disk in chunks — no OOM."""
    out_path = os.path.join(path, filename)
    total = 0
    chunk = []
    CHUNK_SIZE = 500_000

    with open(out_path, 'wb') as f:
        for i, row in enumerate(dataset):
            text = row.get('text', '').strip()
            if not text:
                continue
            chunk.extend(tokenizer.encode(text))
            if len(chunk) >= CHUNK_SIZE:
                arr = np.array(chunk, dtype=np.uint16)
                arr.tofile(f)
                total += len(arr)
                chunk = []
                print(f'  {i+1:,} docs, {total:,} tokens saved...')
            if total >= max_tokens:
                break
        if chunk:
            arr = np.array(chunk, dtype=np.uint16)
            arr.tofile(f)
            total += len(arr)

    print(f'  Done: {total:,} tokens → {out_path}')
    return total

n_train = tokenize_and_save(train_ds, TRAIN_DIR, 'shard_00000.bin', max_tokens=30_000_000)
n_eval  = tokenize_and_save(eval_ds,  EVAL_DIR,  'eval.bin',        max_tokens=1_000_000)
print(f'  Total: {n_train:,} train | {n_eval:,} eval tokens')

# ── 3. Train ──────────────────────────────────────────────────────────────────
print('\n[3/3] Starting training...')
import glob, yaml

with open('configs/training/train_125m.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['checkpointing']['save_dir'] = CHECKPOINT_DIR
with open('configs/training/train_125m.yaml', 'w') as f:
    yaml.dump(cfg, f)

ckpts  = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint-*'))
resume = f'--resume {ckpts[-1]}' if ckpts else ''
if resume:
    print(f'  Resuming from: {ckpts[-1]}')
else:
    print('  Starting from scratch')

os.system(f'python scripts/train.py '
          f'--model_config configs/model/yaya_125m.yaml '
          f'--train_config configs/training/train_125m.yaml '
          f'{resume}')
