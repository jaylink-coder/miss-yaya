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
print('\n[1/3] Downloading OpenWebText (2% sample)...')
from datasets import load_dataset
ds = load_dataset('openwebtext', split='train[:2%]')
print(f'  Loaded {len(ds):,} documents')

# ── 2. Tokenize ───────────────────────────────────────────────────────────────
print('\n[2/3] Tokenizing...')
from src.tokenizer.tokenizer import YayaTokenizer
tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
print(f'  Vocab size: {tokenizer.vocab_size}')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

split    = ds.train_test_split(test_size=0.01, seed=42)
train_ds = split['train']
eval_ds  = split['test']

def tokenize_and_save(dataset, path, filename):
    tokens = []
    for i, row in enumerate(dataset):
        tokens.extend(tokenizer.encode(row['text']))
        if (i + 1) % 5000 == 0:
            print(f'  {i+1:,} docs, {len(tokens):,} tokens')
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(os.path.join(path, filename))
    print(f'  Saved {len(arr):,} tokens → {path}/{filename}')
    return len(arr)

n_train = tokenize_and_save(train_ds, TRAIN_DIR, 'shard_00000.bin')
n_eval  = tokenize_and_save(eval_ds,  EVAL_DIR,  'eval.bin')
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
