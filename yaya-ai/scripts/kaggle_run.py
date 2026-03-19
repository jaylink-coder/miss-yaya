"""Single-script Kaggle runner — download, tokenize, and train."""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# Disable W&B — no interactive prompts
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

CHECKPOINT_DIR = '/kaggle/working/yaya-checkpoints'
TRAIN_DIR = 'data/processed/openwebtext/train'
EVAL_DIR  = 'data/processed/openwebtext/eval'

# ── 1. Download data ──────────────────────────────────────────────────────────
print('\n[1/3] Downloading OpenWebText (1% sample, streaming)...')
from datasets import load_dataset

# Use streaming=True to avoid downloading all 80 shards upfront
ds = load_dataset('openwebtext', split='train', streaming=True)

# ── 2. Tokenize ───────────────────────────────────────────────────────────────
print('\n[2/3] Tokenizing...')
from src.tokenizer.tokenizer import YayaTokenizer
tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
print(f'  Vocab size: {tokenizer.vocab_size}')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

def tokenize_streaming(dataset, path, filename, max_tokens, eval_tokens=0):
    """Stream from HuggingFace and write tokens to disk in chunks — zero OOM."""
    train_path = os.path.join(path, filename)
    eval_path  = os.path.join(EVAL_DIR, 'eval.bin') if eval_tokens else None

    total_train = 0
    total_eval  = 0
    chunk = []
    CHUNK = 500_000

    with open(train_path, 'wb') as ft, \
         (open(eval_path, 'wb') if eval_path else open(os.devnull, 'wb')) as fe:

        for i, row in enumerate(dataset):
            text = row.get('text', '').strip()
            if not text:
                continue
            toks = tokenizer.encode(text)
            chunk.extend(toks)

            while len(chunk) >= CHUNK:
                arr = np.array(chunk[:CHUNK], dtype=np.uint16)
                if total_eval < eval_tokens:
                    arr.tofile(fe)
                    total_eval += CHUNK
                else:
                    arr.tofile(ft)
                    total_train += CHUNK
                chunk = chunk[CHUNK:]

            if (i + 1) % 10000 == 0:
                print(f'  {i+1:,} docs | train {total_train:,} | eval {total_eval:,} tokens')

            if total_train >= max_tokens:
                break

        # flush remainder to train
        if chunk:
            arr = np.array(chunk, dtype=np.uint16)
            arr.tofile(ft)
            total_train += len(arr)

    print(f'  Train: {total_train:,} tokens → {train_path}')
    if eval_path:
        print(f'  Eval:  {total_eval:,} tokens → {eval_path}')
    return total_train

n_train = tokenize_streaming(
    ds, TRAIN_DIR, 'shard_00000.bin',
    max_tokens=20_000_000,   # 20M tokens — enough for real learning
    eval_tokens=500_000,
)
print(f'  Done: {n_train:,} train tokens')

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

os.system(f'WANDB_DISABLED=true WANDB_MODE=disabled python scripts/train.py '
          f'--model_config configs/model/yaya_125m.yaml '
          f'--train_config configs/training/train_125m.yaml '
          f'{resume}')
