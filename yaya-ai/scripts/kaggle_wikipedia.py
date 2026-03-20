"""Kaggle runner — tokenize Wikipedia parquet files and train yaya-125M."""

import os
import sys
import glob
import random
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

WIKI_DIR       = '/kaggle/input/notebooks/takashisomeya/wikipedia-plaintext-20230801'
CHECKPOINT_DIR = '/kaggle/working/yaya-checkpoints'
TRAIN_DIR      = 'data/processed/wikipedia/train'
EVAL_DIR       = 'data/processed/wikipedia/eval'
MAX_TRAIN_TOKENS  = 500_000_000   # 500M tokens (~7,600 steps)
EVAL_TOKENS       =   2_000_000   # 2M eval tokens
TOKENS_PER_SHARD  =  20_000_000   # 20M tokens per shard → ~25 shards for 500M
CHUNK             =   1_000_000   # buffer 1M tokens before writing

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── 1. Tokenize (skip if enough shards already exist) ──────────────────────────
eval_path = os.path.join(EVAL_DIR, 'eval.bin')

existing_shards = sorted([
    f for f in os.listdir(TRAIN_DIR)
    if f.startswith('shard_') and f.endswith('.bin')
])

if len(existing_shards) >= 3:
    total_tok = sum(
        os.path.getsize(os.path.join(TRAIN_DIR, s)) // 2
        for s in existing_shards
    )
    print(f'\n[1/2] Tokenization already done: {len(existing_shards)} shards, '
          f'{total_tok/1e6:.0f}M tokens. Skipping.', flush=True)
else:
    print('\n[1/2] Tokenizing Wikipedia into shuffled shards...', flush=True)
    from src.tokenizer.tokenizer import YayaTokenizer
    import pandas as pd

    tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
    print(f'  Vocab size: {tokenizer.vocab_size}', flush=True)

    # SHUFFLE parquet file order so data is NOT alphabetical
    parquet_files = [
        os.path.join(WIKI_DIR, f)
        for f in os.listdir(WIKI_DIR)
        if f.endswith('.parquet') and len(f) == 9
    ]
    random.seed(42)
    random.shuffle(parquet_files)
    print(f'  Found {len(parquet_files)} parquet files (shuffled)', flush=True)
    print(f'  Order: {[os.path.basename(p) for p in parquet_files]}', flush=True)

    total_train  = 0
    total_eval   = 0
    chunk_buf    = []
    done         = False
    shard_idx    = 0
    shard_tokens = 0

    def open_shard(idx):
        path = os.path.join(TRAIN_DIR, f'shard_{idx:05d}.bin')
        return open(path, 'wb'), path

    ft, cur_shard = open_shard(shard_idx)
    fe = open(eval_path, 'wb')

    try:
        for pfile in parquet_files:
            if done:
                break
            letter = os.path.basename(pfile)
            print(f'  Processing {letter}...', flush=True)
            df = pd.read_parquet(pfile, columns=['text'])

            for row in df.itertuples(index=False):
                text = str(row.text).strip()
                if not text or len(text) < 50:
                    continue
                chunk_buf.extend(tokenizer.encode(text))

                while len(chunk_buf) >= CHUNK:
                    arr = np.array(chunk_buf[:CHUNK], dtype=np.uint16)
                    chunk_buf = chunk_buf[CHUNK:]

                    if total_eval < EVAL_TOKENS:
                        arr.tofile(fe)
                        total_eval += CHUNK
                    else:
                        arr.tofile(ft)
                        total_train  += CHUNK
                        shard_tokens += CHUNK
                        if shard_tokens >= TOKENS_PER_SHARD:
                            ft.close()
                            print(f'    Shard {shard_idx}: {shard_tokens/1e6:.0f}M tokens', flush=True)
                            shard_idx   += 1
                            shard_tokens = 0
                            ft, cur_shard = open_shard(shard_idx)

                if total_train >= MAX_TRAIN_TOKENS:
                    done = True
                    break

            print(f'    train {total_train/1e6:.1f}M | eval {total_eval/1e6:.1f}M tokens', flush=True)

        if chunk_buf:
            arr = np.array(chunk_buf, dtype=np.uint16)
            arr.tofile(ft)
            total_train += len(arr)

    finally:
        ft.close()
        fe.close()

    print(f'\n  Train: {total_train:,} tokens across {shard_idx + 1} shards', flush=True)
    print(f'  Eval:  {total_eval:,} tokens → {eval_path}', flush=True)

# ── 2. Update config and train ─────────────────────────────────────────────────
print('\n[2/2] Starting training...', flush=True)
import yaml

with open('configs/training/train_125m.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['checkpointing']['save_dir'] = CHECKPOINT_DIR
cfg['data']['train_data'] = TRAIN_DIR
cfg['data']['eval_data']  = EVAL_DIR
cfg['training']['learning_rate'] = 2e-5   # safe, stable LR
cfg['training']['warmup_steps']  = 1000   # 13% of 7600 steps

with open('configs/training/train_125m.yaml', 'w') as f:
    yaml.dump(cfg, f)

print(f"  LR={cfg['training']['learning_rate']}  warmup={cfg['training']['warmup_steps']}  "
      f"steps={cfg['training']['max_steps']}", flush=True)

ckpts  = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint-*'))
resume = f'--resume {ckpts[-1]}' if ckpts else ''
print(f'  Resuming from: {ckpts[-1]}' if ckpts else '  Starting from scratch', flush=True)

os.system(
    f'WANDB_DISABLED=true WANDB_MODE=disabled python -u scripts/train.py '
    f'--model_config configs/model/yaya_125m.yaml '
    f'--train_config configs/training/train_125m.yaml '
    f'{resume}'
)
