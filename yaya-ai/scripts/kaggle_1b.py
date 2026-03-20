"""Kaggle runner — tokenize ALL Wikipedia (3B tokens) and train yaya-1B."""

import os, sys, glob
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE']     = 'disabled'

WIKI_DIR       = '/kaggle/input/notebooks/takashisomeya/wikipedia-plaintext-20230801'
CHECKPOINT_DIR = '/kaggle/working/yaya-checkpoints-1b'
TRAIN_DIR      = 'data/processed/wikipedia/train'
EVAL_DIR       = 'data/processed/wikipedia/eval'
MAX_TRAIN_TOKENS = 3_000_000_000   # 3B tokens — full Wikipedia
EVAL_TOKENS      =     5_000_000   # 5M eval tokens
CHUNK            = 2_000_000       # write 2M tokens at a time

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── 1. Tokenize ────────────────────────────────────────────────────────────────
train_path = os.path.join(TRAIN_DIR, 'shard_00000.bin')
eval_path  = os.path.join(EVAL_DIR,  'eval.bin')

# Skip tokenization if already done
if os.path.exists(train_path) and os.path.getsize(train_path) > 100_000_000:
    tokens_on_disk = os.path.getsize(train_path) // 2
    print(f'\n[1/2] Tokenization already done: {tokens_on_disk/1e9:.2f}B tokens on disk. Skipping.')
else:
    print('\n[1/2] Tokenizing ALL Wikipedia (this takes ~60 minutes)...')
    from src.tokenizer.tokenizer import YayaTokenizer
    import pandas as pd

    tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
    print(f'  Vocab size: {tokenizer.vocab_size}')

    parquet_files = sorted([
        os.path.join(WIKI_DIR, f)
        for f in os.listdir(WIKI_DIR)
        if f.endswith('.parquet') and len(f) == 9
    ])
    print(f'  Found {len(parquet_files)} parquet files (~10.9M articles)')

    total_train = 0
    total_eval  = 0
    chunk_buf   = []
    done        = False

    ft = open(train_path, 'wb')
    fe = open(eval_path,  'wb')

    try:
        for pfile in parquet_files:
            if done: break
            letter = os.path.basename(pfile)
            print(f'  Processing {letter}...', flush=True)
            df = pd.read_parquet(pfile, columns=['text'])

            for row in df.itertuples(index=False):
                text = str(row.text).strip()
                if not text or len(text) < 100:
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

            gb = total_train / 1e9
            print(f'    train {gb:.2f}B | eval {total_eval/1e6:.0f}M tokens', flush=True)

        if chunk_buf:
            arr = np.array(chunk_buf, dtype=np.uint16)
            arr.tofile(ft)
            total_train += len(arr)
    finally:
        ft.close()
        fe.close()

    print(f'\n  Train: {total_train/1e9:.2f}B tokens')
    print(f'  Eval:  {total_eval/1e6:.0f}M tokens')

# ── 2. Train ───────────────────────────────────────────────────────────────────
print('\n[2/2] Starting yaya-1B training...')
import yaml

with open('configs/training/train_1b.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['checkpointing']['save_dir'] = CHECKPOINT_DIR
cfg['data']['train_data'] = TRAIN_DIR
cfg['data']['eval_data']  = EVAL_DIR
with open('configs/training/train_1b.yaml', 'w') as f:
    yaml.dump(cfg, f)

ckpts  = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint-*'))
resume = f'--resume {ckpts[-1]}' if ckpts else ''
print(f'  Resuming from: {ckpts[-1]}' if resume else '  Starting from scratch')

os.system(
    f'WANDB_DISABLED=true WANDB_MODE=disabled python scripts/train.py '
    f'--model_config configs/model/yaya_1b.yaml '
    f'--train_config configs/training/train_1b.yaml '
    f'{resume}'
)
