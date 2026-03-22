"""Kaggle runner — tokenize Wikipedia (3B tokens) and train yaya-1B.

Re-run safe: skips tokenization if data already present (≥2B tokens),
auto-resumes from latest checkpoint.

Uses mixed multi-domain data if data/processed/mixed/ is available,
otherwise falls back to Wikipedia-only.
"""

import glob
import os
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE']     = 'disabled'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

WIKI_DIR       = '/kaggle/input/notebooks/takashisomeya/wikipedia-plaintext-20230801'
CHECKPOINT_DIR = '/kaggle/working/yaya-checkpoints-1b'
TRAIN_DIR      = 'data/processed/wikipedia/train'
EVAL_DIR       = 'data/processed/wikipedia/eval'
MIXED_DIR      = 'data/processed/mixed'
MAX_TRAIN_TOKENS = 3_000_000_000   # 3B tokens — full Wikipedia
EVAL_TOKENS      =     5_000_000   # 5M eval tokens
CHUNK            = 2_000_000       # 2M token write chunks
MIN_TOKENS       = 2_000_000_000   # require 2B tokens before calling tokenization done

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── 0. VRAM + dtype check ──────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    props      = torch.cuda.get_device_properties(0)
    vram_gb    = props.total_memory / 1e9
    n_gpus     = torch.cuda.device_count()
    capability = props.major * 10 + props.minor   # e.g. 75 for T4, 80 for A100
    # bfloat16 requires Ampere (8.0+); T4 is compute 7.5 — float16 only
    DTYPE = 'bfloat16' if capability >= 80 else 'float16'
    print(f'GPU:   {props.name}  VRAM: {vram_gb:.1f}GB × {n_gpus}')
    print(f'Compute capability: {props.major}.{props.minor}  → using {DTYPE}')
    total_vram = vram_gb * n_gpus
    if total_vram < 14:
        print(f'WARNING: Only {total_vram:.0f}GB total VRAM. May OOM with 1B model.')
    else:
        print(f'Total VRAM: {total_vram:.0f}GB — OK for 1B with 8-bit Adam')
else:
    print('WARNING: No GPU found. Training will be extremely slow.')
    DTYPE = 'float32'


# ── 1. Tokenize Wikipedia ──────────────────────────────────────────────────────
train_path = os.path.join(TRAIN_DIR, 'shard_00000.bin')
eval_path  = os.path.join(EVAL_DIR,  'eval.bin')

_needs_tokenize = True

if os.path.exists(train_path):
    size = os.path.getsize(train_path)
    tokens_on_disk = size // 2
    if size > MIN_TOKENS * 2:   # uint16: bytes = tokens × 2
        print(f'\n[1/2] Wikipedia already tokenized: {tokens_on_disk/1e9:.2f}B tokens — skipping.')
        _needs_tokenize = False
    else:
        print(f'\n[1/2] Undersized shard ({tokens_on_disk/1e6:.0f}M tokens, need 2B+) — re-tokenizing...')
        for f in glob.glob(os.path.join(TRAIN_DIR, '*.bin')):
            os.remove(f)
        if os.path.exists(eval_path):
            os.remove(eval_path)

if _needs_tokenize:
    print('\n[1/2] Tokenizing Wikipedia (~60 min on T4)...')
    t0 = time.time()

    from src.tokenizer.tokenizer import YayaTokenizer
    import pandas as pd

    tokenizer = YayaTokenizer('data/tokenizer/yaya_tokenizer.model')
    print(f'  Vocab size: {tokenizer.vocab_size}')

    # Accept any *.parquet files in the wiki dir
    parquet_files = sorted([
        os.path.join(WIKI_DIR, f)
        for f in os.listdir(WIKI_DIR)
        if f.endswith('.parquet')
    ])
    if not parquet_files:
        print(f'ERROR: No .parquet files found in {WIKI_DIR}')
        print('  Add the Kaggle Wikipedia dataset to this notebook.')
        sys.exit(1)
    print(f'  Found {len(parquet_files)} parquet files')

    total_train = 0
    total_eval  = 0
    chunk_buf   = []
    done        = False
    last_report = time.time()

    ft = open(train_path, 'wb')
    fe = open(eval_path,  'wb')

    try:
        for pfile in parquet_files:
            if done:
                break
            fname = os.path.basename(pfile)
            print(f'  {fname}...', flush=True)
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

            # Progress + ETA
            elapsed = time.time() - t0
            pct = min(total_train / MAX_TRAIN_TOKENS, 1.0)
            eta = (elapsed / max(pct, 1e-6)) * (1 - pct) / 60 if pct > 0 else 0
            print(
                f'    train {total_train/1e9:.2f}B | eval {total_eval/1e6:.0f}M'
                f' | {elapsed/60:.0f}min elapsed | ~{eta:.0f}min remaining',
                flush=True,
            )

        # Flush remaining
        if chunk_buf:
            arr = np.array(chunk_buf, dtype=np.uint16)
            arr.tofile(ft)
            total_train += len(arr)
    finally:
        ft.close()
        fe.close()

    elapsed = time.time() - t0
    print(f'\n  Tokenization done in {elapsed/60:.1f} min')
    print(f'  Train: {total_train/1e9:.2f}B tokens')
    print(f'  Eval:  {total_eval/1e6:.0f}M tokens')


# ── 2. Choose training data ────────────────────────────────────────────────────
# Prefer mixed multi-domain data if mix-pretrain-data was run
mixed_shards = glob.glob(os.path.join(MIXED_DIR, '*.bin'))
if mixed_shards:
    total_mixed_tokens = sum(os.path.getsize(f) for f in mixed_shards) // 2
    print(f'\n  Using mixed multi-domain data: {len(mixed_shards)} shards, '
          f'{total_mixed_tokens/1e9:.2f}B tokens')
    active_train_dir = MIXED_DIR
else:
    print(f'\n  Using Wikipedia data (run make mix-pretrain-data for multi-domain)')
    active_train_dir = TRAIN_DIR


# ── 3. Train ───────────────────────────────────────────────────────────────────
print('\n[2/2] Starting yaya-1B training...')
import yaml

# Write a temp config so we don't mutate the committed train_1b.yaml
with open('configs/training/train_1b.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['checkpointing']['save_dir']              = CHECKPOINT_DIR
cfg['data']['train_data']                     = active_train_dir
cfg['data']['eval_data']                      = EVAL_DIR
cfg['data']['num_workers']                    = 0        # Kaggle Jupyter: forked workers hang
cfg['training']['dtype']                      = DTYPE    # auto-detected above
cfg['logging']['log_steps']                   = 1        # log every optimizer step (not every 10)

# With n_gpus GPUs via DDP, effective batch = per_device_batch × n_gpus × grad_accum.
# Halve grad_accum per extra GPU so effective batch stays at 32 seqs (65K tokens/step).
cfg['training']['gradient_accumulation_steps'] = max(1, 32 // max(n_gpus, 1))
cfg['checkpointing']['keep_last_n']            = 1   # ~10GB/ckpt; Kaggle /working is 20GB

# Enable DDP when more than one GPU is available
if n_gpus > 1:
    cfg['distributed']['strategy'] = 'ddp'
    print(f'  DDP enabled: {n_gpus} GPUs → effective batch stays 32 seqs (grad_accum={cfg["training"]["gradient_accumulation_steps"]})')

tmp_cfg = tempfile.NamedTemporaryFile(
    mode='w', suffix='.yaml', dir='configs/training',
    prefix='_kaggle_run_', delete=False
)
yaml.dump(cfg, tmp_cfg)
tmp_cfg.close()
tmp_cfg_path = tmp_cfg.name

try:
    ckpts  = sorted(glob.glob(f'{CHECKPOINT_DIR}/checkpoint-*'))
    resume = f'--resume {ckpts[-1]}' if ckpts else ''
    print(f'  Resuming from: {ckpts[-1]}' if resume else '  Starting from scratch')
    print(f'  Train data:    {active_train_dir}')
    print(f'  Checkpoints:   {CHECKPOINT_DIR}')

    launcher = (
        f'torchrun --nproc_per_node={n_gpus} --master_port=29500'
        if n_gpus > 1 else 'python'
    )
    ret = os.system(
        f'WANDB_DISABLED=true WANDB_MODE=disabled {launcher} scripts/train.py '
        f'--model_config configs/model/yaya_1b.yaml '
        f'--train_config {tmp_cfg_path} '
        f'{resume}'
    )
    if ret != 0:
        print(f'\nERROR: train.py exited with code {ret}')
finally:
    os.unlink(tmp_cfg_path)
