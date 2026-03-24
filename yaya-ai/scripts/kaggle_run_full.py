"""
Kaggle full pipeline — pretrain Yaya-125M then SFT, in one 12-hour session.

Run this as a single Kaggle notebook cell:
    %run scripts/kaggle_run_full.py

What it does:
  Phase 1 — Pretrain (≈4-5 h on T4):
    • Streams ~50M tokens of OpenWebText from HuggingFace
    • Trains Yaya-125M from random init for 5 000 steps
    • Saves checkpoint to /kaggle/working/yaya-pretrain-checkpoints

  Phase 2 — SFT (≈3-4 h on T4):
    • Uses existing data/sft/yaya_instruct.jsonl (our Q&A + OpenHermes)
    • Fine-tunes from the pretrain checkpoint for 30 000 steps
    • Saves checkpoint to /kaggle/working/yaya-sft-checkpoints

After the run:
  • Download the final SFT checkpoint from the Kaggle output panel
  • Drop it into checkpoints/yaya-125m-sft/ locally
  • Test with: python scripts/generate.py --model_config configs/model/yaya_125m.yaml
                 --checkpoint checkpoints/yaya-125m-sft/checkpoint-XXXXXXXX
"""

import os, sys, glob, yaml, subprocess, shutil, numpy as np

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAIN_DIR   = '/kaggle/working/yaya-pretrain-checkpoints'
SFT_DIR        = '/kaggle/working/yaya-sft-checkpoints'
TRAIN_DATA_DIR = 'data/processed/openwebtext/train'
EVAL_DATA_DIR  = 'data/processed/openwebtext/eval'
SFT_DATA_PATH  = 'data/sft/yaya_instruct.jsonl'
EVAL_DATA_PATH = 'data/sft/yaya_instruct_eval.jsonl'
TOKENIZER_PATH = 'data/tokenizer/yaya_tokenizer.model'

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ── Redirect ALL HuggingFace caches to /kaggle/working (20 GB) ────────────────
# By default HF writes to /root/.cache which is on a tiny 5-8 GB partition.
HF_CACHE = '/kaggle/working/hf_cache'
os.makedirs(HF_CACHE, exist_ok=True)
os.environ['HF_HOME']                  = HF_CACHE
os.environ['HF_DATASETS_CACHE']        = HF_CACHE
os.environ['TRANSFORMERS_CACHE']       = HF_CACHE
os.environ['HUGGINGFACE_HUB_CACHE']    = HF_CACHE

os.environ['WANDB_DISABLED']           = 'true'
os.environ['WANDB_MODE']               = 'disabled'
os.environ['PYTORCH_CUDA_ALLOC_CONF']  = 'expandable_segments:True'
os.environ['PYTHONIOENCODING']         = 'utf-8'

def disk_free_gb(path='/kaggle/working'):
    st = shutil.disk_usage(path)
    return st.free / 1e9

def print_disk():
    print(f'  Disk free: {disk_free_gb():.1f} GB  (/kaggle/working)')

# Load HF token from Kaggle Secrets (add HF_TOKEN via Settings → Secrets)
if not os.environ.get('HF_TOKEN'):
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
        os.environ['HF_TOKEN'] = token
        print('HF_TOKEN loaded from Kaggle Secrets.')
    except Exception:
        print('WARNING: No HF_TOKEN found — HuggingFace downloads may be rate-limited.')

import torch

if torch.cuda.is_available():
    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    cap     = props.major * 10 + props.minor
    DTYPE   = 'bfloat16' if cap >= 80 else 'float16'
    print(f'GPU: {props.name}  VRAM: {vram_gb:.1f} GB  dtype: {DTYPE}')
else:
    DTYPE = 'float32'
    vram_gb = 0
    print('WARNING: No GPU — training will be extremely slow.')

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: PRETRAIN
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('PHASE 1: PRETRAINING on OpenWebText')
print('='*60)
print_disk()

existing_pretrain = sorted(glob.glob(f'{PRETRAIN_DIR}/checkpoint-*'))
if existing_pretrain:
    print(f'  Pretrain checkpoint already exists: {existing_pretrain[-1]}')
    print('  Skipping pretraining phase.')
    pretrain_ckpt = existing_pretrain[-1]
else:
    # Download and tokenize
    print('\n[1/2] Downloading OpenWebText (streaming, ~50M tokens)...')
    from datasets import load_dataset
    from src.tokenizer.tokenizer import YayaTokenizer

    tokenizer = YayaTokenizer(TOKENIZER_PATH)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(EVAL_DATA_DIR,  exist_ok=True)

    ds = load_dataset('openwebtext', split='train', streaming=True)

    train_path = os.path.join(TRAIN_DATA_DIR, 'shard_00000.bin')
    eval_path  = os.path.join(EVAL_DATA_DIR,  'eval.bin')
    MAX_TRAIN  = 50_000_000
    MAX_EVAL   = 500_000
    total_train = total_eval = 0
    chunk = []
    CHUNK = 500_000

    with open(train_path, 'wb') as ft, open(eval_path, 'wb') as fe:
        for i, row in enumerate(ds):
            text = row.get('text', '').strip()
            if not text:
                continue
            toks = tokenizer.encode(text)
            chunk.extend(toks)
            while len(chunk) >= CHUNK:
                arr = np.array(chunk[:CHUNK], dtype=np.uint16)
                if total_eval < MAX_EVAL:
                    arr.tofile(fe); total_eval += CHUNK
                else:
                    arr.tofile(ft); total_train += CHUNK
                chunk = chunk[CHUNK:]
            if (i + 1) % 10000 == 0:
                print(f'  {i+1:,} docs | train {total_train:,} tokens')
            if total_train >= MAX_TRAIN:
                break
        if chunk:
            arr = np.array(chunk, dtype=np.uint16)
            arr.tofile(ft); total_train += len(arr)

    print(f'  Tokenized: {total_train:,} train / {total_eval:,} eval tokens')

    # Patch train config
    print('\n[2/2] Starting pretraining (5 000 steps)...')
    with open('configs/training/train_125m.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['checkpointing']['save_dir']     = PRETRAIN_DIR
    cfg['checkpointing']['keep_last_n']  = 1          # only keep final checkpoint
    cfg['data']['train_data']            = TRAIN_DATA_DIR
    cfg['data']['eval_data']             = EVAL_DATA_DIR
    cfg['training']['max_steps']         = 5000
    cfg['training']['dtype']             = DTYPE
    cfg['data']['num_workers']           = 2
    with open('configs/training/train_125m.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    result = subprocess.run([
        sys.executable, 'scripts/train.py',
        '--model_config', 'configs/model/yaya_125m.yaml',
        '--train_config', 'configs/training/train_125m.yaml',
    ])
    if result.returncode != 0:
        print('ERROR: Pretraining failed. Check the log above.')
        sys.exit(result.returncode)

    existing_pretrain = sorted(glob.glob(f'{PRETRAIN_DIR}/checkpoint-*'))
    pretrain_ckpt = existing_pretrain[-1] if existing_pretrain else None
    print(f'  Pretraining done. Best checkpoint: {pretrain_ckpt}')

    # Free disk: delete tokenized data and HF cache — no longer needed
    print('  Cleaning up tokenized data and HF cache to free space...')
    shutil.rmtree(TRAIN_DATA_DIR, ignore_errors=True)
    shutil.rmtree(EVAL_DATA_DIR,  ignore_errors=True)
    shutil.rmtree(HF_CACHE, ignore_errors=True)
    print_disk()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: SFT
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('PHASE 2: SFT on Q&A + OpenHermes')
print('='*60)
print_disk()

# Create small eval split from SFT data if eval file missing
if not os.path.exists(EVAL_DATA_PATH):
    print('  Creating eval split (last 500 examples of yaya_instruct.jsonl)...')
    import json
    with open(SFT_DATA_PATH, encoding='utf-8') as f:
        lines = f.readlines()
    with open(EVAL_DATA_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines[-500:])
    print(f'  Eval split: {len(lines[-500:])} examples')

# Check how many SFT examples we have
with open(SFT_DATA_PATH) as f:
    n_sft = sum(1 for _ in f)
print(f'  SFT data: {n_sft:,} examples')

if n_sft < 50_000:
    print('  Fewer than 50k examples — downloading OpenHermes to supplement...')
    subprocess.run([
        sys.executable, 'scripts/download_openhermes.py',
        '--max_examples', '100000',
    ], check=False)

# Patch SFT config
with open('configs/training/sft_125m.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['checkpointing']['save_dir']              = SFT_DIR
cfg['checkpointing']['keep_last_n']           = 1     # save space — only keep final
cfg['training']['dtype']                      = DTYPE
cfg['data']['train_data']                     = SFT_DATA_PATH
cfg['data']['eval_data']                      = EVAL_DATA_PATH
cfg['data']['tokenizer_path']                 = TOKENIZER_PATH
cfg['distributed']['gradient_checkpointing']  = (vram_gb > 0 and vram_gb < 16)
with open('configs/training/sft_125m.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# Build command — resume from any existing SFT ckpt, else init from pretrain
sft_ckpts = sorted(glob.glob(f'{SFT_DIR}/checkpoint-*'))
cmd = [sys.executable, 'scripts/train_sft.py',
       '--model_config', 'configs/model/yaya_125m.yaml',
       '--train_config', 'configs/training/sft_125m.yaml']

if sft_ckpts:
    cmd += ['--resume', sft_ckpts[-1]]
    print(f'  Resuming SFT from: {sft_ckpts[-1]}')
elif pretrain_ckpt:
    cmd += ['--pretrain_checkpoint', pretrain_ckpt]
    print(f'  Initialising SFT from pretrain: {pretrain_ckpt}')
else:
    print('  WARNING: Starting SFT from random init (no pretrain checkpoint).')

print(f'\n  Command: {" ".join(cmd)}\n')
result = subprocess.run(cmd)

if result.returncode == 0:
    final = sorted(glob.glob(f'{SFT_DIR}/checkpoint-*'))[-1]
    print(f'\n{"="*60}')
    print('DONE! Final SFT checkpoint:')
    print(f'  {final}')
    print('\nDownload this folder from Kaggle Output and place it at:')
    print('  checkpoints/yaya-125m-sft/')
    print('Then test locally with:')
    print('  python scripts/generate.py \\')
    print('    --model_config configs/model/yaya_125m.yaml \\')
    print('    --checkpoint checkpoints/yaya-125m-sft/<checkpoint-name>')
    print('='*60)
else:
    print('ERROR: SFT failed. Check log above.')
    sys.exit(result.returncode)