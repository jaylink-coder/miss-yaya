"""Kaggle SFT runner for Yaya-1B.

Loads OpenHermes + hand-crafted data, then fine-tunes the 1B pretrain checkpoint.
Run AFTER kaggle_1b.py has produced a checkpoint.

Usage:
    python scripts/kaggle_sft_1b.py
"""

import os
import sys
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

PRETRAIN_CKPT_DIR = '/kaggle/working/yaya-checkpoints-1b'
SFT_CKPT_DIR      = '/kaggle/working/yaya-sft-1b-checkpoints'

# ── 1. Check pretrain checkpoint ───────────────────────────────────────────────
print('\n[1/3] Looking for pretrain checkpoint...')
ckpts = sorted(glob.glob(f'{PRETRAIN_CKPT_DIR}/checkpoint-*'))
if not ckpts:
    print(f'ERROR: No pretrain checkpoint in {PRETRAIN_CKPT_DIR}')
    print('Run kaggle_1b.py first.')
    sys.exit(1)
best_ckpt = ckpts[-1]
print(f'  Using: {best_ckpt}')

# ── 2. Pull OpenHermes if not already downloaded ───────────────────────────────
import os
openhermes_path = 'data/sft/openhermes.jsonl'

if os.path.exists(openhermes_path):
    import json
    with open(openhermes_path) as f:
        n = sum(1 for line in f if line.strip())
    print(f'\n[2/3] OpenHermes already downloaded: {n:,} examples — skipping.')
else:
    print('\n[2/3] Downloading OpenHermes 100K examples...')
    os.system('python scripts/download_openhermes.py --max_examples 100000')

# ── 3. Run SFT ─────────────────────────────────────────────────────────────────
print('\n[3/3] Starting 1B SFT...')

import tempfile
import torch
import yaml

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

with open('configs/training/sft_1b.yaml') as f:
    cfg = yaml.safe_load(f)

# Detect dtype: T4 (cc 7.5) → float16; Ampere (cc 8.0+) → bfloat16
if torch.cuda.is_available():
    cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
    cfg['training']['dtype'] = 'bfloat16' if cc >= 80 else 'float16'

cfg['checkpointing']['save_dir'] = SFT_CKPT_DIR
cfg['data']['num_workers']       = 0   # Kaggle Jupyter: forked workers hang
cfg['logging']['log_steps']      = 1   # Log every step so progress is visible

# Enable DDP when multiple GPUs available; halve grad_accum to keep effective batch constant
if n_gpus > 1:
    cfg['distributed']['strategy'] = 'ddp'
    cfg['training']['gradient_accumulation_steps'] = max(
        1, cfg['training']['gradient_accumulation_steps'] // n_gpus
    )
    print(f'  DDP enabled: {n_gpus} GPUs')

# Write temp config — never mutate the committed sft_1b.yaml
tmp = tempfile.NamedTemporaryFile(
    mode='w', suffix='.yaml', dir='configs/training',
    prefix='_kaggle_sft_', delete=False
)
yaml.dump(cfg, tmp)
tmp.close()
tmp_path = tmp.name

# Resume SFT if interrupted
sft_ckpts = sorted(glob.glob(f'{SFT_CKPT_DIR}/checkpoint-*'))
if sft_ckpts:
    resume_flag   = f'--resume {sft_ckpts[-1]}'
    pretrain_flag = ''
    print(f'  Resuming SFT from: {sft_ckpts[-1]}')
else:
    resume_flag   = ''
    pretrain_flag = f'--pretrain_checkpoint {best_ckpt}'
    print(f'  Starting SFT from pretrain checkpoint.')

try:
    launcher = (
        f'torchrun --nproc_per_node={n_gpus} --master_port=29500'
        if n_gpus > 1 else 'python'
    )
    ret = os.system(
        f'WANDB_DISABLED=true WANDB_MODE=disabled {launcher} scripts/train_sft.py '
        f'--model_config configs/model/yaya_1b.yaml '
        f'--train_config {tmp_path} '
        f'{pretrain_flag} {resume_flag}'
    )
    if ret != 0:
        print(f'\nERROR: train_sft.py exited with code {ret}')
finally:
    os.unlink(tmp_path)
