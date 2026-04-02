"""
Kaggle SFT runner — downloads 100K OpenHermes, then fine-tunes Yaya-125M.

Setup (one-time in Kaggle):
  1. Add your GitHub repo as a Kaggle dataset or clone it via internet access
  2. Add HF_TOKEN as a Kaggle Secret (Settings → Secrets)
  3. Add WANDB_API_KEY as a Kaggle Secret (optional)
  4. If you have a pretrain checkpoint, upload it as a dataset and set
     PRETRAIN_CKPT_DATASET below

Re-run safe: auto-resumes SFT from latest checkpoint if one exists.
"""

import os
import sys
import glob
import subprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_CKPT_DIR   = '/kaggle/working/yaya-sft-checkpoints'
SFT_DATA_PATH  = 'data/sft/yaya_instruct.jsonl'
EVAL_DATA_PATH = 'data/sft/yaya_instruct_eval.jsonl'
TOKENIZER_PATH = 'data/tokenizer/yaya_tokenizer.model'

# Pretrain checkpoint: look in uploaded Kaggle dataset first, then working dir
PRETRAIN_CKPT_DATASET = '/kaggle/input/yaya-checkpoints'   # upload checkpoint here
PRETRAIN_CKPT_WORKING = '/kaggle/working/yaya-checkpoints'

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# Load WANDB_API_KEY from Kaggle secrets if not already set
if not os.environ.get('WANDB_API_KEY'):
    try:
        from kaggle_secrets import UserSecretsClient
        wandb_key = UserSecretsClient().get_secret('WANDB_API_KEY')
        os.environ['WANDB_API_KEY'] = wandb_key
        print('WANDB_API_KEY loaded from Kaggle secrets.')
    except Exception:
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE']     = 'disabled'
        print('No WANDB_API_KEY found — W&B disabled.')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONIOENCODING']        = 'utf-8'

# ── 0. GPU check ──────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    cap     = props.major * 10 + props.minor
    DTYPE   = 'bfloat16' if cap >= 80 else 'float16'
    print(f'GPU:   {props.name}  VRAM: {vram_gb:.1f}GB')
    print(f'Compute capability: {props.major}.{props.minor}  → {DTYPE}')
    if vram_gb < 12:
        print('WARNING: <12GB VRAM — consider reducing batch size or seq length.')
else:
    print('WARNING: No GPU found. Training will be very slow.')
    DTYPE = 'float32'

# ── 1. HF Token ───────────────────────────────────────────────────────────────
print('\n[1/4] Setting up HuggingFace token...')
hf_token = os.environ.get('HF_TOKEN', '')
if not hf_token:
    # Try Kaggle secrets via kaggle_secrets
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret('HF_TOKEN')
        os.environ['HF_TOKEN'] = hf_token
        print('  HF_TOKEN loaded from Kaggle secrets.')
    except Exception:
        print('  WARNING: No HF_TOKEN found. Downloads may be rate-limited.')
else:
    print('  HF_TOKEN already set.')

# ── 2. Download + prepare SFT data ────────────────────────────────────────────
print('\n[2/4] Preparing SFT data (100K OpenHermes + Yaya-specific)...')
os.makedirs('data/sft', exist_ok=True)

if os.path.exists(SFT_DATA_PATH):
    with open(SFT_DATA_PATH) as f:
        n = sum(1 for l in f if l.strip())
    print(f'  Existing yaya_instruct.jsonl: {n:,} examples')
    if n >= 100_000:
        print('  Dataset already large — skipping download.')
    else:
        print(f'  Only {n:,} examples — downloading fresh from OpenHermes...')
        subprocess.run(
            [sys.executable, 'scripts/download_openhermes.py', '--max_examples', '100000'],
            check=True
        )
else:
    print('  No existing data — downloading from OpenHermes...')
    subprocess.run(
        [sys.executable, 'scripts/download_openhermes.py', '--max_examples', '100000'],
        check=True
    )

# Verify eval file exists (small held-out set)
if not os.path.exists(EVAL_DATA_PATH):
    print(f'  WARNING: eval file not found at {EVAL_DATA_PATH}')

# ── 3. Find pretrain checkpoint ───────────────────────────────────────────────
print('\n[3/4] Locating pretrain checkpoint...')

def find_best_checkpoint(directory: str) -> str | None:
    ckpts = sorted(glob.glob(f'{directory}/checkpoint-*'))
    return ckpts[-1] if ckpts else None

pretrain_ckpt = (
    find_best_checkpoint(PRETRAIN_CKPT_DATASET) or
    find_best_checkpoint(PRETRAIN_CKPT_WORKING)
)

if pretrain_ckpt:
    print(f'  Pretrain checkpoint: {pretrain_ckpt}')
else:
    print('  No pretrain checkpoint found — training from random init.')
    print('  (For best results, upload a pretrain checkpoint as a Kaggle dataset)')

# Check for existing SFT checkpoint (auto-resume)
sft_ckpt = find_best_checkpoint(SFT_CKPT_DIR)
if sft_ckpt:
    print(f'  Resuming SFT from: {sft_ckpt}')

os.makedirs(SFT_CKPT_DIR, exist_ok=True)

# ── 4. Patch config for Kaggle paths + dtype ──────────────────────────────────
print('\n[4/4] Starting SFT training...')

import yaml

with open('configs/training/sft_125m.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['checkpointing']['save_dir'] = SFT_CKPT_DIR
cfg['training']['dtype']         = DTYPE
cfg['data']['train_data']        = SFT_DATA_PATH
cfg['data']['eval_data']         = EVAL_DATA_PATH
cfg['data']['tokenizer_path']    = TOKENIZER_PATH

# Enable gradient checkpointing on low-VRAM GPUs (saves ~30% VRAM)
if torch.cuda.is_available() and vram_gb < 14:
    cfg['distributed']['gradient_checkpointing'] = True
    print(f'  Gradient checkpointing enabled (<14GB VRAM).')

with open('configs/training/sft_125m.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# Build training command
cmd = [
    sys.executable, 'scripts/train_sft.py',
    '--model_config', 'configs/model/yaya_125m.yaml',
    '--train_config', 'configs/training/sft_125m.yaml',
]

if sft_ckpt:
    cmd += ['--resume', sft_ckpt]
elif pretrain_ckpt:
    cmd += ['--pretrain_checkpoint', pretrain_ckpt]

print(f'  Command: {" ".join(cmd)}')
print()

result = subprocess.run(cmd)
sys.exit(result.returncode)
