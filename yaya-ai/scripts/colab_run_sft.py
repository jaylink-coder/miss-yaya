"""
Colab SFT runner — downloads 100K OpenHermes, then fine-tunes Yaya-125M.

Usage (Colab cell):
    from google.colab import drive
    drive.mount('/content/drive')
    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO /content/miss-yaya
    %cd /content/miss-yaya/yaya-ai
    !pip install -q sentencepiece pyyaml datasets
    !python scripts/colab_run_sft.py

Checkpoints are saved to Google Drive so they survive session resets.
Re-run safe: auto-resumes from latest SFT checkpoint.

Secrets: add HF_TOKEN and (optionally) WANDB_API_KEY to Colab Secrets
(left sidebar → key icon) so they are injected as env vars.
"""

import os
import sys
import glob
import subprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_CKPT_DIR   = '/content/drive/MyDrive/yaya-sft-checkpoints'
SFT_DATA_PATH  = 'data/sft/yaya_instruct.jsonl'
EVAL_DATA_PATH = 'data/sft/yaya_instruct_eval.jsonl'
TOKENIZER_PATH = 'data/tokenizer/yaya_tokenizer.model'

# Pretrain checkpoint lives in Drive (from colab_run.py pretrain session)
PRETRAIN_CKPT_DIR = '/content/drive/MyDrive/yaya-checkpoints'

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

if not os.environ.get('WANDB_API_KEY'):
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE']     = 'disabled'

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── 0. GPU check ──────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    cap     = props.major * 10 + props.minor
    DTYPE   = 'bfloat16' if cap >= 80 else 'float16'
    print(f'GPU:   {props.name}  VRAM: {vram_gb:.1f}GB')
    print(f'Compute capability: {props.major}.{props.minor}  → {DTYPE}')
else:
    print('WARNING: No GPU. Go to Runtime → Change runtime type → GPU.')
    DTYPE = 'float32'

# ── 1. HF Token ───────────────────────────────────────────────────────────────
print('\n[1/4] Setting up HuggingFace token...')
hf_token = os.environ.get('HF_TOKEN', '')
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        os.environ['HF_TOKEN'] = hf_token
        print('  HF_TOKEN loaded from Colab Secrets.')
    except Exception:
        print('  WARNING: No HF_TOKEN. Add it via the key icon in the left sidebar.')
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
    print('  No data found — downloading from OpenHermes...')
    subprocess.run(
        [sys.executable, 'scripts/download_openhermes.py', '--max_examples', '100000'],
        check=True
    )

if not os.path.exists(EVAL_DATA_PATH):
    print(f'  WARNING: eval file not found at {EVAL_DATA_PATH}')

# ── 3. Find pretrain checkpoint ───────────────────────────────────────────────
print('\n[3/4] Locating pretrain checkpoint...')

def find_best_checkpoint(directory: str) -> str | None:
    ckpts = sorted(glob.glob(f'{directory}/checkpoint-*'))
    return ckpts[-1] if ckpts else None

pretrain_ckpt = find_best_checkpoint(PRETRAIN_CKPT_DIR)
sft_ckpt      = find_best_checkpoint(SFT_CKPT_DIR)

if pretrain_ckpt:
    print(f'  Pretrain checkpoint: {pretrain_ckpt}')
else:
    print('  No pretrain checkpoint in Drive — training from random init.')
    print(f'  (Run colab_run.py first, or copy a checkpoint to {PRETRAIN_CKPT_DIR})')

if sft_ckpt:
    print(f'  Resuming SFT from: {sft_ckpt}')

os.makedirs(SFT_CKPT_DIR, exist_ok=True)

# ── 4. Patch config + train ───────────────────────────────────────────────────
print('\n[4/4] Starting SFT training...')

import yaml

with open('configs/training/sft_125m.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['checkpointing']['save_dir'] = SFT_CKPT_DIR
cfg['training']['dtype']         = DTYPE
cfg['data']['train_data']        = SFT_DATA_PATH
cfg['data']['eval_data']         = EVAL_DATA_PATH
cfg['data']['tokenizer_path']    = TOKENIZER_PATH

if torch.cuda.is_available() and vram_gb < 14:
    cfg['distributed']['gradient_checkpointing'] = True
    print('  Gradient checkpointing enabled (<14GB VRAM).')

with open('configs/training/sft_125m.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

cmd = [
    sys.executable, 'scripts/train_sft.py',
    '--model_config', 'configs/model/yaya_125m.yaml',
    '--train_config', 'configs/training/sft_125m.yaml',
]

if sft_ckpt:
    cmd += ['--resume', sft_ckpt]
elif pretrain_ckpt:
    cmd += ['--pretrain_checkpoint', pretrain_ckpt]

print(f'  Command: {" ".join(cmd)}\n')
result = subprocess.run(cmd)
sys.exit(result.returncode)
