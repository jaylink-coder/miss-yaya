"""Kaggle Recovery SFT for Yaya-125M.

Targeted 3,000-step re-training on short Q&A + quick facts ONLY.
Loads the DPO checkpoint and overwrites the numbered-list habit with
direct, concise answers.

Usage (Kaggle notebook cell):
    !git clone https://github.com/jaylink-coder/miss-yaya.git /kaggle/working/miss-yaya 2>/dev/null || \
        (cd /kaggle/working/miss-yaya && git pull origin main)
    !pip install -q sentencepiece pyyaml huggingface_hub
    import os; os.chdir('/kaggle/working/miss-yaya/yaya-ai')
    !python scripts/kaggle_run_recovery.py
"""

import json
import os
import sys
import glob
import subprocess
from pathlib import Path

REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOVERY_CKPT   = '/kaggle/working/yaya-recovery-checkpoints'
DPO_CKPT_DIR    = '/kaggle/working/yaya-dpo-checkpoints'
SFT_CKPT_DIR    = '/kaggle/working/yaya-sft-checkpoints'
RECOVERY_HUB_PREFIX = 'recovery-'  # prefix Hub uploads to avoid collision with SFT/DPO
DATA_DIR        = os.path.join(REPO_ROOT, 'data/sft')
TOKENIZER_PATH  = os.path.join(REPO_ROOT, 'data/tokenizer/yaya_tokenizer.model')
HUB_REPO        = 'Jaylink-coder/yaya-125m'

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONIOENCODING'] = 'utf-8'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Secrets ───────────────────────────────────────────────────────────────────
def load_secret(name):
    val = os.environ.get(name)
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


hf_token = load_secret('HF_TOKEN')
HF_TOKEN = hf_token

if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print('HF_TOKEN loaded.')
else:
    print('WARNING: No HF_TOKEN — checkpoints will NOT persist!')

os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f}GB')
    DTYPE = 'float16'
else:
    print('WARNING: No GPU.')
    DTYPE = 'float32'

print()
print('=' * 60)
print(' YAYA-125M RECOVERY SFT')
print('=' * 60)


# ── Step 1: Find best starting checkpoint ─────────────────────────────────────
print('\n[1/4] Finding starting checkpoint...')

def find_local_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint-*')))
    return ckpts[-1] if ckpts else None

# Priority: recovery > DPO > SFT > Hub
start_ckpt = (
    find_local_checkpoint(RECOVERY_CKPT) or
    find_local_checkpoint(DPO_CKPT_DIR) or
    find_local_checkpoint(SFT_CKPT_DIR)
)

if not start_ckpt and HF_TOKEN:
    print('  No local checkpoint — pulling from HF Hub...')
    try:
        from scripts.hub_utils import pull_latest_checkpoint
        start_ckpt = pull_latest_checkpoint(HUB_REPO, DPO_CKPT_DIR, hf_token)
    except Exception as e:
        print(f'  Hub pull failed: {e}')

if not start_ckpt:
    print('ERROR: No checkpoint found. Cannot run recovery.')
    sys.exit(1)

print(f'  Starting from: {start_ckpt}')


# ── Step 2: Build recovery dataset ────────────────────────────────────────────
print('\n[2/4] Building recovery dataset...')

RECOVERY_DATA = os.path.join(DATA_DIR, 'yaya_recovery.jsonl')

SHORT_QA  = os.path.join(DATA_DIR, 'yaya_short_qa.jsonl')
QUICK_FACTS = os.path.join(DATA_DIR, 'teach/quick_facts.jsonl')
SHORT_QA_COMBINED = os.path.join(DATA_DIR, 'yaya_reasoning_combined.jsonl')

sources = []
for src in [SHORT_QA, QUICK_FACTS, SHORT_QA_COMBINED]:
    if os.path.exists(src):
        with open(src, encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
        sources.extend(lines)
        print(f'  + {os.path.basename(src)}: {len(lines)} examples')

# Oversample short_qa and quick_facts 5x to overpower the math habit
boosted = []
for src in [SHORT_QA, QUICK_FACTS]:
    if os.path.exists(src):
        with open(src, encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
        boosted.extend(lines * 5)

all_examples = sources + boosted
import random; random.seed(42); random.shuffle(all_examples)

with open(RECOVERY_DATA, 'w', encoding='utf-8') as f:
    for line in all_examples:
        f.write(line + '\n')

print(f'  Recovery dataset: {len(all_examples)} examples (with 5x oversampling)')
print(f'  Saved → {RECOVERY_DATA}')


# ── Step 3: Train ─────────────────────────────────────────────────────────────
print('\n[3/4] Launching recovery training...')
print('  Strategy: high LR (5e-5), 3,000 steps, short sequences only')
print('  Goal: overwrite numbered-list habit with direct answers')

os.makedirs(RECOVERY_CKPT, exist_ok=True)

# Write a temporary train config with recovery settings
import yaml
base_cfg_path = os.path.join(REPO_ROOT, 'configs/training/sft_125m.yaml')
with open(base_cfg_path) as f:
    recovery_cfg = yaml.safe_load(f)

# Patch recovery-specific values
recovery_cfg['training']['train_data']                = RECOVERY_DATA
recovery_cfg['training']['max_steps']                 = 3000
recovery_cfg['training']['learning_rate']             = 5e-5
recovery_cfg['training']['max_seq_length']            = 128
recovery_cfg['training']['save_steps']                = 500
recovery_cfg['training']['checkpoint_dir']            = RECOVERY_CKPT
recovery_cfg['training']['dtype']                     = DTYPE
recovery_cfg['training']['pretrain_checkpoint']       = start_ckpt
recovery_cfg['data']   = recovery_cfg.get('data', {})
recovery_cfg['data']['train_data'] = RECOVERY_DATA

tmp_cfg_path = '/kaggle/working/recovery_train_config.yaml'
with open(tmp_cfg_path, 'w') as f:
    yaml.dump(recovery_cfg, f)

train_script = os.path.join(REPO_ROOT, 'scripts/train_sft.py')
train_cmd = [
    sys.executable, train_script,
    '--model_config', os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml'),
    '--train_config', tmp_cfg_path,
    '--pretrain_checkpoint', start_ckpt,
]

print(f'  Command: {" ".join(train_cmd[:4])} ...')
result = subprocess.run(train_cmd, cwd=REPO_ROOT)
training_ok = result.returncode == 0

if not training_ok:
    # Fallback: drive the trainer directly
    print('  train_sft.py failed — driving trainer directly...')
    try:
        from src.training.trainer import Trainer, TrainingConfig
        from src.model.yaya_model import YayaForCausalLM
        from src.model.config import ModelConfig
        from src.tokenizer.tokenizer import YayaTokenizer
        from src.data.dataset import InstructionDataset

        with open(os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml')) as f:
            model_cfg_dict = yaml.safe_load(f)

        model_cfg = ModelConfig(**model_cfg_dict)
        tokenizer = YayaTokenizer(TOKENIZER_PATH)
        model = YayaForCausalLM(model_cfg)

        ckpt_file = os.path.join(start_ckpt, 'model.pt')
        state = torch.load(ckpt_file, map_location='cpu')
        weights = state.get('model', state)
        model.load_state_dict(weights, strict=False)
        print('  Model weights loaded.')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        dataset = InstructionDataset(RECOVERY_DATA, tokenizer, max_seq_len=128)

        train_cfg = TrainingConfig(
            max_steps=3000,
            learning_rate=5e-5,
            batch_size=4,
            gradient_accumulation_steps=4,
            save_steps=500,
            checkpoint_dir=RECOVERY_CKPT,
            dtype=DTYPE,
        )

        trainer = Trainer(model, tokenizer, dataset, train_cfg)
        trainer.train()
        training_ok = True
    except Exception as e:
        print(f'  Fallback trainer also failed: {e}')
        training_ok = False


# ── Step 4: Benchmark ─────────────────────────────────────────────────────────
print('\n[4/4] Running benchmark on recovery checkpoint...')

recovery_ckpts = sorted(glob.glob(os.path.join(RECOVERY_CKPT, 'checkpoint-*')))
if recovery_ckpts:
    best_ckpt = recovery_ckpts[-1]

    # Push to Hub with recovery- prefix to avoid collision with SFT/DPO checkpoints
    if HF_TOKEN:
        try:
            from scripts.hub_utils import _get_api
            from huggingface_hub import upload_folder, upload_file
            import io
            ckpt_name = RECOVERY_HUB_PREFIX + os.path.basename(best_ckpt)
            print(f'[Hub] Pushing {ckpt_name} → {HUB_REPO}...')
            upload_folder(
                folder_path=best_ckpt,
                repo_id=HUB_REPO,
                path_in_repo=ckpt_name,
                repo_type='model',
                token=hf_token,
                ignore_patterns=['optimizer.pt'],
                commit_message=f'Recovery checkpoint: {ckpt_name}',
            )
            upload_file(
                path_or_fileobj=io.BytesIO(f'{{"latest": "{ckpt_name}"}}'.encode()),
                path_in_repo='latest.json',
                repo_id=HUB_REPO,
                repo_type='model',
                token=hf_token,
                commit_message=f'Update latest → {ckpt_name}',
            )
            print(f'[Hub] Recovery checkpoint pushed → {HUB_REPO}/{ckpt_name}')
        except Exception as e:
            print(f'[Hub] Push failed (non-fatal): {e}')

    # Benchmark
    bench_cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, 'scripts/benchmark.py'),
        '--checkpoint', best_ckpt,
    ]
    subprocess.run(bench_cmd, cwd=REPO_ROOT)
else:
    print('  No recovery checkpoint found — benchmark skipped.')

print()
print('=' * 60)
print(' RECOVERY COMPLETE' if training_ok else ' RECOVERY FAILED')
print('=' * 60)
sys.exit(0 if training_ok else 1)
