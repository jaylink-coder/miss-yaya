"""Kaggle Patch SFT for Yaya-125M.

Targeted 500-step fix for specific DPO2 failures:
- Factual confusion (planet=Earth, months=12, H2O, Kenya=Nairobi, boiling=100C)
- Language questions (opposites, language identification)
- Word problem arithmetic (10-3=7, 60km/h*2h=120km)
- Reasoning with direct final answer

Starts from dpo2-checkpoint-00001500 on Hub.

Usage (Kaggle cell):
    !git clone https://github.com/jaylink-coder/miss-yaya.git /kaggle/working/miss-yaya 2>/dev/null || \
        (cd /kaggle/working/miss-yaya && git pull origin main)
    !pip install -q sentencepiece pyyaml huggingface_hub
    import os; os.chdir('/kaggle/working/miss-yaya/yaya-ai')
    !python scripts/kaggle_run_patch.py
"""

import json
import os
import sys
import glob
import subprocess
from pathlib import Path

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_CKPT   = '/kaggle/working/yaya-patch-checkpoints'
DATA_DIR     = os.path.join(REPO_ROOT, 'data/sft')
TOKENIZER    = os.path.join(REPO_ROOT, 'data/tokenizer/yaya_tokenizer.model')
HUB_REPO     = 'Jaylink-coder/yaya-125m'

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

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
    print('WARNING: No HF_TOKEN — checkpoints will NOT persist to Hub!')

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
print(' YAYA-125M PATCH SFT (fixing DPO2 failures)')
print('=' * 60)


# ── Step 1: Find starting checkpoint (dpo2 > recovery > dpo > sft) ────────────
print('\n[1/4] Finding starting checkpoint...')

def find_local_checkpoint(ckpt_dir, prefix='checkpoint-'):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, f'{prefix}*')))
    return ckpts[-1] if ckpts else None


# Check if patch already completed
patch_ckpt = find_local_checkpoint(PATCH_CKPT, 'patch-checkpoint-')
if patch_ckpt:
    import json as _j
    meta = _j.load(open(os.path.join(patch_ckpt, 'metadata.json'))) if os.path.exists(os.path.join(patch_ckpt, 'metadata.json')) else {}
    _step = meta.get('step', 0)
    _loss = meta.get('loss', 99)
    if _step >= 600 and _loss > 0.05:
        print(f'  Patch v2 already complete at step {_step} (loss={_loss:.4f}) — skipping to benchmark.')
        start_ckpt = patch_ckpt
        _already_done = True
    else:
        # v1 done (500 steps) — run v2 micro-patch (300 more steps) from patch ckpt
        print(f'  Patch v1 at step {_step} — running v2 micro-patch (300 more steps).')
        start_ckpt = patch_ckpt
        _already_done = False
else:
    start_ckpt = None
    _already_done = False

if not _already_done:
    # Priority: local dpo2 > local recovery > Hub
    DPO2_LOCAL = 'checkpoints/yaya-125m-dpo2'
    RECOVERY_LOCAL = '/kaggle/working/yaya-recovery-checkpoints'
    DPO_LOCAL = '/kaggle/working/yaya-dpo-checkpoints'

    local_ckpt = (
        find_local_checkpoint(DPO2_LOCAL, 'dpo2-checkpoint-') or
        find_local_checkpoint(RECOVERY_LOCAL, 'checkpoint-') or
        find_local_checkpoint(DPO_LOCAL, 'checkpoint-')
    )

    if local_ckpt:
        print(f'  Found local: {local_ckpt}')
        start_ckpt = local_ckpt
    elif hf_token:
        print('  No local checkpoint — pulling dpo2-checkpoint-00001500 from Hub...')
        try:
            from huggingface_hub import list_repo_files, snapshot_download
            hub_files = list(list_repo_files(repo_id=HUB_REPO, repo_type='model', token=hf_token))

            # Priority: dpo2 > recovery > dpo
            hub_ckpt = None
            all_names = sorted({f.split('/')[0] for f in hub_files if '/' in f and '_temp' not in f})
            for prefix in ('dpo2-checkpoint-', 'recovery-checkpoint-', 'dpo-checkpoint-'):
                matches = [c for c in all_names if c.startswith(prefix)]
                if matches:
                    hub_ckpt = matches[-1]
                    break

            if hub_ckpt:
                print(f'  Downloading {hub_ckpt}...')
                dl_dir = '/kaggle/working/yaya-dpo2-checkpoints'
                os.makedirs(dl_dir, exist_ok=True)
                snapshot_download(
                    repo_id=HUB_REPO,
                    allow_patterns=f'{hub_ckpt}/*',
                    local_dir=dl_dir,
                    repo_type='model',
                    token=hf_token,
                )
                local_path = os.path.join(dl_dir, hub_ckpt)
                if os.path.isdir(local_path):
                    start_ckpt = local_path
                    print(f'  Restored: {start_ckpt}')
            else:
                print('  ERROR: No dpo2/recovery/dpo checkpoint on Hub.')
                sys.exit(1)
        except Exception as e:
            print(f'  Hub pull failed: {e}')
            sys.exit(1)
    else:
        print('ERROR: No checkpoint found and no HF_TOKEN.')
        sys.exit(1)

    print(f'  Starting from: {start_ckpt}')


# ── Step 2: Build patch dataset ────────────────────────���───────────────────────
if not _already_done:
    print('\n[2/4] Building patch dataset...')

    PATCH_DATA   = '/kaggle/working/yaya_patch_data.jsonl'
    PATCH_SFT    = os.path.join(DATA_DIR, 'yaya_patch_sft.jsonl')
    SHORT_QA     = os.path.join(DATA_DIR, 'yaya_short_qa.jsonl')
    QUICK_FACTS  = os.path.join(DATA_DIR, 'teach/quick_facts.jsonl')
    FACTUAL_QA   = os.path.join(DATA_DIR, 'yaya_factual_qa.jsonl')

    all_lines = []

    # Patch examples — 5x oversample (150 examples → 750) — these are the specific fixes
    if os.path.exists(PATCH_SFT):
        with open(PATCH_SFT, encoding='utf-8', errors='replace') as f:
            patch_lines = [l.strip() for l in f if l.strip()]
        all_lines.extend(patch_lines * 5)
        print(f'  + yaya_patch_sft.jsonl: {len(patch_lines)} examples x5 = {len(patch_lines)*5}')
    else:
        print(f'  ERROR: {PATCH_SFT} not found. Run scripts/generate_patch_data.py first.')
        sys.exit(1)

    # Short QA — 2x (broad factual/arithmetic grounding)
    if os.path.exists(SHORT_QA):
        with open(SHORT_QA, encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
        all_lines.extend(lines * 2)
        print(f'  + yaya_short_qa.jsonl: {len(lines)} x2 = {len(lines)*2}')

    # Quick facts — 2x
    if os.path.exists(QUICK_FACTS):
        with open(QUICK_FACTS, encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
        all_lines.extend(lines * 2)
        print(f'  + quick_facts.jsonl: {len(lines)} x2 = {len(lines)*2}')

    # Factual QA — 1x (prevents forgetting learned facts)
    if os.path.exists(FACTUAL_QA):
        with open(FACTUAL_QA, encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if l.strip()]
        all_lines.extend(lines)
        print(f'  + yaya_factual_qa.jsonl: {len(lines)} x1')

    import random; random.seed(42)
    random.shuffle(all_lines)

    with open(PATCH_DATA, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    print(f'  Patch dataset: {len(all_lines)} total examples')
    print(f'  Saved → {PATCH_DATA}')


# ── Step 3: Train ───────────────────────────���────────────────────���─────────────
if not _already_done:
    print('\n[3/4] Launching patch training...')
    os.makedirs(PATCH_CKPT, exist_ok=True)

    if HF_TOKEN:
        try:
            from scripts.hub_utils import start_watcher, ensure_repo
            ensure_repo(HUB_REPO, hf_token)
            _watcher = start_watcher(PATCH_CKPT, HUB_REPO, hf_token, interval_sec=120)
        except Exception as _e:
            print(f'  WARNING: Hub watcher failed to start: {_e}')

    import yaml
    base_cfg_path = os.path.join(REPO_ROOT, 'configs/training/sft_125m.yaml')
    with open(base_cfg_path) as f:
        patch_cfg = yaml.safe_load(f)

    patch_cfg['training']['max_steps']      = 500    # short — just patch specific failures
    patch_cfg['training']['learning_rate']  = 1e-5   # conservative — avoid forgetting
    patch_cfg['training']['max_seq_length'] = 256
    patch_cfg['training']['dtype']          = DTYPE
    patch_cfg['checkpointing'] = patch_cfg.get('checkpointing', {})
    patch_cfg['checkpointing']['save_steps'] = 100
    patch_cfg['checkpointing']['save_dir']   = PATCH_CKPT
    patch_cfg['data'] = patch_cfg.get('data', {})
    patch_cfg['data']['train_data'] = PATCH_DATA

    tmp_cfg = '/kaggle/working/patch_train_config.yaml'
    with open(tmp_cfg, 'w') as f:
        yaml.dump(patch_cfg, f)

    train_cmd = [
        sys.executable, '-u', os.path.join(REPO_ROOT, 'scripts/train_sft.py'),
        '--model_config', os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml'),
        '--train_config', tmp_cfg,
        '--pretrain_checkpoint', start_ckpt,
    ]

    print(f'  CMD: {" ".join(train_cmd[2:])}')
    result = subprocess.run(train_cmd, cwd=REPO_ROOT)
    training_ok = result.returncode == 0

    if not training_ok:
        print(f'  ERROR: Training exited with code {result.returncode}')
        sys.exit(result.returncode)

    # Rename final checkpoint to patch-checkpoint-NNNNN
    final_ckpts = sorted(glob.glob(os.path.join(PATCH_CKPT, 'checkpoint-*')))
    if final_ckpts:
        final = final_ckpts[-1]
        step_num = os.path.basename(final).split('-')[-1]
        patch_name = f'patch-checkpoint-{step_num}'
        patch_final = os.path.join(PATCH_CKPT, patch_name)
        os.rename(final, patch_final)
        print(f'  Renamed {os.path.basename(final)} -> {patch_name}')
        patch_ckpt = patch_final
    else:
        print('  WARNING: No checkpoint found after training.')
        sys.exit(1)

    # Push final checkpoint to Hub
    if HF_TOKEN:
        try:
            from scripts.hub_utils import push_checkpoint, ensure_repo
            ensure_repo(HUB_REPO, hf_token)
            push_checkpoint(patch_ckpt, HUB_REPO, hf_token)
        except Exception as _e:
            print(f'  WARNING: Hub push failed: {_e}')


# ── Step 4: Benchmark ──────────────────────────────────────────────────────────
print('\n[4/4] Running benchmark on patch checkpoint...')
ckpt_to_bench = patch_ckpt

bench_cmd = [
    sys.executable, '-u', os.path.join(REPO_ROOT, 'scripts/benchmark.py'),
    '--checkpoint', ckpt_to_bench,
]
bench_result = subprocess.run(bench_cmd, cwd=REPO_ROOT)
if bench_result.returncode != 0:
    print(f'  WARNING: Benchmark exited {bench_result.returncode}')

print()
print('=' * 60)
print(' PATCH SFT COMPLETE')
print(f'  Checkpoint: {ckpt_to_bench}')
print('=' * 60)
