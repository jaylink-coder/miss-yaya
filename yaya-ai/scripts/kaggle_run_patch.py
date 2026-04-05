"""Kaggle Patch SFT for Yaya-125M.

Targeted fix for remaining failures after Recovery+DPO2.
Always pulls the best checkpoint from Hub (patch > dpo2 > recovery > dpo),
trains 300 steps at lr=5e-6, then benchmarks.

v1: 500 steps from dpo2 → 91% (32/35)
v2: 300 steps from patch-checkpoint-00000500 → target 94%+ (33/35)

Usage (Kaggle cells):
    Cell 1:
        !git clone https://github.com/jaylink-coder/miss-yaya.git /kaggle/working/miss-yaya 2>/dev/null || \
            (cd /kaggle/working/miss-yaya && git fetch origin && git reset --hard origin/main)
        !pip install -q sentencepiece pyyaml huggingface_hub
        import os; os.chdir('/kaggle/working/miss-yaya/yaya-ai')

    Cell 2:
        !python scripts/kaggle_run_patch.py
"""

import json
import os
import sys
import glob
import subprocess

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_CKPT = '/kaggle/working/yaya-patch-checkpoints'
DATA_DIR   = os.path.join(REPO_ROOT, 'data/sft')
HUB_REPO   = 'Jaylink-coder/yaya-125m'

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
print(' YAYA-125M PATCH SFT v2')
print('=' * 60)


# ── Step 1: Pull best checkpoint from Hub ─────────────────────────────────────
print('\n[1/4] Finding best starting checkpoint...')

if not hf_token:
    print('ERROR: No HF_TOKEN.')
    sys.exit(1)

try:
    from huggingface_hub import list_repo_files, snapshot_download
    hub_files = list(list_repo_files(repo_id=HUB_REPO, repo_type='model', token=hf_token))
    all_names = sorted({f.split('/')[0] for f in hub_files if '/' in f and '_temp' not in f})

    # Priority: patch > dpo2 > recovery > dpo
    hub_ckpt = None
    for prefix in ('patch-checkpoint-', 'dpo2-checkpoint-', 'recovery-checkpoint-', 'dpo-checkpoint-'):
        matches = [c for c in all_names if c.startswith(prefix)]
        if matches:
            hub_ckpt = matches[-1]
            break

    if not hub_ckpt:
        print('ERROR: No suitable checkpoint on Hub.')
        sys.exit(1)

    # Check if it's already local
    dl_dir = PATCH_CKPT if hub_ckpt.startswith('patch-') else '/kaggle/working/yaya-dpo2-checkpoints'
    local_path = os.path.join(dl_dir, hub_ckpt)

    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, 'model.pt')):
        print(f'  Already local: {local_path}')
    else:
        print(f'  Downloading {hub_ckpt} from Hub...')
        os.makedirs(dl_dir, exist_ok=True)
        snapshot_download(
            repo_id=HUB_REPO,
            allow_patterns=f'{hub_ckpt}/*',
            local_dir=dl_dir,
            repo_type='model',
            token=hf_token,
        )
        print(f'  Downloaded: {local_path}')

    start_ckpt = local_path

    # Read its metadata
    meta_path = os.path.join(start_ckpt, 'metadata.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    start_step = meta.get('step', 0)
    start_loss = meta.get('loss', 99)
    print(f'  Starting from: {hub_ckpt}  step={start_step}  loss={start_loss:.4f}')

except Exception as e:
    print(f'ERROR: Hub pull failed: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)

# Decide if we should train or just benchmark
# patch-checkpoint-00000500 = v1 (run v2 from it)
# patch-checkpoint-00000300 = v2 done (benchmark only)
# anything else = run patch from scratch
_is_patch = hub_ckpt.startswith('patch-')
_ckpt_num = int(hub_ckpt.split('-')[-1]) if _is_patch else 0
_v2_done = _is_patch and _ckpt_num == 300   # v2 produces step=300 (old — no Kenya data)
_v3_done = _is_patch and _ckpt_num == 800   # v3 = 500+300 — done if we also did v3
_v1_done = _is_patch and _ckpt_num == 500   # v1 produces step=500 — run v2 from here
# v3: run from v2 (step=300) adding Kenya/Swahili, 500 more steps, lr=3e-6
_skip_training = _v3_done

if _skip_training:
    print(f'  Patch v3 already complete ({hub_ckpt}) — skipping to benchmark.')
    patch_ckpt = start_ckpt
elif _v2_done:
    print(f'  Patch v2 found ({hub_ckpt}) — running v3 (500 steps, lr=3e-6, Kenya/Swahili data).')
elif _v1_done:
    print(f'  Patch v1 found ({hub_ckpt}) — running v2 (300 steps, lr=5e-6).')
else:
    print(f'  No patch checkpoint — running v1 (500 steps, lr=1e-5) from {hub_ckpt}.')


# ── Step 2: Build patch dataset ───────────────────────────────────────────────
if not _skip_training:
    print('\n[2/4] Building patch dataset...')

    PATCH_DATA      = '/kaggle/working/yaya_patch_data.jsonl'
    PATCH_SFT       = os.path.join(DATA_DIR, 'yaya_patch_sft.jsonl')
    SHORT_QA        = os.path.join(DATA_DIR, 'yaya_short_qa.jsonl')
    QUICK_FACTS     = os.path.join(DATA_DIR, 'teach/quick_facts.jsonl')
    FACTUAL_QA      = os.path.join(DATA_DIR, 'yaya_factual_qa.jsonl')
    KENYA_SWAHILI   = os.path.join(DATA_DIR, 'yaya_kenya_swahili.jsonl')

    all_lines = []

    if not os.path.exists(PATCH_SFT):
        print(f'  ERROR: {PATCH_SFT} not found. Regenerating...')
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'scripts/generate_patch_data.py')],
                       cwd=REPO_ROOT)

    # Auto-generate Kenya/Swahili data if missing
    if not os.path.exists(KENYA_SWAHILI):
        print(f'  Generating Kenya/Swahili data...')
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'scripts/generate_kenya_swahili_data.py')],
                       cwd=REPO_ROOT)

    with open(PATCH_SFT, encoding='utf-8', errors='replace') as f:
        patch_lines = [l.strip() for l in f if l.strip()]
    # 5x oversample for v2 (targeting 6 specific failures)
    all_lines.extend(patch_lines * 5)
    print(f'  + yaya_patch_sft.jsonl: {len(patch_lines)} x5 = {len(patch_lines)*5}')

    for path, label, mult in [(SHORT_QA,      'yaya_short_qa.jsonl',      2),
                               (QUICK_FACTS,   'quick_facts.jsonl',        2),
                               (FACTUAL_QA,    'yaya_factual_qa.jsonl',    1),
                               (KENYA_SWAHILI, 'yaya_kenya_swahili.jsonl', 4)]:
        if os.path.exists(path):
            with open(path, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f if l.strip()]
            all_lines.extend(lines * mult)
            print(f'  + {label}: {len(lines)} x{mult} = {len(lines)*mult}')

    import random; random.seed(42)
    random.shuffle(all_lines)

    with open(PATCH_DATA, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    print(f'  Total: {len(all_lines)} examples → {PATCH_DATA}')


# ── Step 3: Train ─────────────────────────────────────────────────────────────
if not _skip_training:
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
    with open(os.path.join(REPO_ROOT, 'configs/training/sft_125m.yaml')) as f:
        patch_cfg = yaml.safe_load(f)

    # Conservative settings — minimize forgetting
    if _v2_done:
        n_steps, lr = 500, 3e-6   # v3: Kenya/Swahili injection
    elif _v1_done:
        n_steps, lr = 300, 5e-6   # v2: targeted failure fix
    else:
        n_steps, lr = 500, 1e-5   # v1: from dpo2
    patch_cfg['training']['max_steps']      = n_steps
    patch_cfg['training']['learning_rate']  = lr
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

    print(f'  Steps: {n_steps}  LR: {lr}  Starting from: {os.path.basename(start_ckpt)}')
    result = subprocess.run([
        sys.executable, '-u', os.path.join(REPO_ROOT, 'scripts/train_sft.py'),
        '--model_config', os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml'),
        '--train_config', tmp_cfg,
        '--pretrain_checkpoint', start_ckpt,
    ], cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f'  ERROR: Training exited {result.returncode}')
        sys.exit(result.returncode)

    # Rename to patch-checkpoint-NNNNN
    final_ckpts = sorted(glob.glob(os.path.join(PATCH_CKPT, 'checkpoint-*')))
    if not final_ckpts:
        print('  ERROR: No checkpoint after training.')
        sys.exit(1)
    final = final_ckpts[-1]
    step_num = os.path.basename(final).split('-')[-1]
    patch_name = f'patch-checkpoint-{step_num}'
    patch_final = os.path.join(PATCH_CKPT, patch_name)
    os.rename(final, patch_final)
    print(f'  Renamed {os.path.basename(final)} -> {patch_name}')
    patch_ckpt = patch_final

    if HF_TOKEN:
        try:
            from scripts.hub_utils import push_checkpoint, ensure_repo
            ensure_repo(HUB_REPO, hf_token)
            push_checkpoint(patch_ckpt, HUB_REPO, hf_token)
        except Exception as _e:
            print(f'  WARNING: Hub push failed: {_e}')


# ── Step 4: Benchmark ─────────────────────────────────────────────────────────
print('\n[4/4] Running benchmark...')
bench_result = subprocess.run([
    sys.executable, '-u', os.path.join(REPO_ROOT, 'scripts/benchmark.py'),
    '--checkpoint', patch_ckpt,
], cwd=REPO_ROOT)
if bench_result.returncode != 0:
    print(f'  WARNING: Benchmark exited {bench_result.returncode}')

print()
print('=' * 60)
print(' PATCH SFT COMPLETE')
print(f'  Checkpoint: {patch_ckpt}')
print('=' * 60)
