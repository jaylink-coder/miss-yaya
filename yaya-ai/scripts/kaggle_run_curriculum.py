"""Kaggle runner for Yaya True AI Curriculum — 16-phase training.

Each Kaggle session (3.5 hours on T4) can train ~4-5 phases.
Checkpoints pushed to HF Hub after every save for crash resilience.

One-time Kaggle setup (Settings -> Secrets):
  HF_TOKEN       — HuggingFace token (REQUIRED)
  WANDB_API_KEY  — W&B monitoring (optional)

How it works:
  1. Pulls latest checkpoint from HF Hub (or starts from existing SFT base)
  2. Detects which curriculum phase to run next (1-16)
  3. Generates training data for that phase + replay augmentation
  4. Trains for 2000 steps with checkpoint watcher pushing to Hub
  5. Runs phase evaluation
  6. Repeats for next phase until session time runs out
  7. Next session: re-run notebook — auto-resumes from where we left off

Compute budget:
  Per phase: 2000 steps x ~1.2s/step = ~40 min
  Per session: ~4-5 phases (3.5 hour Kaggle limit)
  Total: 16 phases = ~4 sessions
"""

import json
import os
import random
import re
import sys
import glob
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR       = '/kaggle/working/yaya-curriculum-checkpoints'
DATA_DIR       = os.path.join(REPO_ROOT, 'data/sft')
CURRICULUM_DIR = os.path.join(DATA_DIR, 'curriculum')
TOKENIZER_PATH = os.path.join(REPO_ROOT, 'data/tokenizer/yaya_tokenizer.model')
MILESTONES_V2  = os.path.join(REPO_ROOT, 'configs/training/milestones_v2.yaml')
PROGRESS_PATH  = os.path.join(REPO_ROOT, 'docs/curriculum_progress.json')
MODEL_CONFIG   = os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml')

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONIOENCODING']        = 'utf-8'
random.seed(42)

# How many phases to attempt per Kaggle session
MAX_PHASES_PER_SESSION = 8
SESSION_START = time.time()
SESSION_LIMIT_SEC = 3.25 * 3600  # 3h15m — leave 15 min buffer
PHASE_TIMEOUT_SEC = 45 * 60  # 45 min max per phase — kill if stuck

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

hf_token  = load_secret('HF_TOKEN')
HF_TOKEN  = hf_token
wandb_key = load_secret('WANDB_API_KEY')

if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print('HF_TOKEN loaded.')
else:
    print('WARNING: No HF_TOKEN — checkpoints will NOT persist!')

if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key
    print('WANDB_API_KEY loaded.')
else:
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE']     = 'disabled'

# ── GPU ───────────────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    cap     = props.major * 10 + props.minor
    DTYPE   = 'bfloat16' if cap >= 80 else 'float16'
    print(f'\nGPU: {props.name}  VRAM: {vram_gb:.1f}GB  dtype: {DTYPE}')
else:
    print('WARNING: No GPU detected.')
    vram_gb = 0
    DTYPE   = 'float32'

# ── Load milestones_v2 ───────────────────────────────────────────────────────
with open(MILESTONES_V2) as f:
    ms_cfg = yaml.safe_load(f)

HUB_REPO      = ms_cfg.get('hub_repo', 'Jaylink-coder/yaya-125m')
PHASES         = ms_cfg['phases']
STEPS_PER_PHASE = ms_cfg.get('steps_per_phase', 2000)
REPLAY_RATIO   = ms_cfg.get('replay_ratio', 0.20)
BASE_LR        = ms_cfg.get('base_lr', 1.5e-5)
WARMUP_STEPS   = ms_cfg.get('warmup_steps', 100)
MAX_SEQ_LEN    = ms_cfg.get('max_seq_length', 512)
BASE_PRETRAIN_CKPT = 'patch-checkpoint-00000300'  # Original pretrained model

# ── Progress tracking ────────────────────────────────────────────────────────
def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed_phases": [], "current_phase": 1, "history": []}

def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)

def get_next_phase(progress):
    completed = set(progress.get("completed_phases", []))
    for p in PHASES:
        if p['id'] not in completed:
            return p
    return None

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def find_latest_local_checkpoint(directory):
    ckpts = sorted([c for c in glob.glob(os.path.join(directory, '*checkpoint-*'))
                    if not c.endswith('_temp') and os.path.isdir(c)])
    return ckpts[-1] if ckpts else None

def get_step_from_checkpoint(ckpt_path):
    if not ckpt_path:
        return 0
    m = re.search(r'checkpoint-0*(\d+)', os.path.basename(ckpt_path))
    return int(m.group(1)) if m else 0

# ── Data generation ──────────────────────────────────────────────────────────
def generate_phase_data(phase_id):
    """Generate curriculum data for a single phase."""
    print(f'  Generating data for phase {phase_id}...')
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'scripts/generate_curriculum_data.py'),
         '--phase', str(phase_id)],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr[:500]}')
        return False
    print(result.stdout.strip())
    return True

def get_phase_data_file(phase_id):
    """Find or merge the JSONL data for a given phase.

    Primary source: data_sources list from milestones_v2.yaml (content-matched).
    Fallback: flat files or subdirectories.
    """
    # 0. Already-augmented mixed file from a previous run
    mixed = glob.glob(os.path.join(CURRICULUM_DIR, f'phase{phase_id:02d}_*_mixed.jsonl'))
    if mixed:
        return mixed[0]

    # 1. Use data_sources from milestones_v2.yaml (authoritative mapping)
    phase_cfg = next((p for p in PHASES if p['id'] == phase_id), None)
    if phase_cfg and 'data_sources' in phase_cfg:
        sources = phase_cfg['data_sources']
        merged_path = os.path.join(CURRICULUM_DIR, f'phase{phase_id:02d}_merged.jsonl')
        lines = []
        for src in sources:
            src_path = os.path.join(REPO_ROOT, src) if not os.path.isabs(src) else src
            if os.path.isfile(src_path):
                with open(src_path, 'r', encoding='utf-8') as f:
                    lines.extend(line.strip() for line in f if line.strip())
            else:
                print(f'  WARNING: data source not found: {src}')
        if lines:
            random.shuffle(lines)
            with open(merged_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f'  Merged {len(lines)} examples from {len(sources)} sources → {os.path.basename(merged_path)}')
            return merged_path

    # 2. Fallback: flat file (old data_file format or generated)
    flat = glob.glob(os.path.join(CURRICULUM_DIR, f'phase{phase_id:02d}_*.jsonl'))
    flat = [f for f in flat if '_mixed' not in f]
    if flat:
        return flat[0]

    # 3. Fallback: subdirectory with JSONL files
    subdir = os.path.join(CURRICULUM_DIR, f'phase{phase_id:02d}')
    if os.path.isdir(subdir):
        sub_files = sorted(glob.glob(os.path.join(subdir, '*.jsonl')))
        if sub_files:
            merged_path = os.path.join(CURRICULUM_DIR, f'phase{phase_id:02d}_merged.jsonl')
            lines = []
            for sf in sub_files:
                with open(sf, 'r', encoding='utf-8') as f:
                    lines.extend(line.strip() for line in f if line.strip())
            random.shuffle(lines)
            with open(merged_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f'  Merged {len(lines)} examples from {len(sub_files)} files → {os.path.basename(merged_path)}')
            return merged_path

    return None

def augment_with_replay(data_file, phase_id):
    """Mix phase data with base SFT data + prior phase replay to prevent overfitting.

    Strategy:
      - Curriculum seed data is small (~20-65 examples per phase)
      - 2000 steps x batch 32 = 64K examples seen → would overfit on seeds alone
      - Solution: use existing SFT data as bulk (3000 samples) + oversample seeds (5x)
      - This teaches new capabilities while maintaining existing skills
    """
    current = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                current.append(line.strip())

    if not current:
        return data_file

    # Oversample curriculum seeds 5x so they're well-represented
    OVERSAMPLE = 5
    oversampled = current * OVERSAMPLE
    print(f'  Curriculum seeds: {len(current)} x {OVERSAMPLE} = {len(oversampled)}')

    # Collect base data from existing SFT sources (prevents forgetting)
    base_pool = []
    BASE_TARGET = 3000  # enough for ~2000 steps with batch 32

    # Priority sources (high-quality, diverse)
    base_sources = [
        ('yaya_short_qa.jsonl', 800),
        ('yaya_factual_qa.jsonl', 500),
        ('yaya_instruct_clean.jsonl', 800),
        ('teach/quick_facts.jsonl', 400),
        ('yaya_concise_sft.jsonl', 500),
    ]
    for fname, cap in base_sources:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            try:
                lines = []
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            lines.append(line.strip())
                random.shuffle(lines)
                base_pool.extend(lines[:cap])
            except Exception:
                pass

    # Also check for the large reasoning dataset (on Kaggle)
    large_ds = os.path.join(DATA_DIR, 'yaya_reasoning_large.jsonl')
    if os.path.isfile(large_ds) and len(base_pool) < BASE_TARGET:
        need = BASE_TARGET - len(base_pool)
        try:
            reservoir = []
            with open(large_ds, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    if i < need:
                        reservoir.append(line.strip())
                    else:
                        j = random.randint(0, i)
                        if j < need:
                            reservoir[j] = line.strip()
            base_pool.extend(reservoir)
        except Exception:
            pass

    # Replay from prior curriculum phases (use data_sources from YAML)
    replay = []
    for prev_id in range(1, phase_id):
        prev_cfg = next((p for p in PHASES if p['id'] == prev_id), None)
        prev_sources = []
        if prev_cfg and 'data_sources' in prev_cfg:
            prev_sources = [
                os.path.join(REPO_ROOT, s) if not os.path.isabs(s) else s
                for s in prev_cfg['data_sources']
            ]
        else:
            # Fallback: scan flat files and subdirs
            prev_sources = glob.glob(os.path.join(CURRICULUM_DIR, f'phase{prev_id:02d}_*.jsonl'))
            prev_subdir = os.path.join(CURRICULUM_DIR, f'phase{prev_id:02d}')
            if os.path.isdir(prev_subdir):
                prev_sources += glob.glob(os.path.join(prev_subdir, '*.jsonl'))
        for pf in prev_sources:
            if '_mixed' in pf or not os.path.isfile(pf):
                continue
            with open(pf, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        replay.append(line.strip())
    # Oversample prior phases 2x
    if replay:
        replay = replay * 2

    # Combine: oversampled seeds + base data + prior phase replay
    all_data = oversampled + base_pool + replay
    random.shuffle(all_data)

    print(f'  Base SFT data: {len(base_pool)}')
    print(f'  Prior phase replay: {len(replay)}')
    print(f'  Total mixed: {len(all_data)}')

    # Write mixed file
    mixed_path = data_file.replace('.jsonl', '_mixed.jsonl')

    with open(mixed_path, 'w', encoding='utf-8') as f:
        for line in all_data:
            f.write(line + '\n')

    print(f'  Written: {len(all_data)} examples -> {os.path.basename(mixed_path)}')
    return mixed_path

# ── Training config generation ────────────────────────────────────────────────
def create_training_config(phase, data_file):
    """Create a YAML training config for this phase."""
    config = {
        'seed': 42 + phase['id'],
        'training': {
            'per_device_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'learning_rate': BASE_LR,
            'weight_decay': 0.01,
            'adam_beta1': 0.9,
            'adam_beta2': 0.95,
            'adam_epsilon': 1.0e-8,
            'max_grad_norm': 1.0,
            'lr_scheduler': 'cosine',
            'warmup_steps': WARMUP_STEPS,
            'min_lr_ratio': 0.05,
            'max_steps': STEPS_PER_PHASE,
            'max_seq_length': MAX_SEQ_LEN,
            'dtype': DTYPE,
        },
        'checkpointing': {
            'save_steps': STEPS_PER_PHASE,
            'save_dir': os.path.join(CKPT_DIR, f'phase{phase["id"]:02d}'),
            'keep_last_n': 2,
        },
        'logging': {
            'log_steps': 50,
            'wandb_project': None,
            'wandb_run_name': None,
        },
        'eval': {
            'eval_steps': STEPS_PER_PHASE,
            'eval_samples': 50,
        },
        'data': {
            'train_data': data_file,
            'eval_data': data_file,
            'tokenizer_path': TOKENIZER_PATH,
            'num_workers': 0,
        },
        'distributed': {
            'strategy': 'none',
            'gradient_checkpointing': False,
            'cpu_offload': False,
        },
    }

    config_path = os.path.join(REPO_ROOT, 'configs', 'training', f'curriculum_phase{phase["id"]:02d}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path

# ── Run a single phase ────────────────────────────────────────────────────────
def run_phase(phase, checkpoint_path, progress):
    """Train a single curriculum phase. Returns (success, new_checkpoint_path)."""
    phase_id = phase['id']
    phase_name = phase['name']

    print(f'\n{"="*60}')
    print(f'  PHASE {phase_id}: {phase_name}')
    print(f'  {phase.get("description", "").strip()[:100]}')
    print(f'{"="*60}')

    # Step 1: Generate data
    data_file = get_phase_data_file(phase_id)
    if not data_file:
        if not generate_phase_data(phase_id):
            print(f'  FAILED: Could not generate data for phase {phase_id}')
            return False, checkpoint_path
        data_file = get_phase_data_file(phase_id)
        if not data_file:
            print(f'  FAILED: Data file not found after generation')
            return False, checkpoint_path

    n_examples = sum(1 for _ in open(data_file, encoding='utf-8'))
    print(f'  Data: {data_file} ({n_examples} examples)')

    # Step 2: Augment with base SFT data + replay from prior phases
    data_file = augment_with_replay(data_file, phase_id)

    # Step 3: Create config
    config_path = create_training_config(phase, data_file)

    # Step 4: Train
    phase_save_dir = os.path.join(CKPT_DIR, f'phase{phase_id:02d}')
    os.makedirs(phase_save_dir, exist_ok=True)

    cmd = [
        sys.executable, '-u',
        os.path.join(REPO_ROOT, 'scripts/train_sft.py'),
        '--model_config', MODEL_CONFIG,
        '--train_config', config_path,
    ]

    if checkpoint_path:
        cmd += ['--pretrain_checkpoint', checkpoint_path]

    print(f'\n  Training: {STEPS_PER_PHASE} steps, batch 32, lr {BASE_LR}')
    print(f'  Checkpoint: {os.path.basename(checkpoint_path) if checkpoint_path else "random init"}')
    print(f'  Save dir: {phase_save_dir}')
    print()

    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, timeout=PHASE_TIMEOUT_SEC)
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT: Phase {phase_id} exceeded {PHASE_TIMEOUT_SEC//60} min — killed')
        success = False

    # Clear CUDA cache between phases to prevent OOM buildup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 5: Find new checkpoint from THIS phase's save dir
    phase_save_dir = os.path.join(CKPT_DIR, f'phase{phase_id:02d}')
    new_ckpt = find_latest_local_checkpoint(phase_save_dir)
    if not new_ckpt:
        new_ckpt = find_latest_local_checkpoint(CKPT_DIR)

    # Step 6: Push final checkpoint (non-blocking — don't let network stall training)
    if new_ckpt and hf_token:
        from scripts.hub_utils import push_checkpoint_async
        push_checkpoint_async(new_ckpt, HUB_REPO, hf_token)

    # Step 7: Record progress
    if success:
        if phase_id not in progress['completed_phases']:
            progress['completed_phases'].append(phase_id)
        progress['current_phase'] = phase_id + 1
        progress['history'].append({
            'phase_id': phase_id,
            'phase_name': phase_name,
            'status': 'complete',
            'checkpoint': os.path.basename(new_ckpt) if new_ckpt else None,
            'completed_at': datetime.now(timezone.utc).isoformat(),
        })
        save_progress(progress)
        print(f'\n  Phase {phase_id} COMPLETE')
    else:
        progress['history'].append({
            'phase_id': phase_id,
            'phase_name': phase_name,
            'status': 'failed',
            'completed_at': datetime.now(timezone.utc).isoformat(),
        })
        save_progress(progress)
        print(f'\n  Phase {phase_id} FAILED')

    return success, new_ckpt

# ── Run phase evaluation ──────────────────────────────────────────────────────
def run_phase_eval(phase_id, checkpoint_path=None):
    """Run capability evaluation for a phase."""
    print(f'\n  Evaluating phase {phase_id}...')
    try:
        eval_cmd = [
            sys.executable, os.path.join(REPO_ROOT, 'scripts/eval_milestone.py'),
            '--phase', str(phase_id), '--model-only',
        ]
        if checkpoint_path:
            eval_cmd += ['--checkpoint', checkpoint_path]
        result = subprocess.run(
            eval_cmd,
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
        )
        output = result.stdout
        # Print last 1500 chars (summary)
        print(output[-1500:] if len(output) > 1500 else output)
    except Exception as e:
        print(f'  Eval failed: {e}')

# ── Check session time ────────────────────────────────────────────────────────
def time_remaining():
    elapsed = time.time() - SESSION_START
    return SESSION_LIMIT_SEC - elapsed

def has_time_for_phase():
    """Check if we have enough time for another ~20min phase."""
    return time_remaining() > 30 * 60  # 30 min buffer

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print(f' YAYA TRUE AI — CURRICULUM TRAINING')
print(f' 16 phases x {STEPS_PER_PHASE} steps = {16 * STEPS_PER_PHASE:,} total steps')
print('=' * 60)

# Step 1: Ensure directories
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CURRICULUM_DIR, exist_ok=True)

# Step 2: Load progress (from Hub if needed)
progress = load_progress()
if hf_token and not progress.get('completed_phases'):
    try:
        from huggingface_hub import hf_hub_download
        prog_hub = hf_hub_download(
            repo_id=HUB_REPO, filename='curriculum_progress.json',
            repo_type='model', token=hf_token, local_dir='/tmp/hub_meta'
        )
        with open(prog_hub) as f:
            progress = json.load(f)
        save_progress(progress)
        print(f'  Progress restored from Hub: phases {progress.get("completed_phases", [])} done')
    except Exception:
        pass

# Step 3: Determine the RIGHT starting checkpoint based on progress
completed = progress.get('completed_phases', [])
local_ckpt = None

if hf_token:
    from scripts.hub_utils import pull_specific_checkpoint, ensure_repo
    ensure_repo(HUB_REPO, hf_token)

    if not completed:
        # Fresh start — find the best pretrained checkpoint
        # Prefer the new pretraining pipeline checkpoint over the old patch
        try:
            from huggingface_hub import hf_hub_download
            pp = hf_hub_download(HUB_REPO, 'pretrain_progress.json', repo_type='model', token=hf_token)
            pt_progress = json.load(open(pp))
            pt_ckpt = pt_progress.get('latest_checkpoint', '')
            if pt_ckpt:
                print(f'\n[1] Fresh start — pulling pretrained model ({pt_ckpt}, step {pt_progress.get("step",0)})...')
                local_ckpt = pull_specific_checkpoint(HUB_REPO, pt_ckpt, CKPT_DIR, hf_token)
        except Exception:
            pass
        if not local_ckpt:
            print(f'\n[1] Fresh start — pulling base model ({BASE_PRETRAIN_CKPT})...')
            local_ckpt = pull_specific_checkpoint(HUB_REPO, BASE_PRETRAIN_CKPT, CKPT_DIR, hf_token)
    else:
        # Resuming — try to find the checkpoint from the last completed phase
        last_phase = max(completed)
        phase_ckpt_name = f'phase{last_phase:02d}/checkpoint-{STEPS_PER_PHASE:08d}'
        print(f'\n[1] Resuming — pulling Phase {last_phase} checkpoint ({phase_ckpt_name})...')
        local_ckpt = pull_specific_checkpoint(HUB_REPO, phase_ckpt_name, CKPT_DIR, hf_token)
        if not local_ckpt:
            # Fallback: try any checkpoint in phase dir
            local_ckpt = find_latest_local_checkpoint(os.path.join(CKPT_DIR, f'phase{last_phase:02d}'))
        if not local_ckpt:
            # Last resort: pull whatever latest.json says
            print('  Fallback: pulling latest checkpoint from Hub...')
            from scripts.hub_utils import pull_latest_checkpoint
            local_ckpt = pull_latest_checkpoint(HUB_REPO, CKPT_DIR, hf_token)

if local_ckpt:
    print(f'  Starting checkpoint: {os.path.basename(local_ckpt)}')
else:
    print('\n[1] No checkpoint found — starting from scratch')

next_phase = get_next_phase(progress)
if next_phase is None:
    print('\nAll 16 curriculum phases complete!')
    print('Running final benchmark...')
    if local_ckpt:
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, 'scripts/eval_milestone.py'),
             '--all', '--model-only',
             '--save', os.path.join(REPO_ROOT, 'data/eval/curriculum_final.json')],
            cwd=REPO_ROOT
        )
    sys.exit(0)

completed = progress.get('completed_phases', [])
print(f'\n[2] Progress: {len(completed)}/16 phases complete')
print(f'  Completed: {completed if completed else "none"}')
print(f'  Next: Phase {next_phase["id"]} — {next_phase["name"]}')

# Step 4: Generate ALL curriculum data upfront (fast, <10s)
print('\n[3] Generating curriculum data...')
result = subprocess.run(
    [sys.executable, os.path.join(REPO_ROOT, 'scripts/generate_curriculum_data.py'),
     '--phase', '1-16'],
    cwd=REPO_ROOT, capture_output=True, text=True
)
print(result.stdout.strip())

# Step 5: Start checkpoint watcher
if hf_token:
    from scripts.hub_utils import start_watcher
    print('\n[4] Starting checkpoint watcher...')
    start_watcher(CKPT_DIR, HUB_REPO, hf_token, interval_sec=90)
else:
    print('\n[4] No HF_TOKEN — checkpoints local only')

# Step 6: Train phases sequentially
print(f'\n[5] Training — up to {MAX_PHASES_PER_SESSION} phases this session')
phases_trained = 0
current_ckpt = local_ckpt

while True:
    phase = get_next_phase(progress)
    if phase is None:
        print('\nAll phases complete!')
        break

    if phases_trained >= MAX_PHASES_PER_SESSION:
        print(f'\nReached max phases per session ({MAX_PHASES_PER_SESSION})')
        break

    if not has_time_for_phase():
        remaining = time_remaining() / 60
        print(f'\nInsufficient time remaining ({remaining:.0f} min) — stopping')
        break

    # Train this phase
    success, new_ckpt = run_phase(phase, current_ckpt, progress)

    if success and new_ckpt:
        current_ckpt = new_ckpt
        phases_trained += 1

        # Run eval (quick, ~2 min)
        run_phase_eval(phase['id'], checkpoint_path=current_ckpt)
    else:
        print(f'\nPhase {phase["id"]} failed — stopping session')
        break

# ── Push final progress to Hub ────────────────────────────────────────────────
if hf_token:
    try:
        from huggingface_hub import upload_file
        import io
        progress_json = json.dumps(progress, indent=2)
        upload_file(
            path_or_fileobj=io.BytesIO(progress_json.encode()),
            path_in_repo='curriculum_progress.json',
            repo_id=HUB_REPO, repo_type='model', token=hf_token,
            commit_message=f'Curriculum progress: {len(progress.get("completed_phases", []))}/16 phases'
        )
        print(f'\n[Hub] Progress pushed: {len(progress.get("completed_phases", []))}/16 phases')
    except Exception as e:
        print(f'\n[Hub] Progress push failed: {e}')

# ── Final push of latest checkpoint (blocking — last action, worth waiting) ──
if current_ckpt and hf_token:
    from scripts.hub_utils import push_checkpoint
    print('[Hub] Final checkpoint push (blocking, up to 5 min)...')
    push_checkpoint(current_ckpt, HUB_REPO, hf_token)

# ── Session summary ───────────────────────────────────────────────────────────
elapsed = (time.time() - SESSION_START) / 60
completed = progress.get('completed_phases', [])

print(f'\n{"="*60}')
print(f' SESSION SUMMARY')
print(f'{"="*60}')
print(f'  Phases trained this session: {phases_trained}')
print(f'  Total completed: {len(completed)}/16')
print(f'  Completed phases: {sorted(completed)}')
print(f'  Session time: {elapsed:.0f} min')

next_phase = get_next_phase(progress)
if next_phase:
    print(f'\n  Next session: Phase {next_phase["id"]} — {next_phase["name"]}')
    print(f'  Just re-run this notebook!')
else:
    print(f'\n  ALL 16 PHASES COMPLETE! Yaya is now a True AI.')

print()
sys.exit(0)
