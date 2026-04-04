"""Kaggle SFT runner for Yaya-125M — milestone-based, crash-resistant.

Each Kaggle session runs one training phase (~5-15K steps).
Checkpoints are pushed to HuggingFace Hub after every save, so
no progress is lost when the session ends.

One-time Kaggle setup (Settings → Secrets):
  WANDB_API_KEY  — W&B live monitoring (optional but recommended)
  HF_TOKEN       — HuggingFace token (REQUIRED for checkpoint persistence)

How it works:
  1. Pulls latest checkpoint from HF Hub (or starts fresh)
  2. Detects which milestone phase we're in based on current step
  3. Builds/verifies the 262K reasoning dataset
  4. Launches training for this phase's steps with a checkpoint watcher
     that pushes every new checkpoint to HF Hub in the background
  5. After training: runs eval, logs milestone progress, boosts weak areas
  6. Next session: repeats from step 1, resuming from where we left off

Dataset (~262K examples built automatically on first run):
  GSM8K (8K) + MetaMath (100K) + OpenHermes (150K) + Yaya existing (~3.5K)
"""

import json
import os
import random
import re
import sys
import glob
import subprocess
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_CKPT_DIR    = '/kaggle/working/yaya-sft-checkpoints'
DATA_DIR        = os.path.join(REPO_ROOT, 'data/sft')
DATASET_PATH    = os.path.join(DATA_DIR, 'yaya_reasoning_large.jsonl')
TOKENIZER_PATH  = os.path.join(REPO_ROOT, 'data/tokenizer/yaya_tokenizer.model')
MILESTONES_PATH = os.path.join(REPO_ROOT, 'configs/training/milestones.yaml')
PROGRESS_PATH   = os.path.join(REPO_ROOT, 'docs/training_milestones.json')

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONIOENCODING']        = 'utf-8'
random.seed(42)

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
wandb_key = load_secret('WANDB_API_KEY')

if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print('HF_TOKEN loaded.')
else:
    print('WARNING: No HF_TOKEN — checkpoints will NOT persist across sessions!')

if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key
    print('WANDB_API_KEY loaded — live monitoring enabled.')
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
    print(f'\nGPU: {props.name}  VRAM: {vram_gb:.1f}GB  →  {DTYPE}')
else:
    print('WARNING: No GPU.')
    vram_gb = 0
    DTYPE   = 'float32'

# ── Milestone loading ─────────────────────────────────────────────────────────
import yaml

with open(MILESTONES_PATH) as f:
    milestone_cfg = yaml.safe_load(f)

HUB_REPO   = milestone_cfg.get('hub_repo', 'jaylinkcoder/yaya-125m')
PHASES     = milestone_cfg['phases']
TOTAL_STEPS = milestone_cfg.get('total_steps', 40000)

def get_phase(step):
    for phase in PHASES:
        if phase.get('phase_type') == 'dpo':
            continue
        if phase['step_start'] <= step < phase['step_end']:
            return phase
    return None  # training complete

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"phases": {}, "total_steps_done": 0}

def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def find_latest_local_checkpoint(directory):
    # Skip _temp checkpoints — they are incomplete writes from crashed sessions
    ckpts = sorted([c for c in glob.glob(os.path.join(directory, 'checkpoint-*'))
                    if not c.endswith('_temp')])
    return ckpts[-1] if ckpts else None

def get_step_from_checkpoint(ckpt_path):
    """Extract step number from checkpoint directory name."""
    if not ckpt_path:
        return 0
    m = re.search(r'checkpoint-0*(\d+)', os.path.basename(ckpt_path))
    return int(m.group(1)) if m else 0

# ── Dataset building ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly and are always honest."
)

def make_sample(user_msg, assistant_msg, system=SYSTEM_PROMPT):
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user_msg.strip()},
        {"role": "assistant", "content": assistant_msg.strip()},
    ]}

def load_existing_yaya():
    samples = []
    for fname in ['yaya_reasoning_combined.jsonl', 'yaya_instruct.jsonl', 'yaya_short_qa.jsonl']:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
    print(f'  Existing Yaya data: {len(samples):,} examples')
    return samples

def download_gsm8k():
    print('  Downloading GSM8K...')
    try:
        from datasets import load_dataset
        ds = load_dataset('openai/gsm8k', 'main', split='train')
        out = [make_sample(r['question'], r['answer']) for r in ds if r.get('question') and r.get('answer')]
        print(f'    {len(out):,} examples')
        return out
    except Exception as e:
        print(f'    GSM8K failed: {e}')
        return []

def download_metamath(max_n=100000):
    print('  Downloading MetaMath...')
    try:
        from datasets import load_dataset
        ds = load_dataset('meta-math/MetaMathQA', split='train')
        out = []
        for r in ds:
            if r.get('query') and r.get('response'):
                out.append(make_sample(r['query'], r['response']))
                if len(out) >= max_n:
                    break
        print(f'    {len(out):,} examples')
        return out
    except Exception as e:
        print(f'    MetaMath failed: {e}')
        return []

def download_openhermes(max_n=150000):
    print('  Downloading OpenHermes-2.5...')
    try:
        from datasets import load_dataset
        ds = load_dataset('teknium/OpenHermes-2.5', split='train')
        out = []
        for r in ds:
            convs = r.get('conversations', [])
            system = SYSTEM_PROMPT
            user_msg = assistant_msg = ''
            for turn in convs:
                role = turn.get('from', '')
                val  = turn.get('value', '').strip()
                if role == 'system':
                    system = val or SYSTEM_PROMPT
                elif role == 'human' and not user_msg:
                    user_msg = val
                elif role == 'gpt' and not assistant_msg:
                    assistant_msg = val
            if user_msg and assistant_msg:
                out.append(make_sample(user_msg, assistant_msg, system=system))
            if len(out) >= max_n:
                break
        print(f'    {len(out):,} examples')
        return out
    except Exception as e:
        print(f'    OpenHermes failed: {e}')
        return []

def build_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if we already have a large enough local dataset.
    # Also check that short_qa data is already included (fingerprint via line count).
    short_qa_path = os.path.join(DATA_DIR, 'yaya_short_qa.jsonl')
    short_qa_lines = 0
    if os.path.exists(short_qa_path):
        with open(short_qa_path, encoding='utf-8') as f:
            short_qa_lines = sum(1 for l in f if l.strip())

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, encoding='utf-8') as f:
            n = sum(1 for l in f if l.strip())
        # Accept cached dataset only if it's large enough AND already contains short_qa
        # (short_qa has ~2634 lines; if dataset < n+short_qa it was built before short_qa existed)
        if n >= 200_000 and (short_qa_lines == 0 or n >= 205_000):
            print(f'  Dataset ready: {n:,} examples')
            return n
        if short_qa_lines > 0 and n < 205_000:
            print(f'  Dataset missing short Q&A data ({n:,} examples) — rebuilding with short_qa...')
        else:
            print(f'  Dataset too small ({n:,}) — rebuilding...')

    # Try downloading cached dataset from HF Hub (much faster than re-downloading sources)
    if HF_TOKEN:
        try:
            from huggingface_hub import hf_hub_download
            print('  Trying to pull cached dataset from HF Hub...')
            cached = hf_hub_download(
                repo_id=HUB_REPO, filename='dataset/yaya_reasoning_large.jsonl',
                repo_type='model', token=HF_TOKEN, local_dir=DATA_DIR,
            )
            import shutil
            shutil.copy(cached, DATASET_PATH)
            with open(DATASET_PATH, encoding='utf-8') as f:
                n = sum(1 for l in f if l.strip())
            if n >= 200_000:
                print(f'  Dataset pulled from Hub: {n:,} examples (saved ~10 min)')
                return n
            print(f'  Hub dataset too small ({n:,}) — rebuilding from scratch...')
        except Exception as e:
            print(f'  Hub dataset not found ({e}) — building from scratch...')

    print('\nBuilding reasoning dataset...')
    all_samples = load_existing_yaya() + download_gsm8k() + download_metamath() + download_openhermes()

    # Deduplicate
    seen, deduped = set(), []
    for s in all_samples:
        key = s['messages'][1]['content'][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    random.shuffle(deduped)

    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f'  Dataset saved: {len(deduped):,} examples')

    # Push to HF Hub for future sessions to reuse
    if HF_TOKEN:
        try:
            from huggingface_hub import upload_file
            print('  Pushing dataset to HF Hub for future sessions...')
            upload_file(
                path_or_fileobj=DATASET_PATH,
                path_in_repo='dataset/yaya_reasoning_large.jsonl',
                repo_id=HUB_REPO, repo_type='model', token=HF_TOKEN,
            )
            print(f'  [Hub] Dataset cached → {HUB_REPO}/dataset/yaya_reasoning_large.jsonl')
        except Exception as e:
            print(f'  [Hub] Dataset cache push failed (non-fatal): {e}')

    return len(deduped)

# ── Continuous learning: boost weak areas ─────────────────────────────────────
def boost_weak_areas(latest_ckpt, current_phase_id):
    """Run eval, find weak areas, oversample those examples into the dataset."""
    if not latest_ckpt or not os.path.isdir(latest_ckpt):
        return

    print('\n[Continuous Learning] Running eval to find weak areas...')

    boost_path = os.path.join(DATA_DIR, 'yaya_boost.jsonl')

    try:
        result = subprocess.run(
            [sys.executable, 'scripts/eval_math.py', '--checkpoint', latest_ckpt],
            capture_output=True, text=True, timeout=300, cwd=REPO_ROOT
        )
        output = result.stdout
        print(output[-2000:] if len(output) > 2000 else output)
    except Exception as e:
        print(f'  Eval failed: {e}')
        return

    # Parse per-stage scores from output like "Stage 1: 2/3 (67%)"
    weak_stages = []
    for m in re.finditer(r'[Ss]tage\s+(\d+).*?(\d+)/(\d+)', output):
        stage_id  = int(m.group(1))
        passed    = int(m.group(2))
        total     = int(m.group(3))
        score     = passed / total if total > 0 else 0
        if score < 0.5:
            weak_stages.append(stage_id)

    if not weak_stages:
        print('  No weak stages found — no boost needed.')
        return

    print(f'  Weak stages: {weak_stages} — loading boost examples...')

    # Load original dataset and oversample from topics matching weak stages
    # Map stage → keywords to filter examples
    stage_keywords = {
        1: ['multiply', 'sqrt', 'square root', 'arithmetic', '×', '+', '-'],
        2: ['fraction', 'percent', 'decimal', '%', '/'],
        3: ['solve for', 'equation', 'algebra', 'expression', 'x ='],
        4: ['quadratic', 'slope', 'logarithm', 'log', 'graph'],
        5: ['area', 'circle', 'triangle', 'pythagorean', 'geometry', 'angle'],
        6: ['mean', 'median', 'probability', 'statistics', 'average'],
        7: ['speed', 'distance', 'interest', 'rate', 'time'],
        8: ['derivative', 'integral', 'limit', 'calculus'],
    }

    boost_samples = []
    with open(DATASET_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                user_msg = s['messages'][1]['content'].lower()
                for stage_id in weak_stages:
                    keywords = stage_keywords.get(stage_id, [])
                    if any(kw in user_msg for kw in keywords):
                        boost_samples.append(s)
                        break
            except Exception:
                continue

    if not boost_samples:
        print('  No matching boost examples found in dataset.')
        return

    # Take up to 2K boost samples, repeated to fill
    n_boost = min(len(boost_samples), 2000)
    random.shuffle(boost_samples)
    boost_samples = boost_samples[:n_boost]

    with open(boost_path, 'w', encoding='utf-8') as f:
        for s in boost_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f'  Wrote {len(boost_samples)} boost examples → {boost_path}')
    print(f'  These will be included in the next training phase.')

    # Append boost examples to the main dataset for next run
    with open(DATASET_PATH, 'a', encoding='utf-8') as f:
        for s in boost_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f'  Appended to main dataset.')

# ── Main ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print(' YAYA-125M TRAINING — MILESTONE-BASED')
print('='*60)

# Step 1: Restore checkpoint from HF Hub if needed
os.makedirs(SFT_CKPT_DIR, exist_ok=True)
local_ckpt = find_latest_local_checkpoint(SFT_CKPT_DIR)

if not local_ckpt and hf_token:
    print('\n[1/5] No local checkpoint — pulling from HF Hub...')
    from scripts.hub_utils import pull_latest_checkpoint, ensure_repo
    ensure_repo(HUB_REPO, hf_token)
    local_ckpt = pull_latest_checkpoint(HUB_REPO, SFT_CKPT_DIR, hf_token)
else:
    if local_ckpt:
        print(f'\n[1/5] Local checkpoint found: {os.path.basename(local_ckpt)}')
    else:
        print('\n[1/5] Starting fresh (no HF_TOKEN for hub restore).')

# Step 2: Determine current phase
current_step  = get_step_from_checkpoint(local_ckpt)
current_phase = get_phase(current_step)

if current_phase is None:
    print(f'\nTraining complete at step {current_step}/{TOTAL_STEPS}! Running eval...')
    if local_ckpt:
        boost_weak_areas(local_ckpt, phase_id=5)
    sys.exit(0)

print(f'\n[2/5] Current step: {current_step}')
print(f'  Phase {current_phase["id"]}: "{current_phase["name"]}"')
print(f'  Steps this phase: {current_step} → {current_phase["step_end"]}')
print(f'  Loss target: < {current_phase["target_loss"]}')
print(f'  Eval target: {current_phase["eval_target"]*100:.0f}%')

# Step 3: Build dataset
print('\n[3/5] Dataset...')
n_samples = build_dataset()

# Step 4: Start HF Hub checkpoint watcher
if hf_token:
    from scripts.hub_utils import start_watcher
    print('\n[4/5] Starting checkpoint watcher...')
    start_watcher(SFT_CKPT_DIR, HUB_REPO, hf_token, interval_sec=90)
else:
    print('\n[4/5] No HF_TOKEN — checkpoints saved locally only (will be lost at session end).')

# Step 5: Patch config and train
print('\n[5/5] Launching training...')

config_path = os.path.join(REPO_ROOT, 'configs/training/sft_125m.yaml')
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cfg['checkpointing']['save_dir']  = SFT_CKPT_DIR
cfg['checkpointing']['save_steps'] = 500    # save every 500 steps (not 1000)
cfg['training']['dtype']           = DTYPE
cfg['training']['max_steps']       = current_phase['step_end']  # run until phase end
cfg['data']['train_data']          = DATASET_PATH
cfg['data']['eval_data']           = DATASET_PATH
cfg['data']['tokenizer_path']      = TOKENIZER_PATH

if torch.cuda.is_available() and vram_gb < 14:
    cfg['distributed']['gradient_checkpointing'] = True

if wandb_key:
    cfg['logging']['wandb_project']  = 'yaya-ai'
    cfg['logging']['wandb_run_name'] = f'yaya-125m-phase{current_phase["id"]}'

with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

cmd = [sys.executable, 'scripts/train_sft.py',
       '--model_config', 'configs/model/yaya_125m.yaml',
       '--train_config', config_path]

if local_ckpt:
    cmd += ['--resume', local_ckpt]

print(f'  Steps: {current_step} → {current_phase["step_end"]}')
print(f'  Dataset: {n_samples:,} examples')
print(f'  Save every 500 steps → {SFT_CKPT_DIR}')
if hf_token:
    print(f'  Checkpoints auto-pushed to: {HUB_REPO}')
print()

result = subprocess.run(cmd, cwd=REPO_ROOT)
training_ok = result.returncode == 0

# ── Post-training: eval + continuous learning ─────────────────────────────────
print('\n' + '='*60)
print(' POST-PHASE: EVAL + CONTINUOUS LEARNING')
print('='*60)

latest_ckpt = find_latest_local_checkpoint(SFT_CKPT_DIR)
final_step  = get_step_from_checkpoint(latest_ckpt)

# Push final checkpoint immediately (don't wait for watcher)
if latest_ckpt and hf_token:
    from scripts.hub_utils import push_checkpoint
    push_checkpoint(latest_ckpt, HUB_REPO, hf_token)

# Boost weak areas (runs eval, appends to dataset)
boost_weak_areas(latest_ckpt, current_phase['id'])

# Log milestone progress
progress = load_progress()
progress['phases'][str(current_phase['id'])] = {
    'name':        current_phase['name'],
    'status':      'complete' if training_ok else 'interrupted',
    'step_start':  current_step,
    'step_end':    final_step,
    'completed_at': datetime.utcnow().isoformat(),
    'checkpoint':  os.path.basename(latest_ckpt) if latest_ckpt else None,
}
progress['total_steps_done'] = final_step
save_progress(progress)
print(f'\nMilestone progress saved → {PROGRESS_PATH}')

next_phase = get_phase(final_step)
if next_phase:
    print(f'\nNext session: Phase {next_phase["id"]} — "{next_phase["name"]}"')
    print(f'  Steps: {final_step} → {next_phase["step_end"]}')
    print(f'  Just re-run this notebook — it will auto-resume.')
elif final_step >= TOTAL_STEPS:
    print(f'\nAll {TOTAL_STEPS} SFT steps complete! Starting DPO alignment...')
    dpo_data = os.path.join(REPO_ROOT, 'data/sft/yaya_dpo_combined.jsonl')
    dpo_ckpt_dir = '/kaggle/working/yaya-dpo-checkpoints'
    if os.path.exists(dpo_data):
        dpo_cmd = [
            sys.executable, os.path.join(REPO_ROOT, 'scripts/train_dpo.py'),
            '--model_config',   os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml'),
            '--sft_checkpoint', latest_ckpt,
            '--dpo_data',       dpo_data,
            '--tokenizer',      os.path.join(REPO_ROOT, 'data/tokenizer/yaya_tokenizer.model'),
            '--save_dir',       dpo_ckpt_dir,
            '--lr',             '5e-7',
            '--max_steps',      '2500',
            '--batch_size',     '4',
        ]
        print(f'  DPO command: {" ".join(dpo_cmd)}')
        import subprocess
        dpo_result = subprocess.run(dpo_cmd, cwd=REPO_ROOT)
        if dpo_result.returncode == 0:
            print('DPO alignment complete!')
            # Push DPO checkpoint to Hub
            import glob as _glob
            dpo_ckpts = sorted(_glob.glob(os.path.join(dpo_ckpt_dir, 'checkpoint-*')))
            if dpo_ckpts:
                push_checkpoint(dpo_ckpts[-1], HUB_REPO, HF_TOKEN)
                print(f'[Hub] DPO checkpoint pushed → {HUB_REPO}')
        else:
            print('DPO training failed — check logs.')
    else:
        print(f'  DPO data not found at {dpo_data} — skipping DPO phase.')
        print(f'  To run manually: python scripts/train_dpo.py --sft_checkpoint {latest_ckpt}')

sys.exit(0 if training_ok else 1)
