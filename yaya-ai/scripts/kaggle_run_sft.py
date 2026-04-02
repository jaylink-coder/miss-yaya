"""Kaggle SFT runner for Yaya-125M.

One-time Kaggle setup (Settings → Secrets):
  - WANDB_API_KEY  — W&B live monitoring (optional but recommended)
  - HF_TOKEN       — HuggingFace token (optional, avoids rate limits)

Re-run safe: auto-resumes from latest checkpoint.

Dataset built automatically (~262K examples):
  GSM8K (8K) + MetaMath (100K) + OpenHermes (150K) + Yaya existing (~3.5K)
"""

import json
import os
import random
import sys
import glob
import subprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_CKPT_DIR   = '/kaggle/working/yaya-sft-checkpoints'
DATA_DIR       = os.path.join(REPO_ROOT, 'data/sft')
DATASET_PATH   = os.path.join(DATA_DIR, 'yaya_reasoning_large.jsonl')
TOKENIZER_PATH = os.path.join(REPO_ROOT, 'data/tokenizer/yaya_tokenizer.model')

PRETRAIN_CKPT_PATH = '/kaggle/input/yaya-checkpoints'  # upload pretrain ckpt here

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONIOENCODING']        = 'utf-8'
random.seed(42)

# ── Load Kaggle secrets ────────────────────────────────────────────────────────
def load_secret(name):
    val = os.environ.get(name)
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

hf_token    = load_secret('HF_TOKEN')
wandb_key   = load_secret('WANDB_API_KEY')

if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print(f'HF_TOKEN loaded.')
else:
    print('WARNING: No HF_TOKEN — HuggingFace downloads may be rate-limited.')

if wandb_key:
    os.environ['WANDB_API_KEY'] = wandb_key
    print('WANDB_API_KEY loaded — live monitoring enabled.')
else:
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE']     = 'disabled'
    print('No WANDB_API_KEY — W&B disabled.')

# ── GPU check ─────────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    cap     = props.major * 10 + props.minor
    DTYPE   = 'bfloat16' if cap >= 80 else 'float16'
    print(f'\nGPU: {props.name}  VRAM: {vram_gb:.1f}GB  →  {DTYPE}')
    if vram_gb < 12:
        print('WARNING: <12GB VRAM — consider reducing batch size.')
else:
    print('WARNING: No GPU — training will be very slow.')
    DTYPE = 'float32'

# ── Dataset helpers ───────────────────────────────────────────────────────────
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
    for fname in ['yaya_reasoning_combined.jsonl', 'yaya_instruct.jsonl']:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            print(f'  Loaded {len(samples)} from {fname}')
    return samples

def download_gsm8k(max_samples=8000):
    print('  Downloading GSM8K...')
    try:
        from datasets import load_dataset
        ds = load_dataset('openai/gsm8k', 'main', split='train')
        out = []
        for row in ds:
            q = row.get('question', '').strip()
            a = row.get('answer', '').strip()
            if q and a:
                out.append(make_sample(q, a))
            if len(out) >= max_samples:
                break
        print(f'    {len(out)} GSM8K examples')
        return out
    except Exception as e:
        print(f'    GSM8K failed: {e}')
        return []

def download_metamath(max_samples=100000):
    print('  Downloading MetaMath...')
    try:
        from datasets import load_dataset
        ds = load_dataset('meta-math/MetaMathQA', split='train')
        out = []
        for row in ds:
            q = row.get('query', '').strip()
            a = row.get('response', '').strip()
            if q and a:
                out.append(make_sample(q, a))
            if len(out) >= max_samples:
                break
        print(f'    {len(out)} MetaMath examples')
        return out
    except Exception as e:
        print(f'    MetaMath failed: {e}')
        return []

def download_openhermes(max_samples=150000):
    print('  Downloading OpenHermes-2.5...')
    try:
        from datasets import load_dataset
        ds = load_dataset('teknium/OpenHermes-2.5', split='train')
        out = []
        for row in ds:
            convs = row.get('conversations', [])
            if len(convs) < 2:
                continue
            # Find system, human, gpt turns
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
            if len(out) >= max_samples:
                break
        print(f'    {len(out)} OpenHermes examples')
        return out
    except Exception as e:
        print(f'    OpenHermes failed: {e}')
        return []

def build_dataset():
    print('\n[2/4] Building reasoning dataset...')
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, encoding='utf-8') as f:
            n = sum(1 for l in f if l.strip())
        if n >= 200_000:
            print(f'  Dataset already built: {n:,} examples — skipping.')
            return n

    all_samples = []
    all_samples.extend(load_existing_yaya())
    all_samples.extend(download_gsm8k())
    all_samples.extend(download_metamath())
    all_samples.extend(download_openhermes())

    # Deduplicate by first 100 chars of user message
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

    print(f'  Dataset saved: {len(deduped):,} examples → {DATASET_PATH}')
    return len(deduped)

# ── Find checkpoints ───────────────────────────────────────────────────────────
def find_latest_checkpoint(directory):
    ckpts = sorted(glob.glob(f'{directory}/checkpoint-*'))
    return ckpts[-1] if ckpts else None

# ── Main ──────────────────────────────────────────────────────────────────────
print('\n[1/4] GPU check done.')

# Step 2: Build dataset
n_samples = build_dataset()

# Step 3: Find checkpoints
print('\n[3/4] Locating checkpoints...')
pretrain_ckpt = find_latest_checkpoint(PRETRAIN_CKPT_PATH)
sft_ckpt      = find_latest_checkpoint(SFT_CKPT_DIR)

if sft_ckpt:
    print(f'  Resuming SFT from: {sft_ckpt}')
elif pretrain_ckpt:
    print(f'  Starting SFT from pretrain: {pretrain_ckpt}')
else:
    print('  No checkpoint — training from random init.')

os.makedirs(SFT_CKPT_DIR, exist_ok=True)

# Step 4: Patch config and launch
print('\n[4/4] Patching config and launching training...')
import yaml

config_path = os.path.join(REPO_ROOT, 'configs/training/sft_125m.yaml')
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cfg['checkpointing']['save_dir'] = SFT_CKPT_DIR
cfg['training']['dtype']         = DTYPE
cfg['data']['train_data']        = DATASET_PATH
cfg['data']['eval_data']         = DATASET_PATH
cfg['data']['tokenizer_path']    = TOKENIZER_PATH

if torch.cuda.is_available() and vram_gb < 14:
    cfg['distributed']['gradient_checkpointing'] = True

if wandb_key:
    cfg['logging']['wandb_project']  = 'yaya-ai'
    cfg['logging']['wandb_run_name'] = 'yaya-125m-sft'

with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

cmd = [
    sys.executable, 'scripts/train_sft.py',
    '--model_config', 'configs/model/yaya_125m.yaml',
    '--train_config', config_path,
]
if sft_ckpt:
    cmd += ['--resume', sft_ckpt]
elif pretrain_ckpt:
    cmd += ['--pretrain_checkpoint', pretrain_ckpt]

print(f'  Command: {" ".join(cmd)}')
print(f'  Dataset: {n_samples:,} examples')
print(f'  Checkpoint dir: {SFT_CKPT_DIR}')
print()

result = subprocess.run(cmd)
sys.exit(result.returncode)
