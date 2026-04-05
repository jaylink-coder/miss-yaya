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
        from huggingface_hub import list_repo_files, snapshot_download
        # Scan Hub — priority: recovery > dpo > sft
        hub_files = list(list_repo_files(repo_id=HUB_REPO, repo_type='model', token=hf_token))
        hub_ckpt_names = sorted({
            f.split('/')[0] for f in hub_files
            if '/' in f and '_temp' not in f
            and any(f.split('/')[0].startswith(p) for p in
                    ('recovery-checkpoint-', 'dpo-checkpoint-', 'checkpoint-'))
        })
        hub_ckpt = None
        for prefix in ('recovery-checkpoint-', 'dpo-checkpoint-', 'checkpoint-'):
            matches = [c for c in hub_ckpt_names if c.startswith(prefix)]
            if matches:
                hub_ckpt = matches[-1]
                break
        if hub_ckpt:
            print(f'  Downloading {hub_ckpt} from Hub...')
            dl_dir = RECOVERY_CKPT if hub_ckpt.startswith('recovery-') else DPO_CKPT_DIR
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
                print(f'  WARNING: download succeeded but {local_path} not found')
        else:
            print('  No suitable checkpoint found on Hub.')
    except Exception as e:
        print(f'  Hub pull failed: {e}')

if not start_ckpt:
    print('ERROR: No checkpoint found. Cannot run recovery.')
    sys.exit(1)

print(f'  Starting from: {start_ckpt}')

# Check if recovery is already complete — skip training if so
_meta_path = os.path.join(start_ckpt, 'metadata.json')
_already_done = False
if os.path.exists(_meta_path):
    import json as _json
    _meta = _json.load(open(_meta_path))
    if _meta.get('step', 0) >= 5000 and os.path.basename(start_ckpt).startswith('recovery-'):
        print(f'  Recovery already complete at step {_meta["step"]} (loss={_meta.get("loss","?")})')
        print('  Skipping training — going straight to DPO2 + benchmark.')
        _already_done = True
        training_ok = True

# ── Step 1b: Generate fresh data (anti-list DPO + concise SFT) ────────────────
print('\n[1b/4] Generating anti-list DPO pairs and format-enforcement SFT data...')
for gen_script in ['scripts/generate_antlist_dpo.py', 'scripts/add_format_enforcement.py']:
    script_path = os.path.join(REPO_ROOT, gen_script)
    if os.path.exists(script_path):
        r = subprocess.run([sys.executable, script_path], cwd=REPO_ROOT)
        if r.returncode != 0:
            print(f'  WARNING: {gen_script} failed (non-fatal, will use existing data)')
    else:
        print(f'  WARNING: {gen_script} not found — skipping')


if _already_done:
    training_ok = True
else:
    # ── Step 2: Build recovery dataset ────────────────────────────────────────
    print('\n[2/4] Building recovery dataset...')

    RECOVERY_DATA    = os.path.join(DATA_DIR, 'yaya_recovery.jsonl')
    SHORT_QA         = os.path.join(DATA_DIR, 'yaya_short_qa.jsonl')
    QUICK_FACTS      = os.path.join(DATA_DIR, 'teach/quick_facts.jsonl')
    CONCISE_SFT      = os.path.join(DATA_DIR, 'yaya_concise_sft.jsonl')
    INSTRUCT_CLEAN   = os.path.join(DATA_DIR, 'yaya_instruct_clean.jsonl')
    FACTUAL_QA       = os.path.join(DATA_DIR, 'yaya_factual_qa.jsonl')
    QA_FOCUSED       = os.path.join(DATA_DIR, 'yaya_qa_focused.jsonl')

    # Base sources (1x) — diverse to prevent memorization of narrow set
    base_sources = [SHORT_QA, QUICK_FACTS, CONCISE_SFT, FACTUAL_QA, QA_FOCUSED]
    sources = []
    for src in base_sources:
        if os.path.exists(src):
            with open(src, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f if l.strip()]
            sources.extend(lines)
            print(f'  + {os.path.basename(src)}: {len(lines)} examples')
        else:
            print(f'  ! Missing: {os.path.basename(src)}')

    # Add non-chess instruct_clean examples (diverse Q&A, ~13% lists — acceptable)
    if os.path.exists(INSTRUCT_CLEAN):
        import json as _json
        with open(INSTRUCT_CLEAN, encoding='utf-8', errors='replace') as f:
            ic_lines = [l.strip() for l in f if l.strip()]
        # Filter out chess (they add domain-specific noise)
        ic_filtered = []
        for l in ic_lines:
            try:
                obj = _json.loads(l)
                sys_content = obj.get('messages', [{}])[0].get('content', '').lower()
                if 'chess' not in sys_content:
                    ic_filtered.append(l)
            except Exception:
                pass
        sources.extend(ic_filtered)
        print(f'  + yaya_instruct_clean.jsonl (non-chess): {len(ic_filtered)} examples')

    # 2x oversample only short_qa + quick_facts — enough signal without memorization
    boosted = []
    for src in [SHORT_QA, QUICK_FACTS]:
        if os.path.exists(src):
            with open(src, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f if l.strip()]
            boosted.extend(lines * 2)
            print(f'  + {os.path.basename(src)}: 2x boost = {len(lines) * 2} examples')

    import random; random.seed(42)
    all_examples = sources + boosted
    random.shuffle(all_examples)
    with open(RECOVERY_DATA, 'w', encoding='utf-8') as f:
        for line in all_examples:
            f.write(line + '\n')
    print(f'  Recovery dataset: {len(all_examples)} examples (2x oversample, diverse sources)')
    print(f'  Saved → {RECOVERY_DATA}')

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    print('\n[3/4] Launching recovery training...')
    os.makedirs(RECOVERY_CKPT, exist_ok=True)

    import yaml
    base_cfg_path = os.path.join(REPO_ROOT, 'configs/training/sft_125m.yaml')
    with open(base_cfg_path) as f:
        recovery_cfg = yaml.safe_load(f)
    recovery_cfg['training']['max_steps']      = 5000
    recovery_cfg['training']['learning_rate']  = 5e-5
    recovery_cfg['training']['max_seq_length'] = 128
    recovery_cfg['training']['dtype']          = DTYPE
    recovery_cfg['checkpointing'] = recovery_cfg.get('checkpointing', {})
    recovery_cfg['checkpointing']['save_steps'] = 250
    recovery_cfg['checkpointing']['save_dir']   = RECOVERY_CKPT
    recovery_cfg['data'] = recovery_cfg.get('data', {})
    recovery_cfg['data']['train_data'] = RECOVERY_DATA

    tmp_cfg_path = '/kaggle/working/recovery_train_config.yaml'
    with open(tmp_cfg_path, 'w') as f:
        yaml.dump(recovery_cfg, f)

    train_cmd = [
        sys.executable, os.path.join(REPO_ROOT, 'scripts/train_sft.py'),
        '--model_config', os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml'),
        '--train_config', tmp_cfg_path,
        '--pretrain_checkpoint', start_ckpt,
    ]

    if HF_TOKEN:
        try:
            from scripts.hub_utils import start_watcher, ensure_repo
            ensure_repo(HUB_REPO, hf_token)
            _watcher = start_watcher(RECOVERY_CKPT, HUB_REPO, hf_token, interval_sec=120)
        except Exception as e:
            print(f'  WARNING: Hub watcher failed to start: {e}')

    print(f'  Command: {" ".join(train_cmd[:4])} ...')
    result = subprocess.run(train_cmd, cwd=REPO_ROOT)
    training_ok = result.returncode == 0

    if not training_ok:
        print('  train_sft.py failed — driving trainer directly...')
        try:
            from src.training.trainer import Trainer
            from src.model.yaya_model import YayaForCausalLM
            from src.utils.config import load_model_config
            from src.tokenizer.tokenizer import YayaTokenizer
            from src.data.dataset import InstructionDataset
            from src.data.dataloader import create_dataloader

            model_cfg = load_model_config(os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml'))
            tokenizer = YayaTokenizer(TOKENIZER_PATH)
            model = YayaForCausalLM(model_cfg)
            ckpt_file = os.path.join(start_ckpt, 'model.pt')
            state = torch.load(ckpt_file, map_location='cpu')
            model.load_state_dict(state.get('model', state), strict=False)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            dataset = InstructionDataset(RECOVERY_DATA, tokenizer, max_seq_length=128)
            from src.utils.config import TrainingConfig as SrcTrainingConfig
            train_cfg = SrcTrainingConfig(
                max_steps=5000, learning_rate=5e-5, per_device_batch_size=4,
                gradient_accumulation_steps=4, save_steps=250,
                save_dir=RECOVERY_CKPT, dtype=DTYPE,
            )
            train_loader = create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0)
            Trainer(model, train_cfg, train_loader, tokenizer=tokenizer).train()
            training_ok = True
        except Exception as e:
            print(f'  Fallback trainer also failed: {e}')
            training_ok = False


# ── Step 4: Benchmark ─────────────────────────────────────────────────────────
print('\n[4/4] Running benchmark on recovery checkpoint...')

# If already done, use the downloaded recovery checkpoint directly
if _already_done:
    best_ckpt = start_ckpt
    print(f'  Using pre-downloaded recovery checkpoint: {best_ckpt}')
else:
    # Find the best recovery checkpoint — check all possible save locations
    _candidate_dirs = [
        RECOVERY_CKPT,
        os.path.join(REPO_ROOT, 'checkpoints/yaya-125m-sft'),
        '/kaggle/working/yaya-sft-checkpoints',
        os.path.join(REPO_ROOT, 'checkpoints/yaya-125m-recovery'),
    ]
    recovery_ckpts = []
    for _d in _candidate_dirs:
        recovery_ckpts.extend(glob.glob(os.path.join(_d, 'checkpoint-*')))
    recovery_ckpts = sorted(set(recovery_ckpts))
    best_ckpt = recovery_ckpts[-1] if recovery_ckpts else None

if best_ckpt:
    print(f'  Found recovery checkpoint: {best_ckpt}')

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

    # ── Second DPO run on recovery checkpoint ─────────────────────────────────
    print('\n[4b/4] Running second DPO alignment on recovery checkpoint...')
    dpo_data     = os.path.join(DATA_DIR, 'yaya_dpo_combined.jsonl')
    dpo2_ckpt    = '/kaggle/working/yaya-dpo2-checkpoints'
    os.makedirs(dpo2_ckpt, exist_ok=True)

    if os.path.exists(dpo_data):
        dpo_cmd = [
            sys.executable, os.path.join(REPO_ROOT, 'scripts/train_dpo.py'),
            '--sft_checkpoint', best_ckpt,
            '--dpo_data',       dpo_data,
            '--tokenizer',      TOKENIZER_PATH,
            '--save_dir',       dpo2_ckpt,
            '--lr',             '3e-7',   # lower LR — don't overwrite recovery gains
            '--max_steps',      '1500',   # shorter — targeted alignment only
            '--batch_size',     '4',
        ]
        # Start DPO2 watcher so checkpoints survive Kaggle crash
        if HF_TOKEN:
            try:
                from scripts.hub_utils import start_watcher
                _dpo2_watcher = start_watcher(dpo2_ckpt, HUB_REPO, hf_token, interval_sec=120)
            except Exception as _we:
                print(f'  WARNING: DPO2 hub watcher failed: {_we}')

        print(f'  DPO2 command: {" ".join(dpo_cmd[:4])} ...')
        dpo2_result = subprocess.run(dpo_cmd, cwd=REPO_ROOT)

        if dpo2_result.returncode == 0:
            print('Second DPO complete!')
            dpo2_ckpts = sorted(glob.glob(os.path.join(dpo2_ckpt, 'checkpoint-*')))
            if dpo2_ckpts:
                best_ckpt = dpo2_ckpts[-1]  # benchmark the DPO2 checkpoint
                # Push DPO2 to Hub
                if HF_TOKEN:
                    try:
                        from huggingface_hub import upload_folder, upload_file
                        import io
                        dpo2_name = 'dpo2-' + os.path.basename(best_ckpt)
                        upload_folder(
                            folder_path=best_ckpt,
                            repo_id=HUB_REPO, path_in_repo=dpo2_name,
                            repo_type='model', token=hf_token,
                            ignore_patterns=['optimizer.pt'],
                            commit_message=f'DPO2 checkpoint: {dpo2_name}',
                        )
                        upload_file(
                            path_or_fileobj=io.BytesIO(f'{{"latest": "{dpo2_name}"}}'.encode()),
                            path_in_repo='latest.json', repo_id=HUB_REPO,
                            repo_type='model', token=hf_token,
                            commit_message=f'Update latest → {dpo2_name}',
                        )
                        print(f'[Hub] DPO2 checkpoint pushed → {HUB_REPO}/{dpo2_name}')
                    except Exception as e:
                        print(f'[Hub] DPO2 push failed (non-fatal): {e}')
        else:
            print('Second DPO failed — benchmarking recovery checkpoint instead.')
    else:
        print(f'  DPO data not found at {dpo_data} — skipping second DPO.')

    # ── Benchmark ──────────────────────────────────────────────────────────────
    print('\n[4c/4] Running final benchmark...')
    bench_cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, 'scripts/benchmark.py'),
        '--checkpoint', best_ckpt,
    ]
    subprocess.run(bench_cmd, cwd=REPO_ROOT)

    # ── Push README model card to Hub ──────────────────────────────────────────
    if HF_TOKEN:
        try:
            from huggingface_hub import upload_file
            readme = os.path.join(REPO_ROOT, 'README.md')
            if os.path.exists(readme):
                upload_file(
                    path_or_fileobj=readme,
                    path_in_repo='README.md',
                    repo_id=HUB_REPO, repo_type='model', token=hf_token,
                    commit_message='Update model card',
                )
                print(f'[Hub] Model card pushed → {HUB_REPO}')
        except Exception as e:
            print(f'[Hub] Model card push failed (non-fatal): {e}')

else:
    print('  No recovery checkpoint found — skipping DPO2 and benchmark.')
    training_ok = False

print()
print('=' * 60)
print(' RECOVERY + DPO2 COMPLETE' if training_ok else ' RECOVERY FAILED')
print('=' * 60)
sys.exit(0 if training_ok else 1)
