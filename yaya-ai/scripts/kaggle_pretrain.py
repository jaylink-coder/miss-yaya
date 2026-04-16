"""Yaya-125M Pretraining on Kaggle T4.

Downloads a large text corpus from HuggingFace, tokenizes it,
and runs causal language model pretraining with checkpoint persistence
to HuggingFace Hub for resume across sessions.

Usage (Kaggle notebook):
    python -u scripts/kaggle_pretrain.py

Environment:
    HF_TOKEN  — HuggingFace token (Kaggle secret)
    GPU       — Tesla T4 (Kaggle free tier)
    Session   — 3.5 hours max
"""

import json
import os
import sys
import time
import subprocess
import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

CKPT_DIR       = '/kaggle/working/yaya-pretrain-checkpoints'
DATA_DIR       = '/kaggle/working/pretrain-data'
TOKENIZER_PATH = os.path.join(REPO_ROOT, 'data/tokenizer/yaya_tokenizer.model')
MODEL_CONFIG   = os.path.join(REPO_ROOT, 'configs/model/yaya_125m.yaml')
HUB_REPO       = 'Jaylink-coder/yaya-125m'

# ── Training hyperparameters ──────────────────────────────────────────────────
# Optimized for T4 (16GB VRAM), 3h sessions
TOTAL_STEPS        = 20000    # ~640M tokens at 32K tokens/step → enough for coherent English
STEPS_PER_SESSION  = 7000     # ~3h on T4 at ~1.5 sec/step
BATCH_SIZE         = 8
GRAD_ACCUM         = 4        # effective batch = 32 * 1024 = 32K tokens/step
SEQ_LEN            = 1024
LR                 = 3e-4     # standard pretrain LR for 125M model
WARMUP             = 500
SAVE_STEPS         = 1000
EVAL_STEPS         = 2000
LOG_STEPS          = 50

SESSION_START = time.time()

# ── Secrets ───────────────────────────────────────────────────────────────────
def get_hf_token():
    """Get HF token from Kaggle secrets or environment."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get('HF_TOKEN', '')


def get_gpu_info():
    """Detect GPU and return dtype."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        print(f'GPU: {name}  VRAM: {vram:.1f}GB  dtype: {dtype}')
        return dtype
    print('WARNING: No GPU detected — training will be very slow')
    return 'float32'


# ── Data preparation ──────────────────────────────────────────────────────────
def download_and_tokenize():
    """Download WikiText-103 from HuggingFace and tokenize into .bin shards."""
    train_dir = os.path.join(DATA_DIR, 'train')
    eval_dir  = os.path.join(DATA_DIR, 'eval')
    train_shard = os.path.join(train_dir, 'shard_00000.bin')

    if os.path.exists(train_shard):
        size = os.path.getsize(train_shard) // 2  # uint16
        print(f'[Data] Already tokenized: {size:,} train tokens')
        return

    print('[Data] Downloading WikiText-103 from HuggingFace...')
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', trust_remote_code=True)

    print('[Data] Loading tokenizer...')
    from src.tokenizer.tokenizer import YayaTokenizer
    tokenizer = YayaTokenizer(TOKENIZER_PATH)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Tokenize train split
    print('[Data] Tokenizing train split...')
    train_tokens = []
    for i, example in enumerate(ds['train']):
        text = example['text'].strip()
        if len(text) < 50:  # skip very short lines
            continue
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        train_tokens.extend(tokens)
        if (i + 1) % 50000 == 0:
            print(f'  {i+1} docs → {len(train_tokens):,} tokens')

    # Save as uint16 binary shard
    arr = np.array(train_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(train_dir, 'shard_00000.bin'))
    print(f'[Data] Train: {len(train_tokens):,} tokens saved')

    # Tokenize validation split
    print('[Data] Tokenizing validation split...')
    eval_tokens = []
    for example in ds['validation']:
        text = example['text'].strip()
        if len(text) < 50:
            continue
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        eval_tokens.extend(tokens)

    arr = np.array(eval_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(eval_dir, 'shard_00000.bin'))
    # Also save as eval.bin for TextDataset compatibility
    arr.tofile(os.path.join(eval_dir, 'eval.bin'))
    print(f'[Data] Eval: {len(eval_tokens):,} tokens saved')

    # Also make train.bin symlink for TextDataset compatibility
    train_bin = os.path.join(train_dir, 'train.bin')
    if not os.path.exists(train_bin):
        import shutil
        shutil.copy2(os.path.join(train_dir, 'shard_00000.bin'), train_bin)

    return len(train_tokens)


# ── Hub checkpoint management ─────────────────────────────────────────────────
def pull_pretrain_checkpoint(token):
    """Pull the latest pretraining checkpoint from Hub if exists."""
    if not token:
        return None, 0

    try:
        from huggingface_hub import hf_hub_download
        # Check for pretrain progress file
        local = hf_hub_download(
            HUB_REPO, 'pretrain_progress.json',
            repo_type='model', token=token
        )
        progress = json.load(open(local))
        last_ckpt = progress.get('latest_checkpoint', '')
        last_step = progress.get('step', 0)

        if not last_ckpt or last_step == 0:
            print('[Hub] No pretrain progress found — starting fresh')
            return None, 0

        print(f'[Hub] Found pretrain progress: step {last_step}, checkpoint {last_ckpt}')

        # Download the checkpoint
        ckpt_local = os.path.join(CKPT_DIR, last_ckpt)
        if os.path.exists(os.path.join(ckpt_local, 'model.pt')):
            print(f'[Hub] Checkpoint already local: {ckpt_local}')
            return ckpt_local, last_step

        os.makedirs(ckpt_local, exist_ok=True)
        for fname in ['model.pt', 'optimizer.pt', 'scheduler.pt', 'metadata.json']:
            try:
                hub_path = f'{last_ckpt}/{fname}'
                dl = hf_hub_download(
                    HUB_REPO, hub_path,
                    repo_type='model', token=token
                )
                import shutil
                shutil.copy2(dl, os.path.join(ckpt_local, fname))
                print(f'  Downloaded {fname}')
            except Exception as e:
                if fname == 'model.pt':
                    print(f'  CRITICAL: Failed to download model.pt: {e}')
                    return None, 0
                print(f'  Skipped {fname}: {e}')

        return ckpt_local, last_step

    except Exception as e:
        print(f'[Hub] No pretrain progress: {e}')
        return None, 0


def push_pretrain_checkpoint(ckpt_path, step, token):
    """Push checkpoint and progress to Hub (non-blocking wrapper)."""
    if not token or not ckpt_path:
        return

    import threading

    def _push():
        try:
            from huggingface_hub import upload_folder, upload_file
            import io

            ckpt_name = os.path.basename(ckpt_path)
            print(f'[Hub] Pushing {ckpt_name} (step {step})...', flush=True)

            upload_folder(
                folder_path=ckpt_path,
                repo_id=HUB_REPO,
                path_in_repo=ckpt_name,
                repo_type='model',
                token=token,
                ignore_patterns=['optimizer.pt'],
                commit_message=f'Pretrain checkpoint: {ckpt_name} (step {step})',
            )

            # Update progress pointer
            progress = json.dumps({
                'latest_checkpoint': ckpt_name,
                'step': step,
                'total_steps': TOTAL_STEPS,
            })
            upload_file(
                path_or_fileobj=io.BytesIO(progress.encode()),
                path_in_repo='pretrain_progress.json',
                repo_id=HUB_REPO,
                repo_type='model',
                token=token,
                commit_message=f'Pretrain progress: step {step}/{TOTAL_STEPS}',
            )
            print(f'[Hub] {ckpt_name} pushed OK', flush=True)
        except Exception as e:
            if '429' in str(e):
                print(f'[Hub] Rate limited — skipping push', flush=True)
            else:
                print(f'[Hub] Push failed: {e}', flush=True)

    t = threading.Thread(target=_push, daemon=True)
    t.start()
    return t


# ── Build training config ────────────────────────────────────────────────────
def create_training_config(dtype, resume_step=0):
    """Create a YAML config for train.py."""
    remaining_steps = min(TOTAL_STEPS - resume_step, STEPS_PER_SESSION)

    config = {
        'seed': 42,
        'training': {
            'per_device_batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRAD_ACCUM,
            'learning_rate': LR,
            'weight_decay': 0.1,
            'adam_beta1': 0.9,
            'adam_beta2': 0.95,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'lr_scheduler': 'cosine',
            'warmup_steps': WARMUP if resume_step == 0 else 0,
            'min_lr_ratio': 0.1,
            'max_steps': resume_step + remaining_steps,
            'max_seq_length': SEQ_LEN,
            'dtype': dtype,
        },
        'checkpointing': {
            'save_steps': SAVE_STEPS,
            'save_dir': CKPT_DIR,
            'keep_last_n': 3,
        },
        'logging': {
            'log_steps': LOG_STEPS,
            'wandb_project': None,
            'wandb_run_name': None,
        },
        'eval': {
            'eval_steps': EVAL_STEPS,
            'eval_samples': 50,
        },
        'data': {
            'train_data': os.path.join(DATA_DIR, 'train'),
            'eval_data': os.path.join(DATA_DIR, 'eval'),
            'tokenizer_path': TOKENIZER_PATH,
            'num_workers': 2,
        },
        'distributed': {
            'strategy': 'none',
            'gradient_checkpointing': False,
            'cpu_offload': False,
        },
    }

    config_path = '/kaggle/working/pretrain_config.yaml'
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path, remaining_steps


# ── Find latest local checkpoint ─────────────────────────────────────────────
def find_latest_checkpoint():
    """Find the highest-step checkpoint in CKPT_DIR."""
    import glob
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'checkpoint-*')))
    if ckpts:
        # Verify it has model.pt
        for ckpt in reversed(ckpts):
            if os.path.exists(os.path.join(ckpt, 'model.pt')):
                return ckpt
    return None


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 60)
    print('  YAYA-125M PRETRAINING')
    print('  Target: 20K steps → coherent English generation')
    print('=' * 60)

    hf_token = get_hf_token()
    dtype = get_gpu_info()

    if hf_token:
        print(f'HF_TOKEN: loaded')
    else:
        print('WARNING: No HF_TOKEN — checkpoints will NOT persist across sessions')

    # Step 1: Download and tokenize corpus
    print(f'\n{"─"*60}')
    print('  STEP 1: Prepare training data')
    print(f'{"─"*60}')
    download_and_tokenize()

    # Step 2: Check for existing checkpoint
    print(f'\n{"─"*60}')
    print('  STEP 2: Resume from checkpoint')
    print(f'{"─"*60}')

    resume_ckpt, resume_step = pull_pretrain_checkpoint(hf_token)

    # Also check local checkpoints (in case Hub pull failed but local exists)
    local_ckpt = find_latest_checkpoint()
    if local_ckpt and not resume_ckpt:
        # Extract step from dir name
        try:
            local_step = int(os.path.basename(local_ckpt).split('-')[-1])
            if local_step > resume_step:
                resume_ckpt = local_ckpt
                resume_step = local_step
                print(f'[Local] Found checkpoint at step {local_step}')
        except ValueError:
            pass

    if resume_step >= TOTAL_STEPS:
        print(f'\nPretraining COMPLETE ({resume_step}/{TOTAL_STEPS} steps)')
        print('Ready for curriculum SFT!')
        sys.exit(0)

    print(f'\nResuming from step {resume_step}/{TOTAL_STEPS}')

    # Step 3: Create config and run training
    print(f'\n{"─"*60}')
    print('  STEP 3: Training')
    print(f'{"─"*60}')

    config_path, steps_this_session = create_training_config(dtype, resume_step)
    print(f'Steps this session: {steps_this_session}')
    print(f'Target: step {resume_step} → {resume_step + steps_this_session}')

    # Build command
    cmd = [
        sys.executable, '-u',
        os.path.join(REPO_ROOT, 'scripts', 'train.py'),
        '--model_config', MODEL_CONFIG,
        '--train_config', config_path,
    ]
    if resume_ckpt:
        cmd += ['--resume', resume_ckpt]

    print(f'\nCommand: {" ".join(cmd)}\n')

    # Run training with timeout (3h safety)
    timeout = int(3.0 * 3600)  # 3 hours
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT,
            timeout=timeout,
            env={**os.environ, 'PYTHONUNBUFFERED': '1', 'WANDB_DISABLED': 'true'},
        )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print('\n[Timeout] Training hit 3h limit — saving progress')
        success = True  # still push what we have

    # Step 4: Push final checkpoint to Hub
    print(f'\n{"─"*60}')
    print('  STEP 4: Push checkpoint to Hub')
    print(f'{"─"*60}')

    final_ckpt = find_latest_checkpoint()
    if final_ckpt:
        final_step = int(os.path.basename(final_ckpt).split('-')[-1])
        print(f'Final checkpoint: {os.path.basename(final_ckpt)} (step {final_step})')

        if hf_token:
            # Blocking push for final checkpoint
            try:
                from huggingface_hub import upload_folder, upload_file
                import io

                ckpt_name = os.path.basename(final_ckpt)
                print(f'[Hub] Pushing {ckpt_name}...', flush=True)
                upload_folder(
                    folder_path=final_ckpt,
                    repo_id=HUB_REPO,
                    path_in_repo=ckpt_name,
                    repo_type='model',
                    token=hf_token,
                    ignore_patterns=['optimizer.pt'],
                    commit_message=f'Pretrain checkpoint: {ckpt_name} (step {final_step})',
                )
                # Update progress
                progress = json.dumps({
                    'latest_checkpoint': ckpt_name,
                    'step': final_step,
                    'total_steps': TOTAL_STEPS,
                })
                upload_file(
                    path_or_fileobj=io.BytesIO(progress.encode()),
                    path_in_repo='pretrain_progress.json',
                    repo_id=HUB_REPO,
                    repo_type='model',
                    token=hf_token,
                    commit_message=f'Pretrain progress: step {final_step}/{TOTAL_STEPS}',
                )
                # Also update latest.json
                latest = json.dumps({'latest': ckpt_name})
                upload_file(
                    path_or_fileobj=io.BytesIO(latest.encode()),
                    path_in_repo='latest.json',
                    repo_id=HUB_REPO,
                    repo_type='model',
                    token=hf_token,
                    commit_message=f'Update latest pointer to {ckpt_name}',
                )
                print(f'[Hub] Pushed successfully', flush=True)
            except Exception as e:
                print(f'[Hub] Final push failed: {e}', flush=True)
    else:
        print('No checkpoint found to push')

    # Summary
    elapsed = (time.time() - SESSION_START) / 60
    print(f'\n{"="*60}')
    print(f'  PRETRAIN SESSION SUMMARY')
    print(f'{"="*60}')
    if final_ckpt:
        final_step = int(os.path.basename(final_ckpt).split('-')[-1])
        print(f'  Progress: {final_step}/{TOTAL_STEPS} steps ({final_step/TOTAL_STEPS*100:.0f}%)')
        remaining = TOTAL_STEPS - final_step
        if remaining > 0:
            sessions_left = remaining / STEPS_PER_SESSION
            print(f'  Remaining: {remaining} steps (~{sessions_left:.1f} more sessions)')
        else:
            print(f'  PRETRAINING COMPLETE! Ready for curriculum SFT.')
    print(f'  Session time: {elapsed:.0f} min')
    print()
    sys.exit(0)
