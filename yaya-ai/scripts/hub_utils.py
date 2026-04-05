"""HuggingFace Hub utilities for checkpoint persistence.

Keeps training progress safe across Kaggle sessions by pushing checkpoints
to HF Hub immediately after each save. On session start, pulls the latest
checkpoint back if no local one exists.

Usage (in kaggle_run_sft.py):
    from scripts.hub_utils import push_checkpoint, pull_latest_checkpoint, start_watcher

    # Session start — restore progress
    pull_latest_checkpoint("Jaylink-coder/yaya-125m", "/kaggle/working/yaya-ckpts", token)

    # During training — background watcher auto-pushes every new checkpoint
    watcher = start_watcher("/kaggle/working/yaya-ckpts", "Jaylink-coder/yaya-125m", token)
"""

import os
import glob
import json
import time
import shutil
import threading
from pathlib import Path


def _get_api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def ensure_repo(repo_id, token):
    """Create the HF Hub repo if it doesn't exist."""
    try:
        api = _get_api(token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        return True
    except Exception as e:
        print(f"[Hub] Could not create repo {repo_id}: {e}")
        return False


def push_checkpoint(ckpt_path, repo_id, token, verbose=True):
    """Upload a single checkpoint folder to HF Hub.

    The checkpoint is stored at path_in_repo = basename(ckpt_path).
    e.g. /kaggle/working/yaya-ckpts/checkpoint-05000 → checkpoint-05000/
    """
    if not token:
        return False
    ckpt_path = str(ckpt_path)
    if not os.path.isdir(ckpt_path):
        return False

    try:
        from huggingface_hub import upload_folder
        ckpt_name = os.path.basename(ckpt_path)
        if verbose:
            print(f"[Hub] Pushing {ckpt_name} → {repo_id}...", flush=True)
        # Only push model weights + metadata — skip optimizer.pt (1GB, not needed for resume)
        upload_folder(
            folder_path=ckpt_path,
            repo_id=repo_id,
            path_in_repo=ckpt_name,
            repo_type="model",
            token=token,
            ignore_patterns=["optimizer.pt"],
            commit_message=f"Training checkpoint: {ckpt_name}",
        )
        # Also update a pointer file so we know the latest checkpoint
        latest_info = json.dumps({"latest": ckpt_name})
        from huggingface_hub import upload_file
        import io
        upload_file(
            path_or_fileobj=io.BytesIO(latest_info.encode()),
            path_in_repo="latest.json",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Update latest pointer to {ckpt_name}",
        )
        if verbose:
            print(f"[Hub] {ckpt_name} pushed successfully.", flush=True)
        return True
    except Exception as e:
        print(f"[Hub] Push failed for {ckpt_path}: {e}", flush=True)
        return False


def pull_latest_checkpoint(repo_id, local_dir, token, verbose=True):
    """Download the latest checkpoint from HF Hub to local_dir.

    Returns the path to the downloaded checkpoint, or None if unavailable.
    Skips download if a local checkpoint already exists.
    """
    if not token:
        return None

    # Check if we already have a local checkpoint
    existing = sorted(glob.glob(os.path.join(local_dir, "checkpoint-*")))
    if existing:
        if verbose:
            print(f"[Hub] Local checkpoint found: {os.path.basename(existing[-1])} — skipping download.")
        return existing[-1]

    try:
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
        import requests

        if verbose:
            print(f"[Hub] No local checkpoint — checking {repo_id} for saved progress...", flush=True)

        # Get the latest pointer
        try:
            latest_path = hf_hub_download(
                repo_id=repo_id,
                filename="latest.json",
                repo_type="model",
                token=token,
                local_dir="/tmp/hub_meta",
            )
            with open(latest_path) as f:
                latest_name = json.load(f)["latest"]
        except Exception:
            # No latest.json — find most recent checkpoint by listing files
            # Scan all checkpoint prefix types: checkpoint-, dpo-checkpoint-, recovery-, dpo2-
            files = list(list_repo_files(repo_id=repo_id, repo_type="model", token=token))
            CKPT_PREFIXES = ("checkpoint-", "dpo-checkpoint-", "recovery-checkpoint-",
                             "dpo2-checkpoint-", "patch-checkpoint-")
            ckpt_names = sorted({
                f.split("/")[0] for f in files
                if any(f.split("/")[0].startswith(p) for p in CKPT_PREFIXES)
                and "_temp" not in f
            })
            if not ckpt_names:
                print(f"[Hub] No checkpoints found in {repo_id}.")
                return None
            # Prefer patch > dpo2 > recovery > dpo > sft (best training stage)
            for prefix in ("patch-checkpoint-", "dpo2-checkpoint-", "recovery-checkpoint-", "dpo-checkpoint-"):
                prefixed = [c for c in ckpt_names if c.startswith(prefix)]
                if prefixed:
                    latest_name = prefixed[-1]
                    break
            else:
                latest_name = ckpt_names[-1]

        if verbose:
            print(f"[Hub] Downloading {latest_name} from {repo_id}...", flush=True)

        os.makedirs(local_dir, exist_ok=True)
        local_ckpt = os.path.join(local_dir, latest_name)
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{latest_name}/*",
            local_dir=local_dir,
            repo_type="model",
            token=token,
        )
        if os.path.isdir(local_ckpt):
            if verbose:
                print(f"[Hub] Restored: {local_ckpt}", flush=True)
            return local_ckpt
        return None
    except Exception as e:
        print(f"[Hub] Pull failed: {e}", flush=True)
        return None


def start_watcher(ckpt_dir, repo_id, token, interval_sec=90):
    """Start a background thread that pushes new checkpoints to HF Hub.

    Checks every `interval_sec` seconds for new checkpoint directories
    and pushes any that haven't been pushed yet.

    Returns the thread (daemon — dies when training ends).
    """
    if not token:
        print("[Hub] No token — checkpoint watcher disabled.")
        return None

    _GLOB_PATTERNS = ["checkpoint-*", "patch-checkpoint-*", "dpo2-checkpoint-*",
                      "recovery-checkpoint-*", "dpo-checkpoint-*"]

    pushed = set()
    # Pre-populate with any checkpoints that already exist (don't re-push)
    for pat in _GLOB_PATTERNS:
        for ckpt in glob.glob(os.path.join(ckpt_dir, pat)):
            pushed.add(ckpt)

    def watch():
        while True:
            time.sleep(interval_sec)
            try:
                all_ckpts = []
                for pat in _GLOB_PATTERNS:
                    all_ckpts.extend(glob.glob(os.path.join(ckpt_dir, pat)))
                for ckpt in sorted(all_ckpts):
                    if ckpt.endswith('_temp'):
                        continue
                    if ckpt not in pushed:
                        # Wait a moment to ensure write is complete
                        time.sleep(5)
                        if push_checkpoint(ckpt, repo_id, token):
                            pushed.add(ckpt)
            except Exception as e:
                print(f"[Hub] Watcher error: {e}", flush=True)

    t = threading.Thread(target=watch, daemon=True, name="hub-watcher")
    t.start()
    print(f"[Hub] Checkpoint watcher started → {repo_id} (every {interval_sec}s)", flush=True)
    return t
