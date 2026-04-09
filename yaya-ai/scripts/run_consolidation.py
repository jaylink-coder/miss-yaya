"""Consolidation training runner — fixes catastrophic forgetting.

After the 16-phase sequential curriculum, the model scored 39% model-only
because later phases overwrote earlier knowledge (Kenya 0%, Swahili 0%).

This script runs one unified training pass over all phase data mixed together
with smart weighting (failing categories get 2-3x more examples).

Works on both Kaggle and Google Colab (auto-detects environment).

Usage:
    !python scripts/run_consolidation.py           # auto (resume if partial run exists)
    !python scripts/run_consolidation.py --fresh   # clear bad partial run, start clean
    !python scripts/run_consolidation.py --eval-only

Flags:
    --steps N      Training steps (default: 3000)
    --lr F         Learning rate (default: 5e-6)
    --fresh        Delete any partial run dir and start clean (fixes broken optimizer state)
    --no-push      Skip HF Hub push
    --no-drive     Skip Google Drive backup
    --token HF_    Override HF token
"""

import argparse, gc, glob, json, os, shutil, subprocess, sys, threading, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Detect environment ─────────────────────────────────────────────────────────
IN_COLAB  = os.path.exists("/content")
IN_KAGGLE = os.path.exists("/kaggle")

if IN_COLAB:
    ROOT      = "/content/miss-yaya/yaya-ai"
    CKPT_BASE = "/content/checkpoints"
    DRIVE_DIR = "/content/drive/MyDrive/yaya-checkpoints"
elif IN_KAGGLE:
    ROOT      = "/kaggle/working/miss-yaya/yaya-ai"
    CKPT_BASE = "/kaggle/working/checkpoints"
    DRIVE_DIR = None
else:
    ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT_BASE = os.path.join(ROOT, "checkpoints")
    DRIVE_DIR = None

sys.path.insert(0, ROOT)
HUB_REPO = "Jaylink-coder/yaya-125m"
HUB_TAG  = "consolidation"

# A consolidation checkpoint with loss > this is considered broken/untrained
_MAX_VALID_LOSS = 4.0


# ── GPU detection ──────────────────────────────────────────────────────────────
def detect_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", 0, 2, 16, ""
        name = torch.cuda.get_device_name(0).upper()
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram >= 35: return name, vram, 16, 2, "--bf16"
        if vram >= 20: return name, vram,  8, 4, "--fp16"
        if vram >= 14: return name, vram,  4, 8, "--fp16"
        return name, vram, 2, 16, ""
    except Exception:
        return "unknown", 0, 4, 8, ""


def clear_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── Heartbeat + hang-safe subprocess ──────────────────────────────────────────
class _Heartbeat(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()
    def run(self):
        while not self._stop.wait(60):
            print(".", end="", flush=True)
    def stop(self):
        self._stop.set()


def run_subprocess(cmd, cwd, timeout_sec=8*3600):
    hb = _Heartbeat()
    hb.start()
    proc = None
    try:
        proc = subprocess.Popen(cmd, cwd=cwd)
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"\n  TIMEOUT after {timeout_sec//60} min — killing")
            proc.kill()
            proc.wait(timeout=30)
    finally:
        hb.stop()
    return proc


# ── Checkpoint quality check ───────────────────────────────────────────────────
def checkpoint_loss(ckpt_path):
    """Read loss from metadata.json next to model.pt. Returns None if not found."""
    meta_path = os.path.join(os.path.dirname(ckpt_path), "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f).get("loss")
    except Exception:
        return None


def is_good_checkpoint(ckpt_path):
    """Return True if checkpoint has loss < _MAX_VALID_LOSS or no metadata (unknown = assume ok)."""
    if not os.path.isfile(ckpt_path):
        return False
    loss = checkpoint_loss(ckpt_path)
    if loss is None:
        return True        # no metadata — give benefit of the doubt
    return loss < _MAX_VALID_LOSS


# ── HF Hub utilities ───────────────────────────────────────────────────────────
def get_hf_token(cli_token=""):
    if cli_token: return cli_token
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN") or ""
        if tok: return tok
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok: return tok
    except Exception:
        pass
    tok = os.environ.get("HF_TOKEN", "")
    if tok: return tok
    env_file = os.path.join(ROOT, ".env")
    if os.path.exists(env_file):
        for line in open(env_file):
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip()
                if tok: return tok
    return ""


def pull_best_checkpoint(token, local_dir):
    """
    Pull best VALID checkpoint from HF Hub.
    Skips consolidation checkpoints with high loss (broken runs).
    Priority: curriculum-phase16 > curriculum-phase > patch > dpo > consolidation(good)
    """
    from huggingface_hub import list_repo_files, hf_hub_download
    os.makedirs(local_dir, exist_ok=True)
    print("Scanning HF Hub for checkpoints...")
    try:
        files = list(list_repo_files(HUB_REPO, token=token))
    except Exception as e:
        print(f"  Hub scan failed: {e}")
        return None

    ckpt_dirs = set(f.split("/")[0] for f in files if "model.pt" in f)
    if not ckpt_dirs:
        print("  No checkpoints on Hub.")
        return None

    # Priority order — consolidation is LAST now (because it may be broken)
    priority = [
        "curriculum-phase16",
        "curriculum-phase",
        "patch-checkpoint",
        "dpo2-checkpoint",
        "dpo-checkpoint",
        "checkpoint",
        "consolidation",     # last — only use if nothing better exists
    ]

    def step_num(n):
        for p in reversed(n.split("-")):
            if p.isdigit(): return int(p)
        return 0

    # Find best candidate in each priority tier
    candidates = []
    for prefix in priority:
        matches = sorted([d for d in ckpt_dirs if d.startswith(prefix)], key=step_num, reverse=True)
        if matches:
            candidates.append((prefix, matches[0]))

    for prefix, best in candidates:
        dest = os.path.join(local_dir, best)
        os.makedirs(dest, exist_ok=True)

        for fname in ["model.pt", "metadata.json"]:
            hub_path = f"{best}/{fname}"
            local_path = os.path.join(dest, fname)
            if hub_path in files and not os.path.exists(local_path):
                print(f"  Downloading {hub_path}...")
                hf_hub_download(repo_id=HUB_REPO, filename=hub_path,
                                local_dir=dest, local_dir_use_symlinks=False, token=token)

        model_pt = os.path.join(dest, "model.pt")
        if not os.path.exists(model_pt):
            continue

        loss = checkpoint_loss(model_pt)
        loss_str = f"loss={loss:.4f}" if loss is not None else "loss=unknown"

        if is_good_checkpoint(model_pt):
            print(f"  Using: {best}  ({loss_str})")
            return model_pt
        else:
            print(f"  Skipping {best} — {loss_str} > {_MAX_VALID_LOSS} (bad checkpoint)")

    print("  No valid checkpoint found on Hub.")
    return None


def push_to_hub(token, ckpt_dir, actual_step):
    from huggingface_hub import HfApi
    tag = f"{HUB_TAG}-step{actual_step:05d}"
    print(f"  Pushing {tag} to HF Hub...")
    api = HfApi(token=token)
    pushed = 0
    for fname in os.listdir(ckpt_dir):
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.isfile(fpath):
            try:
                api.upload_file(path_or_fileobj=fpath,
                                path_in_repo=f"{tag}/{fname}",
                                repo_id=HUB_REPO, token=token)
                pushed += 1
            except Exception as e:
                print(f"    WARNING: {fname}: {e}")
    print(f"  Pushed {pushed} files as: {tag}")
    return tag


def backup_to_drive(ckpt_dir, actual_step):
    if not DRIVE_DIR or not os.path.exists(os.path.dirname(DRIVE_DIR)):
        return
    os.makedirs(DRIVE_DIR, exist_ok=True)
    tag  = f"{HUB_TAG}-step{actual_step:05d}"
    dest = os.path.join(DRIVE_DIR, tag)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(ckpt_dir, dest)
    print(f"  Backed up to Drive: {dest}")


# ── Data preparation ───────────────────────────────────────────────────────────
def prepare_data():
    out = os.path.join(ROOT, "data/sft/yaya_consolidation.jsonl")
    if os.path.exists(out):
        n = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"  Consolidation data: {n:,} examples")
        return out
    print("  Generating consolidation data...")
    gen_script = os.path.join(ROOT, "scripts/generate_consolidation_data.py")
    subprocess.run([sys.executable, gen_script], cwd=ROOT, check=True)
    return out


# ── Checkpoint utilities ───────────────────────────────────────────────────────
def find_output_ckpt(output_dir):
    """Return path to latest model.pt in output_dir, or None."""
    ckpt_dirs = sorted(
        [d for d in glob.glob(f"{output_dir}/checkpoint-*") if os.path.isdir(d)],
        key=os.path.getmtime, reverse=True
    )
    for d in ckpt_dirs:
        mp = os.path.join(d, "model.pt")
        if os.path.exists(mp):
            return mp
    return None


def find_resume_checkpoint(output_dir):
    """
    Return the latest valid checkpoint dir for resuming, or None.
    A checkpoint is valid for resume if its optimizer.pt looks healthy
    (we check for the file's existence and that it's not tiny/corrupt).
    """
    ckpt_dirs = sorted(
        [d for d in glob.glob(f"{output_dir}/checkpoint-*") if os.path.isdir(d)],
        key=os.path.getmtime, reverse=True
    )
    for d in ckpt_dirs:
        model_pt = os.path.join(d, "model.pt")
        opt_pt   = os.path.join(d, "optimizer.pt")
        if not os.path.exists(model_pt):
            continue
        # Optimizer must exist and be at least 100 MB (8-bit AdamW for 128M model ~1 GB)
        # A tiny optimizer.pt means it was saved before bnb init — skip it
        if os.path.exists(opt_pt) and os.path.getsize(opt_pt) < 100 * 1024 * 1024:
            print(f"  Skipping {os.path.basename(d)} — optimizer.pt too small ({os.path.getsize(opt_pt)//1024//1024} MB), likely corrupt")
            continue
        return d
    return None


def read_step_from_meta(ckpt_dir):
    """Read actual step number from checkpoint metadata.json."""
    meta = os.path.join(ckpt_dir, "metadata.json")
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                return json.load(f).get("step", 0)
        except Exception:
            pass
    # Parse from dir name: checkpoint-00002500 → 2500
    name = os.path.basename(ckpt_dir)
    for part in reversed(name.split("-")):
        if part.isdigit():
            return int(part)
    return 0


# ── Training ───────────────────────────────────────────────────────────────────
def run_training(start_ckpt, data_file, steps, lr, output_dir,
                 batch, grad_accum, precision_flag, fresh):

    if fresh and os.path.exists(output_dir):
        print(f"  --fresh: clearing {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Try to resume from a valid partial checkpoint
    resume_ckpt = None if fresh else find_resume_checkpoint(output_dir)
    use_resume  = resume_ckpt is not None

    if use_resume:
        print(f"  Resuming from: {resume_ckpt}")
    else:
        print(f"  Starting fresh from: {start_ckpt}")

    base_cmd = [
        sys.executable, os.path.join(ROOT, "scripts/train_sft.py"),
        "--model_config",   os.path.join(ROOT, "configs/model/yaya_125m.yaml"),
        "--train_config",   os.path.join(ROOT, "configs/training/sft_125m.yaml"),
    ]
    if use_resume:
        base_cmd += ["--resume", resume_ckpt]
    else:
        base_cmd += ["--pretrain_checkpoint", start_ckpt]

    tail_args = [
        "--data_file",      data_file,
        "--output_dir",     output_dir,
        "--max_steps",      str(steps),
        "--learning_rate",  str(lr),
        "--max_seq_length", "512",
        "--save_steps",     "500",
        "--warmup_steps",   "200",
        "--lr_scheduler",   "cosine",
        "--weight_decay",   "0.01",
        "--max_grad_norm",  "1.0",
        "--dataloader_num_workers", "2",
    ]
    if precision_flag:
        tail_args.append(precision_flag)

    eff = batch * grad_accum
    print(f"\n  Consolidation training {'(RESUME)' if use_resume else '(FRESH)'}")
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch}×{grad_accum}={eff}  {precision_flag or 'fp32'}")

    for attempt, (bs, ga) in enumerate([(batch, grad_accum),
                                         (max(1, batch//2), grad_accum*2),
                                         (1, grad_accum * batch)]):
        if attempt > 0:
            print(f"\n  OOM retry {attempt}: batch={bs} accum={ga}")
            clear_memory()
            # On OOM retry, switch to fresh (resume likely broken too)
            if "--pretrain_checkpoint" not in base_cmd:
                base_cmd = [c for c in base_cmd if c != resume_ckpt and c != "--resume"]
                base_cmd += ["--pretrain_checkpoint", start_ckpt]

        batch_args = [
            "--per_device_batch_size",       str(bs),
            "--gradient_accumulation_steps", str(ga),
        ]
        cmd = base_cmd + batch_args + tail_args

        start = time.time()
        proc  = run_subprocess(cmd, ROOT, timeout_sec=8*3600)
        elapsed = time.time() - start
        rc = proc.returncode if proc else -1
        print(f"\n  Finished in {elapsed/60:.1f} min (exit {rc})")
        clear_memory()

        out = find_output_ckpt(output_dir)
        if out:
            return out
        if rc == 0:
            break

    return None   # training failed to produce output


# ── Benchmark ──────────────────────────────────────────────────────────────────
def run_benchmark(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path) if ckpt_path.endswith(".pt") else ckpt_path
    print("\n  Running benchmark (guarded + model-only)...")
    cmd = [sys.executable, os.path.join(ROOT, "scripts/benchmark.py"),
           "--checkpoint", ckpt_dir, "--dual"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    in_table = False
    for line in result.stdout.split("\n"):
        if any(k in line for k in ["Yaya Benchmark", "OVERALL", "====", "Guard lift", "DUAL"]):
            in_table = True
        if in_table:
            print("   ", line)
        if "Results saved" in line:
            break
    if result.returncode != 0 and result.stderr:
        print("  BENCHMARK ERROR:", result.stderr[-400:])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",     type=int,   default=3000)
    parser.add_argument("--lr",        type=float, default=5e-6)
    parser.add_argument("--token",     type=str,   default="")
    parser.add_argument("--fresh",     action="store_true",
                        help="Delete partial run dir and start clean (fixes broken optimizer state)")
    parser.add_argument("--no-push",   action="store_true")
    parser.add_argument("--no-drive",  action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Yaya — Consolidation Training")
    print(f"  Platform: {'Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'local'}")
    if args.fresh:
        print("  Mode: FRESH (clearing partial run)")
    print("=" * 60)

    hf_token = get_hf_token(args.token)
    if not hf_token:
        print("  WARNING: No HF_TOKEN — Hub push/pull disabled.")
    else:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    # ── Find starting checkpoint ───────────────────────────────────────────────
    ckpt_local_dir = os.path.join(CKPT_BASE, "yaya-125m-consolidation")
    start_ckpt = None

    if hf_token:
        start_ckpt = pull_best_checkpoint(hf_token, ckpt_local_dir)

    # Fallback: Drive
    if not start_ckpt and DRIVE_DIR and os.path.exists(DRIVE_DIR):
        for prefix in ["curriculum-phase16", "curriculum-phase", "patch-checkpoint"]:
            matches = sorted([d for d in os.listdir(DRIVE_DIR) if d.startswith(prefix)], reverse=True)
            for m in matches:
                src = os.path.join(DRIVE_DIR, m, "model.pt")
                if os.path.exists(src) and is_good_checkpoint(src):
                    dest = os.path.join(ckpt_local_dir, m)
                    os.makedirs(dest, exist_ok=True)
                    dst = os.path.join(dest, "model.pt")
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
                    start_ckpt = dst
                    print(f"  Restored from Drive: {m}")
                    break
            if start_ckpt:
                break

    # Fallback: local
    if not start_ckpt:
        local_pts = glob.glob(f"{CKPT_BASE}/**/*.pt", recursive=True)
        good = [p for p in local_pts if is_good_checkpoint(p)]
        if good:
            start_ckpt = sorted(good, key=os.path.getmtime, reverse=True)[0]
            print(f"  Using local checkpoint: {start_ckpt}")

    if not start_ckpt:
        print("ERROR: No valid checkpoint found. Set HF_TOKEN or mount Drive.")
        sys.exit(1)

    print(f"\n  Starting checkpoint: {start_ckpt}")
    loss = checkpoint_loss(start_ckpt)
    if loss:
        print(f"  Checkpoint loss: {loss:.4f}")

    if args.eval_only:
        run_benchmark(start_ckpt)
        return

    # ── Data + GPU ─────────────────────────────────────────────────────────────
    data_file = prepare_data()
    gpu_name, vram, batch, grad_accum, precision_flag = detect_gpu()
    print(f"  GPU: {gpu_name} ({vram:.1f} GB) — batch={batch} accum={grad_accum} {precision_flag or 'fp32'}")

    # ── Train ──────────────────────────────────────────────────────────────────
    output_dir = os.path.join(CKPT_BASE, "yaya-125m-consolidation", "run")
    result_ckpt = run_training(
        start_ckpt, data_file, args.steps, args.lr,
        output_dir, batch, grad_accum, precision_flag, args.fresh
    )

    if result_ckpt is None:
        print("\nERROR: Training failed to produce a checkpoint.")
        print("Try: !python scripts/run_consolidation.py --fresh")
        sys.exit(1)

    # ── Push / backup ──────────────────────────────────────────────────────────
    # Use ACTUAL step count from checkpoint metadata, not args.steps
    out_dir     = os.path.dirname(result_ckpt)
    actual_step = read_step_from_meta(out_dir)
    print(f"\n  Output checkpoint: {result_ckpt}  (step {actual_step})")

    if hf_token and not args.no_push:
        push_to_hub(hf_token, out_dir, actual_step)
    if not args.no_drive:
        backup_to_drive(out_dir, actual_step)

    # ── Benchmark ──────────────────────────────────────────────────────────────
    run_benchmark(result_ckpt)

    print("\n" + "=" * 60)
    print(f"  Consolidation complete — step {actual_step}")
    print(f"  Checkpoint: {result_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
