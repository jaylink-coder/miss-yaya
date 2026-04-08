"""Consolidation training runner — fixes catastrophic forgetting.

After the 16-phase sequential curriculum, the model scored 39% model-only
because later phases overwrote earlier knowledge (Kenya 0%, Swahili 0%).

This script runs one unified training pass over all phase data mixed together
with smart weighting (failing categories get 2-3x more examples).

Works on both Kaggle and Google Colab (auto-detects environment).

Usage (Colab cell):
    !python scripts/run_consolidation.py

Usage (Kaggle cell):
    !python /kaggle/working/miss-yaya/yaya-ai/scripts/run_consolidation.py

Flags:
    --steps N      Override training steps (default: 3000)
    --lr F         Override learning rate (default: 5e-6)
    --no-push      Skip HF Hub push
    --no-drive     Skip Google Drive backup (Colab only)
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
    # Local (Windows / Linux dev)
    ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT_BASE = os.path.join(ROOT, "checkpoints")
    DRIVE_DIR = None

sys.path.insert(0, ROOT)
HUB_REPO  = "Jaylink-coder/yaya-125m"
HUB_TAG   = "consolidation"

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


def run_subprocess(cmd, cwd, timeout_sec=6*3600):
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


# ── HF Hub utilities ───────────────────────────────────────────────────────────
def get_hf_token(cli_token=""):
    if cli_token:
        return cli_token
    # Colab secrets
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN") or ""
        if tok: return tok
    except Exception:
        pass
    # Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok: return tok
    except Exception:
        pass
    # Environment / .env file
    tok = os.environ.get("HF_TOKEN", "")
    if tok: return tok
    env_file = os.path.join(ROOT, ".env")
    if os.path.exists(env_file):
        for line in open(env_file):
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip()
                if tok: return tok
    return ""


def pull_latest_checkpoint(token, local_dir):
    """Pull best checkpoint from HF Hub."""
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

    priority = ["consolidation", "curriculum-phase16", "curriculum-phase",
                "patch-checkpoint", "dpo2-checkpoint", "dpo-checkpoint", "checkpoint"]

    best = None
    for prefix in priority:
        matches = [d for d in ckpt_dirs if d.startswith(prefix)]
        if matches:
            def step_num(n):
                for p in reversed(n.split("-")):
                    if p.isdigit(): return int(p)
                return 0
            best = sorted(matches, key=step_num, reverse=True)[0]
            break
    if not best:
        best = sorted(ckpt_dirs)[-1]

    print(f"  Using: {best}")
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
    return model_pt if os.path.exists(model_pt) else None


def push_to_hub(token, ckpt_dir, step):
    from huggingface_hub import HfApi
    tag = f"{HUB_TAG}-step{step:05d}"
    print(f"  Pushing {tag} to HF Hub...")
    api = HfApi(token=token)
    for fname in os.listdir(ckpt_dir):
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.isfile(fpath):
            try:
                api.upload_file(path_or_fileobj=fpath,
                                path_in_repo=f"{tag}/{fname}",
                                repo_id=HUB_REPO, token=token)
            except Exception as e:
                print(f"    WARNING: {fname}: {e}")
    print(f"  Pushed: {tag}")


def backup_to_drive(ckpt_dir, step):
    if not DRIVE_DIR or not os.path.exists(os.path.dirname(DRIVE_DIR)):
        return
    os.makedirs(DRIVE_DIR, exist_ok=True)
    tag  = f"{HUB_TAG}-step{step:05d}"
    dest = os.path.join(DRIVE_DIR, tag)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(ckpt_dir, dest)
    print(f"  Backed up to Drive: {dest}")


# ── Data preparation ───────────────────────────────────────────────────────────
def prepare_data():
    """Generate consolidation data if not already present."""
    out = os.path.join(ROOT, "data/sft/yaya_consolidation.jsonl")
    if os.path.exists(out):
        n = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"  Consolidation data already exists: {n:,} examples")
        return out
    print("  Generating consolidation data...")
    gen_script = os.path.join(ROOT, "scripts/generate_consolidation_data.py")
    subprocess.run([sys.executable, gen_script], cwd=ROOT, check=True)
    return out


# ── Find output checkpoint ─────────────────────────────────────────────────────
def find_output_ckpt(output_dir, fallback):
    model_files = glob.glob(f"{output_dir}/**/model.pt", recursive=True)
    if model_files:
        return sorted(model_files, key=os.path.getmtime, reverse=True)[0]
    for d in sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=os.path.getmtime, reverse=True):
        mp = os.path.join(d, "model.pt")
        if os.path.exists(mp): return mp
    return fallback


# ── Training ───────────────────────────────────────────────────────────────────
def run_training(start_ckpt, data_file, steps, lr, output_dir, batch, grad_accum, precision_flag):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts/train_sft.py"),
        "--model_config",   os.path.join(ROOT, "configs/model/yaya_125m.yaml"),
        "--train_config",   os.path.join(ROOT, "configs/training/sft_125m.yaml"),
        "--pretrain_checkpoint", start_ckpt,
        "--data_file",      data_file,
        "--output_dir",     output_dir,
        "--max_steps",      str(steps),
        "--learning_rate",  str(lr),
        "--per_device_batch_size",       str(batch),
        "--gradient_accumulation_steps", str(grad_accum),
        "--max_seq_length", "512",
        "--save_steps",     str(steps // 3),   # save at 33%, 66%, 100%
        "--warmup_steps",   "200",
        "--lr_scheduler",   "cosine",
        "--weight_decay",   "0.01",
        "--max_grad_norm",  "1.0",
        "--dataloader_num_workers", "2",
    ]
    if precision_flag:
        cmd.append(precision_flag)

    eff = batch * grad_accum
    print(f"\n  Consolidation training")
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch}×{grad_accum}={eff}  {precision_flag or 'fp32'}")
    print(f"  Data:  {data_file}")
    print(f"  Output: {output_dir}")

    for attempt, (bs, ga) in enumerate([(batch, grad_accum),
                                         (max(1, batch//2), grad_accum*2),
                                         (1, grad_accum * batch)]):
        if attempt > 0:
            print(f"\n  OOM retry {attempt}: batch={bs} accum={ga}")
            cmd_r = [c if c != str(batch) else str(bs) for c in cmd]
            cmd_r = [c if c != str(grad_accum) else str(ga) for c in cmd_r]
            clear_memory()
        else:
            cmd_r = cmd

        start = time.time()
        proc  = run_subprocess(cmd_r, ROOT, timeout_sec=8*3600)
        elapsed = time.time() - start
        rc = proc.returncode if proc else -1
        print(f"\n  Finished in {elapsed/60:.1f} min (exit {rc})")
        clear_memory()

        out = find_output_ckpt(output_dir, None)
        if out: return out
        if rc == 0: break   # clean exit, no file — unusual

    return start_ckpt   # fallback


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
        if any(k in line for k in ["Yaya Benchmark", "OVERALL", "====", "Guard lift", "DUAL", "Guard"]):
            in_table = True
        if in_table:
            print("   ", line)
        if "Results saved" in line:
            break
    if result.returncode != 0 and result.stderr:
        print("  BENCHMARK ERROR:", result.stderr[-400:])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Consolidation training — fix catastrophic forgetting")
    parser.add_argument("--steps",    type=int,   default=3000)
    parser.add_argument("--lr",       type=float, default=5e-6)
    parser.add_argument("--token",    type=str,   default="")
    parser.add_argument("--no-push",  action="store_true")
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--eval-only",action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Yaya — Consolidation Training")
    print(f"  Platform: {'Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'local'}")
    print("=" * 60)

    hf_token = get_hf_token(args.token)
    if not hf_token:
        print("  WARNING: No HF_TOKEN — Hub push/pull disabled.")

    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    # ── Pull latest checkpoint ─────────────────────────────────────────────────
    ckpt_local_dir = os.path.join(CKPT_BASE, "yaya-125m-consolidation")
    start_ckpt = None

    if hf_token:
        start_ckpt = pull_latest_checkpoint(hf_token, ckpt_local_dir)

    if not start_ckpt:
        # Try Drive (Colab)
        if DRIVE_DIR and os.path.exists(DRIVE_DIR):
            for prefix in ["consolidation", "curriculum-phase16", "curriculum-phase"]:
                matches = sorted([d for d in os.listdir(DRIVE_DIR) if d.startswith(prefix)], reverse=True)
                if matches:
                    src = os.path.join(DRIVE_DIR, matches[0], "model.pt")
                    if os.path.exists(src):
                        dest = os.path.join(ckpt_local_dir, matches[0])
                        os.makedirs(dest, exist_ok=True)
                        dst = os.path.join(dest, "model.pt")
                        if not os.path.exists(dst):
                            shutil.copy(src, dst)
                        start_ckpt = dst
                        print(f"  Restored from Drive: {matches[0]}")
                        break

    if not start_ckpt:
        local_pts = glob.glob(f"{CKPT_BASE}/**/*.pt", recursive=True)
        if local_pts:
            start_ckpt = sorted(local_pts, key=os.path.getmtime, reverse=True)[0]
            print(f"  Using local checkpoint: {start_ckpt}")

    if not start_ckpt:
        print("ERROR: No checkpoint found. Set HF_TOKEN or mount Drive with a checkpoint.")
        sys.exit(1)

    print(f"\n  Starting from: {start_ckpt}")

    if args.eval_only:
        run_benchmark(start_ckpt)
        return

    # ── Prepare data ───────────────────────────────────────────────────────────
    data_file = prepare_data()

    # ── Detect GPU ─────────────────────────────────────────────────────────────
    gpu_name, vram, batch, grad_accum, precision_flag = detect_gpu()
    print(f"  GPU: {gpu_name} ({vram:.1f} GB) — batch={batch} accum={grad_accum} {precision_flag or 'fp32'}")

    # ── Train ──────────────────────────────────────────────────────────────────
    output_dir = os.path.join(CKPT_BASE, "yaya-125m-consolidation", "run")
    result_ckpt = run_training(
        start_ckpt, data_file, args.steps, args.lr,
        output_dir, batch, grad_accum, precision_flag
    )

    # ── Push / backup ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(result_ckpt) if result_ckpt.endswith(".pt") else result_ckpt
    if hf_token and not args.no_push:
        push_to_hub(hf_token, out_dir, args.steps)
    if not args.no_drive and not args.eval_only:
        backup_to_drive(out_dir, args.steps)

    # ── Benchmark ──────────────────────────────────────────────────────────────
    run_benchmark(result_ckpt)

    print("\n" + "=" * 60)
    print("  Consolidation training complete.")
    print(f"  Checkpoint: {result_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
