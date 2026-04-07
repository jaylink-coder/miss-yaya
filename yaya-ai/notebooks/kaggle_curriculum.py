"""
Yaya True AI — Curriculum Training (Kaggle Notebook)
=====================================================
Copy this entire file into a single Kaggle notebook cell.

Prerequisites (Kaggle Settings -> Secrets):
  - HF_TOKEN      (REQUIRED — checkpoint persistence)
  - WANDB_API_KEY  (optional — live monitoring)

GPU: T4 (free tier) — select under Accelerator
Internet: ON (required for HF Hub + git clone)

Each session trains ~8 phases (~20 min each).
Total: 16 phases across ~2 sessions = True AI.
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import subprocess, sys, os, shutil

# Clone the repo
REPO_URL = "https://github.com/jaylink-coder/miss-yaya.git"
WORK_DIR = "/kaggle/working/Yaya"
REPO_DIR = os.path.join(WORK_DIR, "yaya-ai")

if not os.path.isdir(REPO_DIR):
    print("Cloning repo...")
    subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True)
else:
    print("Repo exists — pulling latest...")
    subprocess.run(["git", "pull"], cwd=WORK_DIR)

# Clean stale state from previous buggy runs
CKPT_DIR = "/kaggle/working/yaya-curriculum-checkpoints"
prog_path = os.path.join(CKPT_DIR, "curriculum_progress.json")
if os.path.exists(prog_path):
    os.remove(prog_path)
    print("Removed stale local progress")
if os.path.exists(CKPT_DIR):
    for item in os.listdir(CKPT_DIR):
        if "checkpoint" in item:
            shutil.rmtree(os.path.join(CKPT_DIR, item), ignore_errors=True)
            print(f"Removed old checkpoint: {item}")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "sentencepiece",
                "pyyaml", "safetensors"], check=False)

# ── Cell 2: Run curriculum training ───────────────────────────────────────────
print("\n" + "=" * 60)
print(" LAUNCHING YAYA TRUE AI CURRICULUM TRAINING")
print("=" * 60 + "\n")

# Use -u for unbuffered output so logs appear in real time
result = subprocess.run(
    [sys.executable, "-u", os.path.join(REPO_DIR, "scripts", "kaggle_run_curriculum.py")],
    cwd=REPO_DIR,
    env={**os.environ, "PYTHONUNBUFFERED": "1"}
)

if result.returncode == 0:
    print("\nSession complete! Re-run this notebook for the next batch of phases.")
else:
    print(f"\nTraining exited with code {result.returncode}. Check logs above.")
