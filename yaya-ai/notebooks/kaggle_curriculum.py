"""
Yaya True AI — Curriculum Training (Kaggle Notebook)
=====================================================
Copy this entire file into a single Kaggle notebook cell.

Prerequisites (Kaggle Settings -> Secrets):
  - HF_TOKEN      (REQUIRED — checkpoint persistence)
  - WANDB_API_KEY  (optional — live monitoring)

GPU: T4 (free tier) — select under Accelerator
Internet: ON (required for HF Hub + git clone)

Each session trains ~4-5 phases (~40 min each).
Total: 16 phases across ~4 sessions = True AI.
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import subprocess, sys, os

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

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "wandb", "huggingface_hub", "datasets", "sentencepiece",
                "pyyaml", "torch", "safetensors"], check=False)

# ── Cell 2: Run curriculum training ───────────────────────────────────────────
print("\n" + "=" * 60)
print(" LAUNCHING YAYA TRUE AI CURRICULUM TRAINING")
print("=" * 60 + "\n")

result = subprocess.run(
    [sys.executable, os.path.join(REPO_DIR, "scripts", "kaggle_run_curriculum.py")],
    cwd=REPO_DIR
)

if result.returncode == 0:
    print("\nSession complete! Re-run this notebook for the next batch of phases.")
else:
    print(f"\nTraining exited with code {result.returncode}. Check logs above.")
