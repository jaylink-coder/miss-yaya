"""Google Colab runner for the Yaya True AI 16-phase curriculum.

This script runs on Google Colab (T4 / L4 / A100 GPU). It:
1. Installs dependencies and clones the repo (first run)
2. Mounts Google Drive for checkpoint persistence across sessions
3. Pulls the latest checkpoint from HuggingFace Hub
4. Auto-detects which phase to run next (or uses --phase N)
5. Trains for ≤2000 steps on curriculum data for that phase
6. Pushes checkpoint to HF Hub + backs up to Drive
7. Runs the benchmark (guarded + model-only)
8. Marks the phase as complete and continues

Usage (in a Colab cell):
    # First cell — setup (run once per session):
    !pip install -q huggingface_hub sentencepiece torch
    from google.colab import drive
    drive.mount('/content/drive')

    # Second cell — run training:
    !python /content/miss-yaya/yaya-ai/scripts/colab_run_phases.py
    !python /content/miss-yaya/yaya-ai/scripts/colab_run_phases.py --phase 3
    !python /content/miss-yaya/yaya-ai/scripts/colab_run_phases.py --phase 1-4
    !python /content/miss-yaya/yaya-ai/scripts/colab_run_phases.py --eval-only

Tip — Run phases 1-4 first (replace guards with real knowledge):
    !python /content/miss-yaya/yaya-ai/scripts/colab_run_phases.py --phase 1-4

To resume after session expiry, just run the same command — it reads
the phase_done.json from Drive and picks up where it left off.
"""

import argparse, gc, json, os, signal, sys, time, shutil, glob, subprocess, threading
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Detect environment ────────────────────────────────────────────────────────
IN_COLAB = os.path.exists("/content")
DRIVE_MOUNT = "/content/drive/MyDrive"
DRIVE_CKPT  = os.path.join(DRIVE_MOUNT, "yaya-checkpoints")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = "/content/miss-yaya/yaya-ai"
CKPT_BASE = "/content/checkpoints"         # fast local NVMe
CURR_DIR  = os.path.join(ROOT, "data/sft/curriculum")
HUB_REPO  = "Jaylink-coder/yaya-125m"
PHASE_DONE_LOCAL = os.path.join(CKPT_BASE, "yaya-125m-curriculum", "phase_done.json")
PHASE_DONE_DRIVE = os.path.join(DRIVE_CKPT, "phase_done.json")

sys.path.insert(0, ROOT)

# ── Hub checkpoint priority (highest → lowest) ────────────────────────────────
_HUB_PREFIXES = [
    "curriculum-phase",
    "p8-checkpoint",
    "patch-checkpoint",
    "dpo2-checkpoint",
    "recovery-checkpoint",
    "dpo-checkpoint",
    "checkpoint",
]

# ── Phase definitions (mirrors milestones_v2.yaml) ────────────────────────────
PHASES = [
    # id, name, data_file, steps, lr, description
    (1,  "World Knowledge",        "phase01_world_knowledge.jsonl",    2000, 1.5e-5, "Factual recall — replace guards with learned knowledge"),
    (2,  "Conversational Fluency", "phase02_conversational.jsonl",     2000, 1.2e-5, "Identity, greetings, personality, empathy"),
    (3,  "Instruction Following",  "phase03_instruction_follow.jsonl", 2000, 1.2e-5, "Format constraints, one-word answers, yes/no, lists"),
    (4,  "Direct Q&A + Kenya",     "phase04_direct_qa.jsonl",          2000, 1.0e-5, "Kenya knowledge, Swahili vocab, concise answers"),
    (5,  "Chain of Thought",       "phase05_chain_of_thought.jsonl",   2000, 9.0e-6, "Think-before-answer using <|think|> tags"),
    (6,  "Math Reasoning",         "phase06_math_reasoning.jsonl",     2000, 8.0e-6, "Arithmetic and word problems with working shown"),
    (7,  "Logical Reasoning",      "phase07_logical_reasoning.jsonl",  2000, 7.0e-6, "Deduction, analogies, puzzles"),
    (8,  "Self-Reflection",        "phase08_self_reflection.jsonl",    1500, 6.0e-6, "Check own answers, admit uncertainty"),
    (9,  "Tool Calling",           "phase09_tool_basics.jsonl",        1500, 6.0e-6, "Calculator, datetime tool calls"),
    (10, "Multi-Step Tools",       "phase10_multi_tool.jsonl",         1500, 5.0e-6, "ReAct: think→act→observe→answer"),
    (11, "RAG Grounding",          "phase11_rag_grounding.jsonl",      1500, 5.0e-6, "Answer from provided context, cite evidence"),
    (12, "Code",                   "phase12_code.jsonl",               1500, 5.0e-6, "Python, debugging, code explanation"),
    (13, "Structured Output",      "phase13_structured_output.jsonl",  1000, 4.0e-6, "JSON, markdown, formatted responses"),
    (14, "Swahili Fluency",        "phase14_kenya_swahili.jsonl",      2000, 5.0e-6, "Full Swahili conversations, code-switching"),
    (15, "Safety + Alignment",     "phase15_safety.jsonl",             1000, 3.0e-6, "Safe refusals, honesty, no hallucination"),
    (16, "DPO Alignment",          "phase16_dpo_pairs.jsonl",          1500, 8.0e-7, "Preference training: concise > verbose, honest > hallucinated"),
]
PHASES_BY_ID = {p[0]: p for p in PHASES}


# ── Setup ─────────────────────────────────────────────────────────────────────
def setup_repo():
    """Clone or update the repo from GitHub."""
    parent = "/content/miss-yaya"
    if os.path.exists(ROOT):
        print("Repo already cloned — pulling latest...")
        result = subprocess.run(["git", "pull"], cwd=ROOT, capture_output=True, text=True)
        print(result.stdout.strip() or "Already up-to-date.")
    else:
        print("Cloning yaya repo from GitHub...")
        os.makedirs(parent, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/jaylink-coder/miss-yaya.git", parent],
            check=True,
        )
        print("Clone complete.")

    # Install dependencies
    req_file = os.path.join(ROOT, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
            check=False,
        )


def setup_drive():
    """Mount Google Drive if available and sync phase_done.json."""
    drive_mounted = os.path.exists(DRIVE_MOUNT)
    if not drive_mounted:
        print("Google Drive not mounted. Checkpoints will NOT persist across sessions.")
        print("To enable persistence, run in a cell before this script:")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        return False

    os.makedirs(DRIVE_CKPT, exist_ok=True)
    # Sync phase_done from Drive → local (so we resume correctly)
    os.makedirs(os.path.dirname(PHASE_DONE_LOCAL), exist_ok=True)
    if os.path.exists(PHASE_DONE_DRIVE) and not os.path.exists(PHASE_DONE_LOCAL):
        shutil.copy(PHASE_DONE_DRIVE, PHASE_DONE_LOCAL)
        print(f"  Synced phase_done.json from Drive")
    return True


def setup_hf(token):
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)


def get_hf_token():
    """Get HF token from Colab secrets, env, or prompt."""
    # 1. Try Colab userdata
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass

    # 2. Try environment
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        return tok

    # 3. Check .env file in repo
    env_file = os.path.join(ROOT, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    tok = line.split("=", 1)[1].strip()
                    if tok:
                        return tok

    return ""


# ── Drive persistence ──────────────────────────────────────────────────────────
def backup_to_drive(local_ckpt_dir, phase_id, step):
    """Copy checkpoint to Google Drive for persistence."""
    if not os.path.exists(DRIVE_MOUNT):
        return
    tag = f"curriculum-phase{phase_id:02d}-step{step:05d}"
    dest = os.path.join(DRIVE_CKPT, tag)
    os.makedirs(dest, exist_ok=True)
    for fname in os.listdir(local_ckpt_dir):
        src = os.path.join(local_ckpt_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, fname))
    print(f"  Backed up to Drive: {dest}")

    # Sync phase_done to Drive
    if os.path.exists(PHASE_DONE_LOCAL):
        shutil.copy(PHASE_DONE_LOCAL, PHASE_DONE_DRIVE)
        print(f"  Synced phase_done.json to Drive")


def restore_from_drive():
    """If no local checkpoint, try to restore latest from Drive."""
    if not os.path.exists(DRIVE_CKPT):
        return None

    # Find best checkpoint in Drive
    for prefix in _HUB_PREFIXES:
        matches = sorted(
            [d for d in os.listdir(DRIVE_CKPT) if d.startswith(prefix)],
            reverse=True,
        )
        if matches:
            drive_ckpt = os.path.join(DRIVE_CKPT, matches[0])
            model_pt = os.path.join(drive_ckpt, "model.pt")
            if os.path.exists(model_pt):
                # Copy to local fast storage
                local_dest = os.path.join(CKPT_BASE, "yaya-125m-curriculum", matches[0])
                if not os.path.exists(local_dest):
                    print(f"  Restoring from Drive: {matches[0]}")
                    shutil.copytree(drive_ckpt, local_dest)
                return os.path.join(local_dest, "model.pt")
    return None


# ── Hub utilities ──────────────────────────────────────────────────────────────
def pull_best_checkpoint(token, local_dir):
    """Pull best available checkpoint from HF Hub."""
    from huggingface_hub import list_repo_files, hf_hub_download
    os.makedirs(local_dir, exist_ok=True)

    print("Scanning HF Hub for checkpoints...")
    try:
        files = list(list_repo_files(HUB_REPO, token=token))
    except Exception as e:
        print(f"  Hub scan failed: {e}")
        return None

    ckpt_dirs = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2 and "model.pt" in f:
            ckpt_dirs.add(parts[0])

    if not ckpt_dirs:
        print("  No checkpoints found on Hub")
        return None

    best = None
    for prefix in _HUB_PREFIXES:
        matches = [d for d in ckpt_dirs if d.startswith(prefix)]
        if matches:
            def step_num(name):
                for p in reversed(name.split("-")):
                    if p.isdigit():
                        return int(p)
                return 0
            best = sorted(matches, key=step_num, reverse=True)[0]
            break

    if not best:
        best = sorted(ckpt_dirs)[-1]

    print(f"  Using checkpoint: {best}")
    ckpt_local = os.path.join(local_dir, best)
    os.makedirs(ckpt_local, exist_ok=True)

    for fname in ["model.pt", "metadata.json"]:
        hub_path = f"{best}/{fname}"
        if hub_path in files:
            local_path = os.path.join(ckpt_local, fname)
            if not os.path.exists(local_path):
                print(f"  Downloading {hub_path}...")
                hf_hub_download(repo_id=HUB_REPO, filename=hub_path,
                                local_dir=ckpt_local,
                                local_dir_use_symlinks=False, token=token)

    model_pt = os.path.join(ckpt_local, "model.pt")
    if os.path.exists(model_pt):
        print(f"  Checkpoint ready: {model_pt}")
        return model_pt
    return None


def push_checkpoint(token, local_ckpt_dir, phase_id, step):
    """Push phase checkpoint to HF Hub."""
    from huggingface_hub import HfApi
    tag = f"curriculum-phase{phase_id:02d}-step{step:05d}"
    print(f"  Pushing {tag} to HF Hub...")
    api = HfApi(token=token)
    for fname in os.listdir(local_ckpt_dir):
        fpath = os.path.join(local_ckpt_dir, fname)
        if os.path.isfile(fpath):
            try:
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f"{tag}/{fname}",
                    repo_id=HUB_REPO,
                    token=token,
                )
            except Exception as e:
                print(f"    WARNING: upload failed for {fname}: {e}")
    print(f"  Pushed: {tag}")
    return tag


# ── Phase management ───────────────────────────────────────────────────────────
def load_done():
    if os.path.exists(PHASE_DONE_LOCAL):
        with open(PHASE_DONE_LOCAL) as f:
            return set(json.load(f).get("completed", []))
    return set()


def mark_done(phase_id):
    os.makedirs(os.path.dirname(PHASE_DONE_LOCAL), exist_ok=True)
    done = load_done()
    done.add(phase_id)
    with open(PHASE_DONE_LOCAL, "w") as f:
        json.dump({"completed": sorted(done)}, f)
    # Sync to Drive immediately
    if os.path.exists(DRIVE_MOUNT):
        os.makedirs(DRIVE_CKPT, exist_ok=True)
        shutil.copy(PHASE_DONE_LOCAL, PHASE_DONE_DRIVE)


def next_phase():
    done = load_done()
    for ph_id, *_ in PHASES:
        if ph_id not in done:
            return ph_id
    return None


# ── GPU detection & speed config ──────────────────────────────────────────────
def detect_gpu():
    """Return (gpu_name, vram_gb, batch_size, grad_accum, precision_flag)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ("cpu", 0, 2, 16, "")
        name = torch.cuda.get_device_name(0).upper()
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # A100 / H100 (40-80 GB) — large batch + bf16
        if vram >= 35:
            return (name, vram, 16, 2, "--bf16")
        # L4 / A10G (24 GB) — medium batch + fp16
        if vram >= 20:
            return (name, vram, 8, 4, "--fp16")
        # T4 (16 GB) — standard config + fp16
        if vram >= 14:
            return (name, vram, 4, 8, "--fp16")
        # Smaller GPU — tiny batch, no precision flag
        return (name, vram, 2, 16, "")
    except Exception:
        return ("unknown", 0, 4, 8, "")


def clear_memory():
    """Free GPU memory between phases."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── Hang-safe subprocess ───────────────────────────────────────────────────────
class _Heartbeat(threading.Thread):
    """Print a dot every 60 s so Colab doesn't think the kernel is idle."""
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(60):
            print(".", end="", flush=True)

    def stop(self):
        self._stop.set()


def run_subprocess(cmd, cwd, timeout_sec):
    """
    Run cmd with timeout. Kill cleanly on hang or timeout.
    Returns subprocess.CompletedProcess-like namedtuple (returncode).
    """
    hb = _Heartbeat()
    hb.start()
    proc = None
    try:
        proc = subprocess.Popen(cmd, cwd=cwd)
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"\n  TIMEOUT after {timeout_sec//60} min — killing process")
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    finally:
        hb.stop()
    return proc


# ── Training ───────────────────────────────────────────────────────────────────
def prepare_data(phase_id, data_file, replay_ratio=0.15):
    """Mix phase data with replay from prior phases."""
    import random
    random.seed(42 + phase_id)

    phase_path = os.path.join(CURR_DIR, data_file)
    if not os.path.exists(phase_path):
        print(f"  Phase data not found: {phase_path}")
        gen_script = os.path.join(ROOT, "scripts/generate_curriculum_data.py")
        if os.path.exists(gen_script):
            subprocess.run([sys.executable, gen_script, "--phase", str(phase_id)], cwd=ROOT)
        if not os.path.exists(phase_path):
            print(f"  ERROR: Could not generate phase data")
            return None

    with open(phase_path, encoding="utf-8") as f:
        phase_data = [l.strip() for l in f if l.strip()]

    replay_data = []
    if replay_ratio > 0 and phase_id > 1:
        n_replay = max(1, int(len(phase_data) * replay_ratio))
        prior_files = [
            os.path.join(CURR_DIR, PHASES_BY_ID[i][2])
            for i in range(max(1, phase_id - 3), phase_id)
            if i in PHASES_BY_ID and os.path.exists(os.path.join(CURR_DIR, PHASES_BY_ID[i][2]))
        ]
        replay_pool = []
        for pf in prior_files:
            with open(pf, encoding="utf-8") as f:
                replay_pool.extend([l.strip() for l in f if l.strip()])
        if replay_pool:
            replay_data = random.sample(replay_pool, min(n_replay, len(replay_pool)))

    all_data = phase_data + replay_data
    random.shuffle(all_data)

    train_file = f"/content/phase{phase_id:02d}_train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for line in all_data:
            f.write(line + "\n")

    print(f"  Training data: {len(phase_data)} phase + {len(replay_data)} replay = {len(all_data)} total")
    return train_file


def run_training(checkpoint_path, train_file, phase_id, steps, lr):
    """Run SFT training for one phase."""
    output_dir = f"{CKPT_BASE}/yaya-125m-curriculum/phase{phase_id:02d}"
    os.makedirs(output_dir, exist_ok=True)

    if phase_id == 16:
        return run_dpo(checkpoint_path, train_file, output_dir, steps, lr)

    cmd = [
        sys.executable, os.path.join(ROOT, "scripts/train_sft.py"),
        "--model_config",  os.path.join(ROOT, "configs/model/yaya_125m.yaml"),
        "--data_file",     train_file,
        "--output_dir",    output_dir,
        "--pretrain_checkpoint", checkpoint_path,
        "--max_steps",     str(steps),
        "--learning_rate", str(lr),
        "--per_device_batch_size", "4",
        "--gradient_accumulation_steps", "8",
        "--max_seq_length", "512",
        "--save_steps",    str(steps),
        "--warmup_steps",  "100",
        "--lr_scheduler",  "cosine",
        "--weight_decay",  "0.01",
        "--max_grad_norm", "1.0",
    ]

    print(f"\n  Running phase {phase_id} training ({steps} steps, lr={lr})")
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start
    print(f"  Training complete in {elapsed/60:.1f} min (exit {result.returncode})")

    model_files = glob.glob(f"{output_dir}/**/model.pt", recursive=True)
    if model_files:
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0]

    ckpt_dirs = sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=os.path.getmtime, reverse=True)
    if ckpt_dirs:
        mp = os.path.join(ckpt_dirs[0], "model.pt")
        if os.path.exists(mp):
            return mp

    print(f"  WARNING: No output checkpoint found in {output_dir}")
    return checkpoint_path


def run_dpo(checkpoint_path, train_file, output_dir, steps, lr):
    """Run DPO alignment for phase 16."""
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts/train_dpo.py"),
        "--model_config", os.path.join(ROOT, "configs/model/yaya_125m.yaml"),
        "--data_file",    train_file,
        "--output_dir",   output_dir,
        "--sft_checkpoint", checkpoint_path,
        "--max_steps",    str(steps),
        "--learning_rate", str(lr),
        "--beta",         "0.1",
    ]
    print(f"\n  Running DPO phase 16 ({steps} steps, lr={lr})")
    subprocess.run(cmd, cwd=ROOT)
    model_files = glob.glob(f"{output_dir}/**/model.pt", recursive=True)
    if model_files:
        return sorted(model_files, key=os.path.getmtime, reverse=True)[0]
    return checkpoint_path


# ── Benchmark ──────────────────────────────────────────────────────────────────
def run_benchmark(checkpoint_path, phase_id):
    """Run guarded + model-only benchmark."""
    print(f"\n  Running benchmark for phase {phase_id}...")
    ckpt_dir = os.path.dirname(checkpoint_path)
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts/benchmark.py"),
        "--checkpoint", ckpt_dir, "--dual",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    # Print summary tables only
    in_table = False
    for line in result.stdout.split("\n"):
        if any(k in line for k in ["Yaya Benchmark", "OVERALL", "====", "Guard lift", "DUAL"]):
            in_table = True
        if in_table:
            print("   ", line)
        if "Results saved" in line:
            break
    if result.returncode != 0 and result.stderr:
        print("  BENCHMARK ERROR:", result.stderr[-300:])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      default="auto",
                        help="Phase number (1-16), range '1-4', or 'auto'")
    parser.add_argument("--token",      default="",
                        help="HuggingFace token (or set HF_TOKEN env / Colab secret)")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--no-push",    action="store_true",
                        help="Skip HF Hub push (still backs up to Drive)")
    parser.add_argument("--no-drive",   action="store_true",
                        help="Skip Google Drive backup")
    parser.add_argument("--no-clone",   action="store_true",
                        help="Skip git clone/pull (repo already present)")
    parser.add_argument("--steps",      type=int, default=None,
                        help="Override steps for this phase")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    if not args.no_clone:
        setup_repo()

    sys.path.insert(0, ROOT)  # re-insert after possible clone

    drive_available = False
    if not args.no_drive:
        drive_available = setup_drive()

    hf_token = args.token or get_hf_token()
    if not hf_token:
        print("WARNING: No HF_TOKEN — Hub push/pull disabled.")
        print("  Set via: Colab Secrets → HF_TOKEN, or --token flag")
    else:
        setup_hf(hf_token)

    # ── Determine phases ─────────────────────────────────────────────────────
    if args.phase == "auto":
        ph = next_phase()
        if ph is None:
            print("All 16 phases complete!")
            return
        phases_to_run = [ph]
    elif "-" in args.phase:
        a, b = args.phase.split("-")
        phases_to_run = list(range(int(a), int(b) + 1))
    else:
        phases_to_run = [int(args.phase)]

    print(f"\n{'='*60}")
    print(f"  Yaya True AI Curriculum — Phases {phases_to_run}")
    print(f"  Platform: Google Colab")
    print(f"  Drive backup: {'enabled' if drive_available else 'disabled'}")
    print(f"  HF Hub: {'enabled' if hf_token else 'disabled'}")
    print(f"{'='*60}")

    # ── Pull/restore starting checkpoint ────────────────────────────────────
    ckpt_local_dir = f"{CKPT_BASE}/yaya-125m-curriculum"
    os.makedirs(ckpt_local_dir, exist_ok=True)
    current_ckpt = None

    if hf_token:
        current_ckpt = pull_best_checkpoint(hf_token, ckpt_local_dir)

    if not current_ckpt and drive_available:
        print("Trying to restore checkpoint from Drive...")
        current_ckpt = restore_from_drive()

    if not current_ckpt:
        local_files = glob.glob(f"{CKPT_BASE}/**/*.pt", recursive=True)
        if local_files:
            current_ckpt = sorted(local_files, key=os.path.getmtime, reverse=True)[0]

    if not current_ckpt:
        print("ERROR: No checkpoint found. Cannot proceed.")
        print("  Options:")
        print("  1. Set HF_TOKEN in Colab Secrets (Colab menu → Secrets)")
        print("  2. Mount Drive with a saved checkpoint")
        sys.exit(1)

    print(f"\n  Starting checkpoint: {current_ckpt}")

    # ── Run phases ────────────────────────────────────────────────────────────
    for phase_id in phases_to_run:
        if phase_id not in PHASES_BY_ID:
            print(f"  Phase {phase_id} not defined, skipping")
            continue

        ph_id, name, data_file, steps, lr, desc = PHASES_BY_ID[phase_id]
        if args.steps:
            steps = args.steps

        print(f"\n{'─'*60}")
        print(f"  PHASE {ph_id}: {name}")
        print(f"  {desc}")
        print(f"  Steps: {steps}  LR: {lr}")
        print(f"{'─'*60}")

        if args.eval_only:
            run_benchmark(current_ckpt, phase_id)
            continue

        train_file = prepare_data(phase_id, data_file)
        if not train_file:
            print(f"  SKIP: No data for phase {phase_id}")
            continue

        output_ckpt = run_training(current_ckpt, train_file, phase_id, steps, lr)

        # Push to Hub
        if hf_token and not args.no_push:
            output_dir = os.path.dirname(output_ckpt)
            push_checkpoint(hf_token, output_dir, phase_id, steps)

        # Backup to Drive
        if drive_available and not args.no_drive:
            output_dir = os.path.dirname(output_ckpt)
            backup_to_drive(output_dir, phase_id, steps)

        run_benchmark(output_ckpt, phase_id)
        mark_done(phase_id)
        current_ckpt = output_ckpt

        print(f"\n  Phase {phase_id} complete. Checkpoint: {current_ckpt}")

    print(f"\n{'='*60}")
    print(f"  Done. Completed phases: {sorted(load_done())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
