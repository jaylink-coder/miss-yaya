"""Kaggle runner for the Yaya True AI 16-phase curriculum.

This script runs on Kaggle T4 GPU. It:
1. Pulls the latest checkpoint from HuggingFace Hub
2. Auto-detects which phase to run next (or uses --phase N)
3. Trains for ≤2000 steps on curriculum data for that phase
4. Pushes the checkpoint back to HF Hub with phase tag
5. Runs the benchmark (guarded + model-only)
6. Marks the phase as complete and loops to next

Usage (in Kaggle notebook cell):
    !python scripts/kaggle_run_phases.py
    !python scripts/kaggle_run_phases.py --phase 3
    !python scripts/kaggle_run_phases.py --phase 1-4
    !python scripts/kaggle_run_phases.py --eval-only
"""

import argparse, json, os, sys, time, shutil, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = "/kaggle/working/miss-yaya/yaya-ai"
CKPT_BASE     = "/kaggle/working/checkpoints"
CURR_DIR      = os.path.join(ROOT, "data/sft/curriculum")
PHASE_DONE    = os.path.join(CKPT_BASE, "yaya-125m-curriculum", "phase_done.json")
HUB_REPO      = "Jaylink-coder/yaya-125m"

# ── Hub checkpoint priority (highest → lowest) ─────────────────────────────────
_HUB_PREFIXES = [
    "curriculum-phase",   # latest curriculum phase
    "p8-checkpoint",      # legacy naming
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


# ── Hub utilities ──────────────────────────────────────────────────────────────
def setup_hf(token):
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)


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

    # Find all checkpoint dirs
    ckpt_dirs = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2 and parts[0].endswith((".pt", "")) and "model.pt" in f:
            ckpt_dirs.add(parts[0])

    if not ckpt_dirs:
        print("  No checkpoints found on Hub")
        return None

    # Pick best by prefix priority
    best = None
    for prefix in _HUB_PREFIXES:
        matches = [d for d in ckpt_dirs if d.startswith(prefix)]
        if matches:
            # sort by step number (last numeric segment)
            def step_num(name):
                parts = name.split("-")
                for p in reversed(parts):
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
    if os.path.exists(PHASE_DONE):
        with open(PHASE_DONE) as f:
            return set(json.load(f).get("completed", []))
    return set()


def mark_done(phase_id):
    os.makedirs(os.path.dirname(PHASE_DONE), exist_ok=True)
    done = load_done()
    done.add(phase_id)
    with open(PHASE_DONE, "w") as f:
        json.dump({"completed": sorted(done)}, f)


def next_phase():
    done = load_done()
    for ph_id, *_ in PHASES:
        if ph_id not in done:
            return ph_id
    return None


# ── Training ───────────────────────────────────────────────────────────────────
def prepare_data(phase_id, data_file, replay_ratio=0.15):
    """Mix phase data with replay from prior phases."""
    import random
    random.seed(42 + phase_id)

    phase_path = os.path.join(CURR_DIR, data_file)
    if not os.path.exists(phase_path):
        print(f"  Phase data not found: {phase_path}")
        # Try generating it
        gen_script = os.path.join(ROOT, "scripts/generate_curriculum_data.py")
        if os.path.exists(gen_script):
            subprocess.run([sys.executable, gen_script, "--phase", str(phase_id)],
                           cwd=ROOT)
        if not os.path.exists(phase_path):
            print(f"  ERROR: Could not generate phase data")
            return None

    with open(phase_path, encoding="utf-8") as f:
        phase_data = [l.strip() for l in f if l.strip()]

    # Replay: sample from earlier phases
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

    # Write combined training file
    train_file = f"/kaggle/working/phase{phase_id:02d}_train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for line in all_data:
            f.write(line + "\n")

    print(f"  Training data: {len(phase_data)} phase + {len(replay_data)} replay = {len(all_data)} total")
    return train_file


def run_training(checkpoint_path, train_file, phase_id, steps, lr):
    """Run SFT training for one phase."""
    output_dir = f"{CKPT_BASE}/yaya-125m-curriculum/phase{phase_id:02d}"
    os.makedirs(output_dir, exist_ok=True)

    # Phase 16 is DPO
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
        "--save_steps",    str(steps),  # save once at end
        "--warmup_steps",  "100",
        "--lr_scheduler",  "cosine",
        "--weight_decay",  "0.01",
        "--max_grad_norm", "1.0",
    ]

    print(f"\n  Running: {' '.join(cmd[:4])} ... (phase {phase_id}, {steps} steps, lr={lr})")
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start
    print(f"  Training complete in {elapsed/60:.1f} min (exit code {result.returncode})")

    # Find output checkpoint
    model_files = glob.glob(f"{output_dir}/**/model.pt", recursive=True)
    if model_files:
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0]

    # Fallback: look for checkpoint-NNNNN dirs
    ckpt_dirs = sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=os.path.getmtime, reverse=True)
    if ckpt_dirs:
        mp = os.path.join(ckpt_dirs[0], "model.pt")
        if os.path.exists(mp):
            return mp

    print(f"  WARNING: No output checkpoint found in {output_dir}")
    return checkpoint_path  # fallback to input


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
    print(f"\n  Running DPO (phase 16, {steps} steps, lr={lr})")
    subprocess.run(cmd, cwd=ROOT)
    model_files = glob.glob(f"{output_dir}/**/model.pt", recursive=True)
    if model_files:
        return sorted(model_files, key=os.path.getmtime, reverse=True)[0]
    return checkpoint_path


# ── Benchmark ──────────────────────────────────────────────────────────────────
def run_benchmark(checkpoint_path, phase_id):
    """Run guarded + model-only benchmark."""
    print(f"\n  Running benchmark for phase {phase_id}...")
    for mode in ["--dual"]:
        cmd = [
            sys.executable, os.path.join(ROOT, "scripts/benchmark.py"),
            "--checkpoint", checkpoint_path, mode,
        ]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        # Print just the summary table
        lines = result.stdout.split("\n")
        in_table = False
        for line in lines:
            if "Yaya Benchmark" in line or "OVERALL" in line or "====" in line:
                in_table = True
            if in_table:
                print("   ", line)
            if line.startswith("  Results saved") or "Failures:" in line:
                break


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",     default="auto",
                        help="Phase number (1-16), range '1-4', or 'auto'")
    parser.add_argument("--token",     default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-push",   action="store_true")
    parser.add_argument("--steps",     type=int, default=None,
                        help="Override steps for this phase")
    args = parser.parse_args()

    # Get token
    hf_token = args.token
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if not hf_token:
        print("WARNING: No HF_TOKEN — Hub push/pull disabled")

    # Setup HF
    if hf_token:
        setup_hf(hf_token)

    # Determine phases to run
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
    print(f"{'='*60}")

    # Pull starting checkpoint
    ckpt_local_dir = f"{CKPT_BASE}/yaya-125m-curriculum"
    os.makedirs(ckpt_local_dir, exist_ok=True)

    if hf_token:
        current_ckpt = pull_best_checkpoint(hf_token, ckpt_local_dir)
    else:
        # Find local checkpoint
        local_files = glob.glob(f"{CKPT_BASE}/**/*.pt", recursive=True)
        current_ckpt = sorted(local_files, key=os.path.getmtime, reverse=True)[0] if local_files else None

    if not current_ckpt:
        print("ERROR: No checkpoint found. Cannot proceed.")
        sys.exit(1)

    print(f"\n  Starting checkpoint: {current_ckpt}")

    # ── Run phases ─────────────────────────────────────────────────────────────
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

        # Prepare training data
        train_file = prepare_data(phase_id, data_file)
        if not train_file:
            print(f"  SKIP: No data for phase {phase_id}")
            continue

        # Train
        output_ckpt = run_training(current_ckpt, train_file, phase_id, steps, lr)

        # Push to Hub
        if hf_token and not args.no_push:
            output_dir = os.path.dirname(output_ckpt)
            push_checkpoint(hf_token, output_dir, phase_id, steps)

        # Benchmark
        run_benchmark(output_ckpt, phase_id)

        # Mark done
        mark_done(phase_id)

        # Update for next phase
        current_ckpt = output_ckpt
        print(f"\n  Phase {phase_id} complete. Checkpoint: {current_ckpt}")

    print(f"\n{'='*60}")
    print(f"  Done. Completed phases: {sorted(load_done())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
