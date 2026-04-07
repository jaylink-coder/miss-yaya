#!/usr/bin/env python3
"""Run a single milestone phase from the Yaya True AI curriculum.

Reads milestones_v2.yaml, finds/generates the training data, creates a
temporary training config, trains for <=2000 steps, then runs phase eval.

Usage:
    python scripts/run_milestone.py --phase 1              # run phase 1
    python scripts/run_milestone.py --phase 1 --eval-only  # eval only
    python scripts/run_milestone.py --phase 1 --gen-only   # generate data only
    python scripts/run_milestone.py --auto                 # auto-detect next phase
"""
import argparse, json, os, sys, yaml, glob, shutil, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MILESTONES_PATH = os.path.join(ROOT, "configs", "training", "milestones_v2.yaml")
CURRICULUM_DIR = os.path.join(ROOT, "data", "sft", "curriculum")
CHECKPOINT_BASE = os.path.join(ROOT, "checkpoints")


def load_milestones():
    with open(MILESTONES_PATH, "r") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint():
    """Find the most recent yaya-125m checkpoint (SFT or curriculum)."""
    # Check curriculum checkpoints first (phase-specific)
    curriculum_dir = os.path.join(CHECKPOINT_BASE, "yaya-125m-curriculum")
    if os.path.isdir(curriculum_dir):
        latest_file = os.path.join(curriculum_dir, "latest")
        if os.path.isfile(latest_file):
            with open(latest_file) as f:
                ckpt = f.read().strip()
            path = os.path.join(curriculum_dir, ckpt)
            if os.path.isfile(path):
                return path

    # Fall back to SFT checkpoints
    for subdir in ["yaya-125m-sft", "yaya-125m-dpo", "yaya-125m"]:
        d = os.path.join(CHECKPOINT_BASE, subdir)
        if os.path.isdir(d):
            latest_file = os.path.join(d, "latest")
            if os.path.isfile(latest_file):
                with open(latest_file) as f:
                    ckpt = f.read().strip()
                path = os.path.join(d, ckpt)
                if os.path.isfile(path):
                    return path

    # Search for any .pt file
    for pattern in ["checkpoints/yaya-125m*/**/*.pt", "checkpoints/**/*.pt"]:
        files = glob.glob(os.path.join(ROOT, pattern), recursive=True)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]

    return None


def detect_next_phase():
    """Detect which phase to run next based on existing curriculum checkpoints."""
    curriculum_dir = os.path.join(CHECKPOINT_BASE, "yaya-125m-curriculum")
    if not os.path.isdir(curriculum_dir):
        return 1

    completed = set()
    for f in os.listdir(curriculum_dir):
        if f.startswith("phase") and f.endswith(".done"):
            try:
                phase_num = int(f.replace("phase", "").replace(".done", ""))
                completed.add(phase_num)
            except ValueError:
                pass

    for i in range(1, 17):
        if i not in completed:
            return i
    return None


def ensure_data(phase_id, milestones):
    """Ensure training data exists for the given phase."""
    phase_info = None
    for p in milestones["phases"]:
        if p["id"] == phase_id:
            phase_info = p
            break

    if phase_info is None:
        print(f"ERROR: Phase {phase_id} not found in milestones")
        sys.exit(1)

    data_file = os.path.join(ROOT, phase_info["data_file"])

    if os.path.isfile(data_file):
        count = sum(1 for _ in open(data_file, encoding="utf-8"))
        print(f"  Data exists: {data_file} ({count} examples)")
        return data_file, phase_info

    # Generate data
    print(f"  Generating data for phase {phase_id}...")
    scripts_dir = os.path.join(ROOT, "scripts")
    sys.path.insert(0, scripts_dir)
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "generate_curriculum_data.py"),
         "--phase", str(phase_id)],
        cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR generating data: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

    if os.path.isfile(data_file):
        return data_file, phase_info

    # Try alternate naming
    alt_files = glob.glob(os.path.join(CURRICULUM_DIR, f"phase{phase_id:02d}_*.jsonl"))
    if alt_files:
        return alt_files[0], phase_info

    print(f"  ERROR: Data file not found after generation: {data_file}")
    sys.exit(1)


def augment_with_replay(data_file, phase_id, milestones, replay_ratio=0.20):
    """Mix current phase data with replay from earlier phases to prevent forgetting."""
    if phase_id <= 1:
        return data_file  # No replay for first phase

    current = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            current.append(line.strip())

    if not current:
        return data_file

    # Collect replay from previous phases
    replay_pool = []
    for prev_id in range(1, phase_id):
        prev_files = glob.glob(os.path.join(CURRICULUM_DIR, f"phase{prev_id:02d}_*.jsonl"))
        for pf in prev_files:
            with open(pf, "r", encoding="utf-8") as f:
                for line in f:
                    replay_pool.append(line.strip())

    if not replay_pool:
        return data_file

    # Also include existing SFT data for additional replay diversity
    extra_files = [
        os.path.join(ROOT, "data", "sft", "yaya_instruct_clean.jsonl"),
        os.path.join(ROOT, "data", "sft", "yaya_factual_qa.jsonl"),
        os.path.join(ROOT, "data", "sft", "yaya_short_qa.jsonl"),
    ]
    for ef in extra_files:
        if os.path.isfile(ef):
            try:
                with open(ef, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 50:  # Cap at 50 per file
                            break
                        replay_pool.append(line.strip())
            except Exception:
                pass

    import random
    random.seed(42 + phase_id)

    # Calculate replay count
    n_replay = max(1, int(len(current) * replay_ratio / (1 - replay_ratio)))
    n_replay = min(n_replay, len(replay_pool))
    replay_sample = random.sample(replay_pool, n_replay)

    # Write mixed file
    mixed_path = data_file.replace(".jsonl", "_mixed.jsonl")
    all_data = current + replay_sample
    random.shuffle(all_data)

    with open(mixed_path, "w", encoding="utf-8") as f:
        for line in all_data:
            f.write(line + "\n")

    print(f"  Mixed: {len(current)} current + {n_replay} replay = {len(all_data)} total")
    return mixed_path


def create_phase_config(phase_id, phase_info, data_file, milestones, checkpoint_path):
    """Create a temporary training config YAML for this phase."""
    steps = milestones.get("steps_per_phase", 2000)
    lr = milestones.get("base_lr", 1.5e-5)
    warmup = milestones.get("warmup_steps", 200)
    max_seq = milestones.get("max_seq_length", 512)

    is_dpo = phase_info.get("phase_type") == "dpo"

    config = {
        "seed": 42 + phase_id,
        "training": {
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": lr,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine",
            "warmup_steps": warmup,
            "min_lr_ratio": 0.05,
            "max_steps": steps,
            "max_seq_length": max_seq,
            "dtype": "float16",
        },
        "checkpointing": {
            "save_steps": 500,
            "save_dir": f"checkpoints/yaya-125m-curriculum",
            "keep_last_n": 3,
        },
        "logging": {
            "log_steps": 50,
            "wandb_project": "yaya-ai",
            "wandb_run_name": f"yaya-125m-phase{phase_id:02d}-{phase_info['name'].lower().replace(' ', '-')}",
        },
        "eval": {
            "eval_steps": 500,
            "eval_samples": 100,
        },
        "data": {
            "train_data": data_file,
            "eval_data": data_file,
            "tokenizer_path": "data/tokenizer/yaya_tokenizer.model",
            "num_workers": 0,
        },
        "distributed": {
            "strategy": "none",
            "gradient_checkpointing": True,
            "cpu_offload": False,
        },
    }

    # Write temp config
    config_path = os.path.join(ROOT, "configs", "training", f"phase{phase_id:02d}_auto.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Config: {config_path}")
    return config_path


def mark_phase_done(phase_id):
    """Mark a phase as completed."""
    curriculum_dir = os.path.join(CHECKPOINT_BASE, "yaya-125m-curriculum")
    os.makedirs(curriculum_dir, exist_ok=True)
    marker = os.path.join(curriculum_dir, f"phase{phase_id:02d}.done")
    with open(marker, "w") as f:
        f.write(f"Phase {phase_id} completed\n")


def run_training(phase_id, phase_info, config_path, checkpoint_path):
    """Run the actual training for this phase."""
    import subprocess

    cmd = [
        sys.executable, "-u",
        os.path.join(ROOT, "scripts", "train_sft.py"),
        "--model_config", os.path.join(ROOT, "configs", "model", "yaya_125m.yaml"),
        "--train_config", config_path,
    ]

    if checkpoint_path:
        cmd.extend(["--pretrain_checkpoint", checkpoint_path])

    print(f"\n  Running: {' '.join(os.path.basename(c) for c in cmd)}")
    print(f"  Phase {phase_id}: {phase_info['name']}")
    print(f"  Checkpoint: {checkpoint_path or 'random init'}")
    print(f"  {'='*60}")

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def run_eval(phase_id, phase_info):
    """Run evaluation for this phase."""
    import subprocess

    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "eval_milestone.py"),
        "--phase", str(phase_id),
    ]

    print(f"\n  Running evaluation for phase {phase_id}...")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run Yaya curriculum milestone")
    parser.add_argument("--phase", type=int, help="Phase number (1-16)")
    parser.add_argument("--auto", action="store_true", help="Auto-detect next phase")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--gen-only", action="store_true", help="Generate data only")
    parser.add_argument("--no-replay", action="store_true", help="Skip replay augmentation")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training eval")
    args = parser.parse_args()

    milestones = load_milestones()

    # Determine phase
    if args.auto:
        phase_id = detect_next_phase()
        if phase_id is None:
            print("All 16 phases completed!")
            return
        print(f"Auto-detected next phase: {phase_id}")
    elif args.phase:
        phase_id = args.phase
    else:
        parser.error("Specify --phase N or --auto")
        return

    print(f"\n{'='*60}")
    print(f"  YAYA TRUE AI — Phase {phase_id}")
    print(f"{'='*60}\n")

    # Step 1: Ensure data
    data_file, phase_info = ensure_data(phase_id, milestones)

    if args.gen_only:
        print("Data generated. Exiting (--gen-only).")
        return

    # Step 2: Augment with replay
    if not args.no_replay and phase_id > 1:
        replay_ratio = milestones.get("replay_ratio", 0.20)
        data_file = augment_with_replay(data_file, phase_id, milestones, replay_ratio)

    # Step 3: Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    print(f"  Base checkpoint: {checkpoint_path or 'None (random init)'}")

    if args.eval_only:
        run_eval(phase_id, phase_info)
        return

    # Step 4: Create training config
    config_path = create_phase_config(phase_id, phase_info, data_file, milestones, checkpoint_path)

    # Step 5: Train
    print(f"\n  Starting training: {phase_info['name']}")
    print(f"  Description: {phase_info.get('description', '').strip()}")
    success = run_training(phase_id, phase_info, config_path, checkpoint_path)

    if success:
        mark_phase_done(phase_id)
        print(f"\n  Phase {phase_id} training COMPLETE")
    else:
        print(f"\n  Phase {phase_id} training FAILED")
        sys.exit(1)

    # Step 6: Evaluate
    if not args.no_eval:
        run_eval(phase_id, phase_info)

    print(f"\n{'='*60}")
    print(f"  Phase {phase_id} ({phase_info['name']}) — DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
