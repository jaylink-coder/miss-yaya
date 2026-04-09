"""Yaya — Stage-Based Training Runner (Colab / Kaggle)

One stage at a time. Each stage is a complete, self-contained training block.
Run it once per Colab session. When it finishes — move to the next stage.

Usage:
    # Run a full stage (recommended — one session = one stage)
    !python scripts/colab_run_phases.py --stage 1

    # Resume within a stage if the session was interrupted
    !python scripts/colab_run_phases.py --stage 1 --resume

    # Run a specific phase within a stage
    !python scripts/colab_run_phases.py --stage 1 --phase 2

    # Run a specific sub-phase
    !python scripts/colab_run_phases.py --stage 1 --phase 2 --subphase b

    # Check what's done
    !python scripts/colab_run_phases.py --status

    # Benchmark current best checkpoint
    !python scripts/colab_run_phases.py --benchmark

Stages:
    1  Language & Knowledge Foundation   (Phases 1-2)
    2  Identity, Personality & Comms     (Phases 3-4)
    3  Instruction Following             (Phase  5)
    4  Reasoning Engine                  (Phases 6-8)
    5  Kenyan & Swahili Identity         (Phases 9-10)
    6  Technical Skills                  (Phases 11-13)
    7  Alignment & Values                (Phases 14-15)
"""

import argparse, gc, glob, json, os, random, shutil, subprocess, sys, threading, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Environment detection ──────────────────────────────────────────────────────
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

HUB_REPO           = "Jaylink-coder/yaya-125m"
PROGRESS_FILE_LOCAL = os.path.join(CKPT_BASE, "yaya-125m-curriculum", "progress.json")
PROGRESS_FILE_DRIVE = os.path.join(DRIVE_DIR, "progress.json") if DRIVE_DIR else None

sys.path.insert(0, ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE & CURRICULUM DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

STAGE_NAMES = {
    1: "Language & Knowledge Foundation",
    2: "Identity, Personality & Communication",
    3: "Instruction Following & Task Execution",
    4: "Reasoning Engine",
    5: "Kenyan & Swahili Identity",
    6: "Technical Skills",
    7: "Alignment & Values",
}

# Each entry: (stage, phase, sub, name, data_file_relative, steps, lr, is_dpo)
# data_file_relative is relative to ROOT/data/sft/curriculum/
CURRICULUM = [
    # ── Stage 1: Language & Knowledge Foundation ──────────────────────────────
    (1, 1, "a", "Direct Q&A Warmup",         "phase01/1a_direct_qa.jsonl",           800, 1.5e-5, False),
    (1, 1, "b", "Reading Comprehension",      "phase01/1b_reading.jsonl",              800, 1.4e-5, False),
    (1, 1, "c", "Paraphrase & Equivalence",   "phase01/1c_paraphrase.jsonl",           800, 1.3e-5, False),
    (1, 1, "d", "Output Format Control",      "phase01/1d_format_control.jsonl",       800, 1.2e-5, False),
    (1, 2, "a", "Science & Nature",           "phase02/2a_science.jsonl",              800, 1.4e-5, False),
    (1, 2, "b", "History & Geography",        "phase02/2b_history_geo.jsonl",          800, 1.3e-5, False),
    (1, 2, "c", "Culture, Arts & Society",    "phase02/2c_culture.jsonl",              800, 1.2e-5, False),
    (1, 2, "d", "Practical & Daily Life",     "phase02/2d_practical.jsonl",            800, 1.1e-5, False),

    # ── Stage 2: Identity, Personality & Communication ────────────────────────
    (2, 3, "a", "Core Identity",              "phase03/3a_core_identity.jsonl",        600, 1.2e-5, False),
    (2, 3, "b", "Personality & Values",       "phase03/3b_personality.jsonl",          800, 1.1e-5, False),
    (2, 3, "c", "Empathy & Emotion",          "phase03/3c_empathy.jsonl",              800, 1.0e-5, False),
    (2, 3, "d", "Consistency Under Pressure", "phase03/3d_consistency.jsonl",          800, 9.0e-6, False),
    (2, 4, "a", "Greetings & Social Phrases", "phase04/4a_greetings.jsonl",            600, 1.1e-5, False),
    (2, 4, "b", "Multi-Turn Coherence",       "phase04/4b_multi_turn.jsonl",           900, 1.0e-5, False),
    (2, 4, "c", "Topic Transitions",          "phase04/4c_transitions.jsonl",          900, 9.0e-6, False),
    (2, 4, "d", "Ambiguity Handling",         "phase04/4d_ambiguity.jsonl",            800, 8.0e-6, False),

    # ── Stage 3: Instruction Following ────────────────────────────────────────
    (3, 5, "a", "Format Constraints",         "phase05/5a_format.jsonl",               900, 1.2e-5, False),
    (3, 5, "b", "Task Execution",             "phase05/5b_tasks.jsonl",                900, 1.1e-5, False),
    (3, 5, "c", "Compound Instructions",      "phase05/5c_compound.jsonl",             900, 1.0e-5, False),
    (3, 5, "d", "Role & Persona",             "phase05/5d_roles.jsonl",                700, 9.0e-6, False),

    # ── Stage 4: Reasoning Engine ─────────────────────────────────────────────
    (4, 6, "a", "Arithmetic Foundations",     "phase06/6a_arithmetic.jsonl",           900, 1.2e-5, False),
    (4, 6, "b", "Word Problems",              "phase06/6b_word_problems.jsonl",        900, 1.1e-5, False),
    (4, 6, "c", "Percentages & Fractions",    "phase06/6c_percentages.jsonl",          900, 1.0e-5, False),
    (4, 6, "d", "Estimation & Mental Math",   "phase06/6d_estimation.jsonl",           700, 9.0e-6, False),
    (4, 7, "a", "Deductive Reasoning",        "phase07/7a_deduction.jsonl",            800, 1.1e-5, False),
    (4, 7, "b", "Analogies & Patterns",       "phase07/7b_analogies.jsonl",            800, 1.0e-5, False),
    (4, 7, "c", "Counterfactuals",            "phase07/7c_counterfactuals.jsonl",      800, 9.0e-6, False),
    (4, 7, "d", "Spatial & Temporal",         "phase07/7d_spatial_temporal.jsonl",     800, 8.0e-6, False),
    (4, 8, "a", "Step-by-Step Reasoning",     "phase08/8a_step_by_step.jsonl",         900, 1.0e-5, False),
    (4, 8, "b", "Problem Decomposition",      "phase08/8b_decomposition.jsonl",        900, 9.0e-6, False),
    (4, 8, "c", "Self-Verification",          "phase08/8c_verification.jsonl",         900, 8.0e-6, False),
    (4, 8, "d", "Honest Uncertainty",         "phase08/8d_uncertainty.jsonl",          700, 7.0e-6, False),

    # ── Stage 5: Kenyan & Swahili Identity ────────────────────────────────────
    (5, 9,  "a", "Kenya Geography & Nature",  "phase09/9a_geography.jsonl",            800, 1.1e-5, False),
    (5, 9,  "b", "Kenya History & Politics",  "phase09/9b_history.jsonl",              800, 1.0e-5, False),
    (5, 9,  "c", "Kenya Culture & Society",   "phase09/9c_culture.jsonl",              800, 9.0e-6, False),
    (5, 9,  "d", "Kenya Economy & Tech",      "phase09/9d_economy.jsonl",              800, 8.0e-6, False),
    (5, 10, "a", "Swahili Vocabulary",        "phase10/10a_vocabulary.jsonl",         1000, 1.2e-5, False),
    (5, 10, "b", "Swahili Grammar",           "phase10/10b_grammar.jsonl",            1000, 1.1e-5, False),
    (5, 10, "c", "Bidirectional Translation", "phase10/10c_translation.jsonl",        1000, 1.0e-5, False),
    (5, 10, "d", "Code-Switching & Sheng",    "phase10/10d_code_switching.jsonl",     1000, 9.0e-6, False),

    # ── Stage 6: Technical Skills ─────────────────────────────────────────────
    (6, 11, "a", "Code Reading",              "phase11/11a_reading.jsonl",             900, 1.1e-5, False),
    (6, 11, "b", "Code Writing",              "phase11/11b_writing.jsonl",             900, 1.0e-5, False),
    (6, 11, "c", "Debugging",                 "phase11/11c_debugging.jsonl",           900, 9.0e-6, False),
    (6, 11, "d", "Algorithms",                "phase11/11d_algorithms.jsonl",          900, 8.0e-6, False),
    (6, 12, "a", "Tool Call Format",          "phase12/12a_tool_format.jsonl",         900, 1.1e-5, False),
    (6, 12, "b", "Single Tool Use",           "phase12/12b_single_tool.jsonl",         900, 1.0e-5, False),
    (6, 12, "c", "ReAct Pattern",             "phase12/12c_react.jsonl",               900, 9.0e-6, False),
    (6, 12, "d", "Tool Failure Handling",     "phase12/12d_failure_handling.jsonl",    900, 8.0e-6, False),
    (6, 13, "a", "JSON Output",               "phase13/13a_json.jsonl",                800, 1.0e-5, False),
    (6, 13, "b", "Markdown & Tables",         "phase13/13b_markdown.jsonl",            800, 9.0e-6, False),
    (6, 13, "c", "RAG Context Grounding",     "phase13/13c_rag.jsonl",                 800, 8.0e-6, False),
    (6, 13, "d", "Unanswerable Context",      "phase13/13d_unanswerable.jsonl",        800, 7.0e-6, False),

    # ── Stage 7: Alignment & Values ───────────────────────────────────────────
    (7, 14, "a", "Harmful Refusals",          "phase14/14a_refusals.jsonl",            800, 9.0e-6, False),
    (7, 14, "b", "Honesty & Anti-Confab",     "phase14/14b_honesty.jsonl",             800, 8.0e-6, False),
    (7, 14, "c", "Prompt Injection Resist",   "phase14/14c_injection.jsonl",           800, 7.0e-6, False),
    (7, 14, "d", "Ethical Reasoning",         "phase14/14d_ethics.jsonl",              800, 6.0e-6, False),
    (7, 15, "a", "Quality Preferences",       "phase15/15a_quality.jsonl",            1000, 5.0e-7, True),
    (7, 15, "b", "Style Preferences",         "phase15/15b_style.jsonl",              1000, 5.0e-7, True),
    (7, 15, "c", "Safety Preferences",        "phase15/15c_safety.jsonl",             1000, 5.0e-7, True),
    (7, 15, "d", "Yaya Persona Preferences",  "phase15/15d_persona.jsonl",            1000, 5.0e-7, True),
]


# ══════════════════════════════════════════════════════════════════════════════
# GPU DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU", 0, 2, 16, ""
        name = torch.cuda.get_device_name(0).upper()
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram >= 35: return name, vram, 16, 2,  "--bf16"
        if vram >= 20: return name, vram,  8, 4,  "--fp16"
        if vram >= 14: return name, vram,  4, 8,  "--fp16"
        return name, vram, 2, 16, ""
    except Exception:
        return "UNKNOWN", 0, 4, 8, ""


def clear_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def load_progress():
    """Returns set of completed 'phase.sub' keys (persists across sessions via Drive)."""
    for path in [PROGRESS_FILE_DRIVE, PROGRESS_FILE_LOCAL]:
        if path and os.path.exists(path):
            try:
                return set(json.load(open(path)).get("completed", []))
            except Exception:
                pass
    return set()


def save_progress(completed: set):
    data = {"completed": sorted(completed), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    for path in [PROGRESS_FILE_LOCAL, PROGRESS_FILE_DRIVE]:
        if not path:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


def sp_key(phase, sub):
    return f"{phase}.{sub}"


def print_status(completed):
    print("\n" + "=" * 60)
    print("  Yaya Training Progress")
    print("=" * 60)
    cur_stage = cur_phase = None
    for s, p, sub, name, _, steps, _, _ in CURRICULUM:
        key = sp_key(p, sub)
        done = "✓" if key in completed else "·"
        if s != cur_stage:
            cur_stage = s
            total_s = [e for e in CURRICULUM if e[0] == s]
            done_s  = [e for e in total_s if sp_key(e[1], e[2]) in completed]
            pct = 100 * len(done_s) // len(total_s)
            print(f"\n  Stage {s}: {STAGE_NAMES[s]}  [{len(done_s)}/{len(total_s)} — {pct}%]")
        if p != cur_phase:
            cur_phase = p
            print(f"    Phase {p}:")
        print(f"      [{done}] {p}{sub}. {name}  ({steps} steps)")
    total = len(CURRICULUM)
    done_n = len(completed)
    print(f"\n  Total: {done_n}/{total} sub-phases  ({100*done_n//total}%)")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SUBPROCESS WITH HEARTBEAT
# ══════════════════════════════════════════════════════════════════════════════

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
            print(f"\n  TIMEOUT after {timeout_sec//3600}h — killing")
            proc.kill()
            proc.wait(timeout=30)
    finally:
        hb.stop()
    return proc


# ══════════════════════════════════════════════════════════════════════════════
# HF TOKEN + HUB OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_hf_token(cli_token=""):
    if cli_token:
        return cli_token
    for getter in [
        lambda: __import__("google.colab", fromlist=["userdata"]).userdata.get("HF_TOKEN"),
        lambda: __import__("kaggle_secrets", fromlist=["UserSecretsClient"]).UserSecretsClient().get_secret("HF_TOKEN"),
    ]:
        try:
            tok = getter()
            if tok:
                return tok
        except Exception:
            pass
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        return tok
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip()
                if tok:
                    return tok
    return ""


def _checkpoint_loss(model_pt):
    """Read loss from metadata.json next to model.pt."""
    meta = os.path.join(os.path.dirname(model_pt), "metadata.json")
    if os.path.exists(meta):
        try:
            return json.load(open(meta)).get("loss")
        except Exception:
            pass
    return None


def pull_best_checkpoint(token, stage_id):
    """
    Pull the best valid starting checkpoint for the given stage from Hub.
    Tries the highest completed sub-phase of (stage_id - 1) first.
    Falls back through earlier stages, then legacy checkpoint names.
    """
    from huggingface_hub import list_repo_files, hf_hub_download
    local_dir = os.path.join(CKPT_BASE, "yaya-125m-curriculum")
    os.makedirs(local_dir, exist_ok=True)

    print("  Scanning HF Hub...")
    try:
        files = list(list_repo_files(HUB_REPO, token=token))
    except Exception as e:
        print(f"  Hub scan failed: {e}")
        return None

    hub_ckpt_dirs = set(f.split("/")[0] for f in files if "model.pt" in f)

    def step_num(name):
        for p in reversed(name.split("-")):
            if p.isdigit():
                return int(p)
        return 0

    # Build priority list: latest from stage N-1 down to stage 1, then legacy names
    prefixes = []
    for st in range(stage_id - 1, 0, -1):
        stage_phases = sorted(set(e[1] for e in CURRICULUM if e[0] == st), reverse=True)
        for ph in stage_phases:
            for sub in ["d", "c", "b", "a"]:
                prefixes.append(f"curriculum-s{st}-p{ph}{sub}")
        prefixes.append(f"curriculum-s{st}-p")  # any from this stage
    prefixes += ["curriculum-phase", "patch-checkpoint", "dpo2-checkpoint",
                 "dpo-checkpoint", "checkpoint"]

    for prefix in prefixes:
        matches = sorted([d for d in hub_ckpt_dirs if d.startswith(prefix)],
                         key=step_num, reverse=True)
        if not matches:
            continue
        best = matches[0]
        dest = os.path.join(local_dir, best)
        os.makedirs(dest, exist_ok=True)

        for fname in ["model.pt", "metadata.json"]:
            hub_path = f"{best}/{fname}"
            local_path = os.path.join(dest, fname)
            if hub_path in files and not os.path.exists(local_path):
                print(f"  Downloading {hub_path}...")
                try:
                    hf_hub_download(repo_id=HUB_REPO, filename=hub_path,
                                    local_dir=dest, local_dir_use_symlinks=False,
                                    token=token)
                except Exception as e:
                    print(f"  Warning: {e}")

        # Walk to find model.pt (hf_hub_download may nest subdirs)
        model_pt = None
        for root, _, fnames in os.walk(dest):
            if "model.pt" in fnames:
                model_pt = os.path.join(root, "model.pt")
                break

        if not model_pt:
            continue

        loss = _checkpoint_loss(model_pt)
        if loss is not None and loss > 4.0:
            print(f"  Skipping {best} — loss={loss:.2f} (bad checkpoint)")
            continue

        loss_str = f"loss={loss:.4f}" if loss else "loss=unknown"
        print(f"  Using: {best}  ({loss_str})")
        return model_pt

    print("  No valid checkpoint on Hub.")
    return None


def push_checkpoint(token, ckpt_dir, stage_id, phase_id, sub_id, step):
    from huggingface_hub import HfApi
    tag = f"curriculum-s{stage_id}-p{phase_id}{sub_id}-step{step:05d}"
    print(f"  Pushing {tag} to Hub...")
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
                print(f"    Warning: {fname}: {e}")
    print(f"  Pushed {pushed} files as: {tag}")
    return tag


def backup_to_drive(ckpt_dir, tag):
    if not DRIVE_DIR or not os.path.exists(os.path.dirname(DRIVE_DIR)):
        return
    os.makedirs(DRIVE_DIR, exist_ok=True)
    dest = os.path.join(DRIVE_DIR, tag)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    try:
        shutil.copytree(ckpt_dir, dest)
        print(f"  Drive backup: {tag}")
    except Exception as e:
        print(f"  Drive backup failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def ensure_data(phase_id, sub_id, data_file_rel, min_examples=1000):
    """
    Ensure data file exists with >= min_examples.
    Runs the generator if missing or too small.
    Returns absolute path, or None if cannot be created.
    """
    data_path = os.path.join(ROOT, "data/sft/curriculum", data_file_rel)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if os.path.exists(data_path):
        count = sum(1 for _ in open(data_path, encoding="utf-8", errors="replace"))
        if count >= min_examples:
            print(f"  Data: {data_file_rel}  ({count:,} examples) ✓")
            return data_path
        print(f"  Data: {data_file_rel}  only {count} examples < {min_examples} — regenerating...")

    gen_script = os.path.join(ROOT, "scripts/generate_phase_data.py")
    if not os.path.exists(gen_script):
        # No generator yet — if file exists with any data, use it
        if os.path.exists(data_path):
            print(f"  Warning: generator not found, using existing {data_path}")
            return data_path
        print(f"  ERROR: No data and no generator for {phase_id}{sub_id}")
        return None

    print(f"  Generating data for {phase_id}{sub_id}...")
    result = subprocess.run(
        [sys.executable, gen_script,
         "--phase", str(phase_id), "--sub", sub_id,
         "--output", data_path, "--min-examples", str(min_examples)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print(f"  Generator failed:\n{result.stderr[-400:]}")
        return os.path.join(data_path) if os.path.exists(data_path) else None

    if os.path.exists(data_path):
        count = sum(1 for _ in open(data_path, encoding="utf-8", errors="replace"))
        print(f"  Generated {count:,} examples for {phase_id}{sub_id} ✓")
        return data_path
    return None


def build_replay_mix(current_data_path, prior_data_paths, phase_id, sub_id,
                     replay_ratio=0.20):
    """
    Returns a path to a training file that is:
      80% current sub-phase data + 20% uniform sample from all prior data.
    If no prior data yet, returns the current data path unchanged.
    """
    if not prior_data_paths:
        return current_data_path

    # Load current
    current = []
    with open(current_data_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                current.append(line)

    # Load all prior
    all_prior = []
    for path in prior_data_paths:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_prior.append(line)

    if not all_prior:
        return current_data_path

    n_replay = min(int(len(current) * replay_ratio / (1.0 - replay_ratio)), len(all_prior))
    sampled = random.sample(all_prior, n_replay)
    mixed = current + sampled
    random.shuffle(mixed)

    mixed_path = os.path.join(
        os.path.dirname(current_data_path),
        f"_mixed_{phase_id}{sub_id}.jsonl"
    )
    with open(mixed_path, "w", encoding="utf-8") as f:
        for line in mixed:
            f.write(line + "\n")

    print(f"  Training mix: {len(current):,} current + {n_replay:,} replay = {len(mixed):,} total")
    return mixed_path


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(output_dir):
    """Return (model_pt_path, ckpt_dir) for the latest checkpoint in output_dir."""
    dirs = sorted(
        [d for d in glob.glob(f"{output_dir}/checkpoint-*") if os.path.isdir(d)],
        key=os.path.getmtime, reverse=True
    )
    for d in dirs:
        mp = os.path.join(d, "model.pt")
        if os.path.exists(mp):
            return mp, d
    return None, None


def read_step(ckpt_dir):
    meta = os.path.join(ckpt_dir, "metadata.json")
    if os.path.exists(meta):
        try:
            return json.load(open(meta)).get("step", 0)
        except Exception:
            pass
    for part in reversed(os.path.basename(ckpt_dir).split("-")):
        if part.isdigit():
            return int(part)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE SUB-PHASE
# ══════════════════════════════════════════════════════════════════════════════

def train_subphase(entry, start_ckpt, batch, grad_accum, precision_flag,
                   prior_data_paths):
    """
    Train a single sub-phase. Returns (model_pt, ckpt_dir) or (None, None).
    start_ckpt: path to model.pt to start from.
    prior_data_paths: list of data file paths from all previously completed sub-phases.
    """
    stage_id, phase_id, sub_id, name, data_file_rel, steps, lr, is_dpo = entry
    MIN_EX = 2000 if is_dpo else 1000

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stage {stage_id}  ·  Phase {phase_id}{sub_id}  ·  {name}")
    print(f"  Steps: {steps}   LR: {lr:.2e}   Batch: {batch}×{grad_accum}={batch*grad_accum}  {precision_flag or 'fp32'}")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = ensure_data(phase_id, sub_id, data_file_rel, MIN_EX)
    if not data_path:
        print(f"  SKIP: Cannot get training data for {phase_id}{sub_id}")
        return None, None

    train_file = build_replay_mix(data_path, prior_data_paths, phase_id, sub_id) \
                 if not is_dpo else data_path

    # ── Output dir ────────────────────────────────────────────────────────────
    output_dir = os.path.join(CKPT_BASE, "yaya-125m-curriculum",
                              f"s{stage_id}-p{phase_id}{sub_id}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Command ───────────────────────────────────────────────────────────────
    if is_dpo:
        cmd = [
            sys.executable, os.path.join(ROOT, "scripts/train_dpo.py"),
            "--model_config",       os.path.join(ROOT, "configs/model/yaya_125m.yaml"),
            "--pretrain_checkpoint", start_ckpt,
            "--dpo_data",           train_file,
            "--save_dir",           output_dir,
            "--steps",              str(steps),
            "--lr",                 str(lr),
        ]
    else:
        cmd = [
            sys.executable, os.path.join(ROOT, "scripts/train_sft.py"),
            "--model_config",       os.path.join(ROOT, "configs/model/yaya_125m.yaml"),
            "--train_config",       os.path.join(ROOT, "configs/training/sft_125m.yaml"),
            "--pretrain_checkpoint", start_ckpt,
            "--data_file",          train_file,
            "--output_dir",         output_dir,
            "--max_steps",          str(steps),
            "--learning_rate",      str(lr),
            "--max_seq_length",     "512",
            "--save_steps",         "500",
            "--warmup_steps",       str(max(50, steps // 20)),
            "--lr_scheduler",       "cosine",
            "--weight_decay",       "0.01",
            "--max_grad_norm",      "1.0",
            "--dataloader_num_workers", "2",
            "--per_device_batch_size",        str(batch),
            "--gradient_accumulation_steps",  str(grad_accum),
        ]
        if precision_flag:
            cmd.append(precision_flag)

    # ── OOM-safe training loop ────────────────────────────────────────────────
    for attempt, (bs, ga) in enumerate([(batch, grad_accum),
                                         (max(1, batch // 2), grad_accum * 2),
                                         (1, batch * grad_accum)]):
        if attempt > 0:
            print(f"\n  OOM retry {attempt}: batch={bs}  accum={ga}")
            clear_memory()
            if not is_dpo:
                for flag, val in [("--per_device_batch_size", str(bs)),
                                   ("--gradient_accumulation_steps", str(ga))]:
                    if flag in cmd:
                        cmd[cmd.index(flag) + 1] = val

        t0 = time.time()
        proc = run_subprocess(cmd, ROOT, timeout_sec=6*3600)
        elapsed = time.time() - t0
        rc = proc.returncode if proc else -1
        print(f"\n  Done in {elapsed/60:.1f} min  (exit {rc})")
        clear_memory()

        model_pt, ckpt_dir = find_latest_checkpoint(output_dir)
        if model_pt:
            return model_pt, ckpt_dir
        if rc == 0:
            break

    print(f"  ERROR: No checkpoint produced for {phase_id}{sub_id}")
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(ckpt_path, label=""):
    ckpt_dir = os.path.dirname(ckpt_path) if ckpt_path.endswith(".pt") else ckpt_path
    print(f"\n  Running benchmark{' — ' + label if label else ''}...")
    cmd = [sys.executable, os.path.join(ROOT, "scripts/benchmark.py"),
           "--checkpoint", ckpt_dir, "--dual"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    in_table = False
    for line in result.stdout.split("\n"):
        if any(k in line for k in ["Yaya Benchmark", "OVERALL", "====",
                                    "Guard lift", "DUAL", "Results saved"]):
            in_table = True
        if in_table:
            print("   ", line)
        if "Results saved" in line:
            break
    if result.returncode != 0 and result.stderr:
        print("  BENCHMARK ERROR:", result.stderr[-300:])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Yaya stage-based trainer. One stage per Colab session."
    )
    parser.add_argument("--stage",     type=int,  default=None,
                        help="Stage to train (1–7). Runs every phase in the stage.")
    parser.add_argument("--phase",     type=int,  default=None,
                        help="Limit to a specific phase number within the stage.")
    parser.add_argument("--subphase",  type=str,  default=None,
                        help="Limit to a specific sub-phase letter (a/b/c/d).")
    parser.add_argument("--resume",    action="store_true",
                        help="Auto-detect where we left off and continue from there.")
    parser.add_argument("--status",    action="store_true",
                        help="Print progress and exit.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark current best checkpoint and exit.")
    parser.add_argument("--no-push",   action="store_true")
    parser.add_argument("--no-drive",  action="store_true")
    parser.add_argument("--token",     type=str, default="")
    args = parser.parse_args()

    # ── Auth ──────────────────────────────────────────────────────────────────
    hf_token = get_hf_token(args.token)
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    else:
        print("  WARNING: No HF_TOKEN — Hub push/pull disabled.")

    completed = load_progress()

    # ── Status / Benchmark shortcuts ─────────────────────────────────────────
    if args.status:
        print_status(completed)
        return

    if args.benchmark:
        ckpt = pull_best_checkpoint(hf_token, 99) if hf_token else None
        if ckpt:
            run_benchmark(ckpt, "current best checkpoint")
        return

    # ── Resolve stage ─────────────────────────────────────────────────────────
    if args.resume:
        remaining = [e for e in CURRICULUM if sp_key(e[1], e[2]) not in completed]
        if not remaining:
            print("  All stages complete!")
            print_status(completed)
            return
        args.stage = remaining[0][0]
        print(f"  Resuming at Stage {args.stage}: {STAGE_NAMES[args.stage]}")

    if args.stage is None:
        parser.print_help()
        print("\n  Specify --stage N  (1–7), or use --resume / --status\n")
        sys.exit(1)

    if args.stage not in STAGE_NAMES:
        print(f"  Invalid stage {args.stage}. Valid: 1–7")
        sys.exit(1)

    # ── Build work list ───────────────────────────────────────────────────────
    work = [e for e in CURRICULUM if e[0] == args.stage]
    if args.phase:
        work = [e for e in work if e[1] == args.phase]
    if args.subphase:
        work = [e for e in work if e[2] == args.subphase.lower()]
    work = [e for e in work if sp_key(e[1], e[2]) not in completed]

    if not work:
        print(f"  Stage {args.stage} is already complete.")
        print_status(completed)
        return

    # ── Print plan ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Yaya — Stage {args.stage}: {STAGE_NAMES[args.stage]}")
    print("=" * 60)
    total_steps = sum(e[5] for e in work)
    print(f"  Sub-phases: {len(work)}   Total steps: {total_steps:,}")
    for s, p, sub, name, _, steps, lr, _ in work:
        print(f"    {p}{sub}. {name:<32} {steps:4d} steps  lr={lr:.1e}")
    print()

    # ── GPU ───────────────────────────────────────────────────────────────────
    gpu_name, vram, batch, grad_accum, precision_flag = detect_gpu()
    print(f"  GPU: {gpu_name} ({vram:.1f} GB)  batch={batch}×{grad_accum}  {precision_flag or 'fp32'}\n")

    # ── Starting checkpoint ───────────────────────────────────────────────────
    start_ckpt = pull_best_checkpoint(hf_token, args.stage) if hf_token else None

    if not start_ckpt:
        # Fall back to local
        pts = sorted(glob.glob(f"{CKPT_BASE}/**/*.pt", recursive=True),
                     key=os.path.getmtime, reverse=True)
        for pt in pts:
            loss = _checkpoint_loss(pt)
            if loss is None or loss < 4.0:
                start_ckpt = pt
                print(f"  Using local checkpoint: {pt}")
                break

    if not start_ckpt:
        if args.stage == 1:
            print("  No checkpoint found — Stage 1 will start from random init.")
            print("  (This is OK only for Stage 1. For later stages, get a checkpoint.)")
        else:
            print(f"  ERROR: No valid checkpoint for Stage {args.stage}.")
            print(f"  Stages 2–7 must start from the output of the previous stage.")
            print(f"  Run Stage {args.stage - 1} first, or set HF_TOKEN to pull from Hub.")
            sys.exit(1)

    # ── Prior data for replay ─────────────────────────────────────────────────
    # All COMPLETED sub-phases that come before our first work item in curriculum order
    first_idx = CURRICULUM.index(work[0])
    prior_data = []
    for e in CURRICULUM[:first_idx]:
        if sp_key(e[1], e[2]) in completed and not e[7]:   # exclude DPO from replay
            dp = os.path.join(ROOT, "data/sft/curriculum", e[4])
            if os.path.exists(dp):
                prior_data.append(dp)

    # ── Run sub-phases ────────────────────────────────────────────────────────
    for entry in work:
        s, p, sub, name, data_file_rel, steps, lr, is_dpo = entry
        key = sp_key(p, sub)

        model_pt, ckpt_dir = train_subphase(
            entry, start_ckpt, batch, grad_accum, precision_flag, prior_data
        )

        if model_pt is None:
            print(f"\n  FAILED: {p}{sub} — {name}")
            print(f"  Fix the issue and re-run:")
            print(f"    !python scripts/colab_run_phases.py --stage {s} --phase {p} --subphase {sub}")
            sys.exit(1)

        actual_step = read_step(ckpt_dir)

        # Push + backup
        if hf_token and not args.no_push:
            tag = push_checkpoint(hf_token, ckpt_dir, s, p, sub, actual_step)
            if not args.no_drive:
                backup_to_drive(ckpt_dir, tag)

        # Record progress
        completed.add(key)
        save_progress(completed)

        # This checkpoint is the start for the next sub-phase
        start_ckpt = model_pt

        # Add this data to the replay pool (SFT only)
        if not is_dpo:
            dp = os.path.join(ROOT, "data/sft/curriculum", data_file_rel)
            if os.path.exists(dp):
                prior_data.append(dp)

        print(f"\n  ✓ {p}{sub} complete — {name}")

    # ── End of stage: benchmark ───────────────────────────────────────────────
    if start_ckpt:
        run_benchmark(start_ckpt, f"after Stage {args.stage}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Stage {args.stage} complete — {STAGE_NAMES[args.stage]}")
    print_status(completed)

    next_stage = args.stage + 1
    if next_stage <= 7:
        print(f"  Next session:")
        print(f"    !python scripts/colab_run_phases.py --stage {next_stage}")
        print(f"  ({STAGE_NAMES[next_stage]})")
    else:
        print("  ALL STAGES COMPLETE — Yaya is a true AI.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
