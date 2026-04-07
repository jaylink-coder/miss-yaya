"""Live training monitor for Yaya curriculum.

Polls HF Hub every 60 seconds to track checkpoint progress,
phase completion, and training status.

Usage:
    python scripts/monitor_training.py
    python scripts/monitor_training.py --interval 30   # poll every 30s
    python scripts/monitor_training.py --once           # single check
"""

import json
import os
import sys
import time
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, '.env')

# Load tokens from .env
def load_env():
    tokens = {}
    if os.path.isfile(ENV_PATH):
        with open(ENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    tokens[key.strip()] = val.strip()
    return tokens

env = load_env()
HF_TOKEN = env.get('HF_TOKEN') or os.environ.get('HF_TOKEN')
HUB_REPO = 'Jaylink-coder/yaya-125m'

if not HF_TOKEN:
    print("ERROR: No HF_TOKEN found. Run: python scripts/setup_tokens.py")
    sys.exit(1)

try:
    from huggingface_hub import HfApi, list_repo_files, hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'])
    from huggingface_hub import HfApi, list_repo_files, hf_hub_download

api = HfApi(token=HF_TOKEN)

# Phase info for display
PHASE_NAMES = {
    1: "World Knowledge", 2: "Conversational Fluency", 3: "Instruction Following",
    4: "Direct Q&A", 5: "Chain-of-Thought", 6: "Math Reasoning",
    7: "Logical Reasoning", 8: "Self-Reflection", 9: "Tool Calling Basics",
    10: "Multi-Step Tools", 11: "RAG Grounding", 12: "Code Understanding",
    13: "Structured Output", 14: "Kenya & Swahili", 15: "Safety & Refusals",
    16: "DPO Alignment",
}

def get_hub_status():
    """Check HF Hub for training progress."""
    status = {
        'checkpoints': [],
        'latest': None,
        'progress': None,
        'last_update': None,
    }

    try:
        files = list(list_repo_files(repo_id=HUB_REPO, repo_type='model', token=HF_TOKEN))
    except Exception as e:
        return {'error': str(e)}

    # Find checkpoints
    ckpt_names = sorted({
        f.split('/')[0] for f in files
        if f.split('/')[0].startswith('checkpoint-') and '_temp' not in f
    })
    status['checkpoints'] = ckpt_names

    # Get latest pointer
    try:
        latest_path = hf_hub_download(
            repo_id=HUB_REPO, filename='latest.json',
            repo_type='model', token=HF_TOKEN,
            local_dir='/tmp/yaya_monitor', force_download=True
        )
        with open(latest_path) as f:
            status['latest'] = json.load(f).get('latest')
    except Exception:
        if ckpt_names:
            status['latest'] = ckpt_names[-1]

    # Get curriculum progress
    try:
        prog_path = hf_hub_download(
            repo_id=HUB_REPO, filename='curriculum_progress.json',
            repo_type='model', token=HF_TOKEN,
            local_dir='/tmp/yaya_monitor', force_download=True
        )
        with open(prog_path) as f:
            status['progress'] = json.load(f)
    except Exception:
        pass

    # Get repo last modified
    try:
        info = api.repo_info(repo_id=HUB_REPO, repo_type='model', token=HF_TOKEN)
        status['last_update'] = str(info.last_modified) if info.last_modified else None
    except Exception:
        pass

    return status

def extract_step(ckpt_name):
    m = re.search(r'checkpoint-0*(\d+)', ckpt_name)
    return int(m.group(1)) if m else 0

def display_status(status, clear=True):
    """Pretty-print training status."""
    if clear and os.name != 'nt':
        os.system('clear')
    elif clear:
        os.system('cls')

    now = datetime.now().strftime('%H:%M:%S')

    print()
    print("=" * 60)
    print(f"  YAYA TRUE AI — TRAINING MONITOR  [{now}]")
    print("=" * 60)

    if 'error' in status:
        print(f"\n  ERROR: {status['error']}")
        return

    # Hub info
    print(f"\n  Hub: {HUB_REPO}")
    if status.get('last_update'):
        print(f"  Last update: {status['last_update']}")

    # Latest checkpoint
    latest = status.get('latest', 'none')
    step = extract_step(latest) if latest else 0
    print(f"\n  Latest checkpoint: {latest}")
    print(f"  Global step: {step:,}")

    # Curriculum progress
    progress = status.get('progress')
    if progress:
        completed = progress.get('completed_phases', [])
        n_done = len(completed)

        print(f"\n  Curriculum: {n_done}/16 phases complete")
        print(f"  {'=' * 40}")

        # Phase progress bar
        for pid in range(1, 17):
            name = PHASE_NAMES.get(pid, f"Phase {pid}")
            if pid in completed:
                marker = " DONE"
                icon = "[##########]"
            elif pid == progress.get('current_phase'):
                marker = " << TRAINING"
                icon = "[####......] "
            else:
                marker = ""
                icon = "[..........]"
            print(f"    {pid:2d}. {icon} {name}{marker}")

        # History
        history = progress.get('history', [])
        if history:
            print(f"\n  Recent activity:")
            for h in history[-5:]:
                ts = h.get('completed_at', '?')[:19]
                print(f"    {ts} — Phase {h['phase_id']}: {h['phase_name']} [{h['status']}]")
    else:
        print("\n  No curriculum progress found yet.")
        print("  (Progress file appears after first phase completes)")

    # All checkpoints
    ckpts = status.get('checkpoints', [])
    if ckpts:
        print(f"\n  Checkpoints on Hub ({len(ckpts)}):")
        for c in ckpts[-10:]:  # last 10
            print(f"    {c}")

    print(f"\n{'=' * 60}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monitor Yaya curriculum training')
    parser.add_argument('--interval', type=int, default=60, help='Poll interval in seconds')
    parser.add_argument('--once', action='store_true', help='Check once and exit')
    args = parser.parse_args()

    # Verify token
    try:
        user = api.whoami()
        print(f"Authenticated as: {user.get('name', 'OK')}")
    except Exception as e:
        print(f"ERROR: HF token invalid: {e}")
        sys.exit(1)

    if args.once:
        status = get_hub_status()
        display_status(status, clear=False)
        return

    print(f"Monitoring {HUB_REPO} every {args.interval}s... (Ctrl+C to stop)\n")

    last_latest = None
    while True:
        try:
            status = get_hub_status()
            current_latest = status.get('latest')

            # Only refresh display if something changed, or first run
            if current_latest != last_latest or last_latest is None:
                display_status(status)
                if last_latest is not None and current_latest != last_latest:
                    print(f"\n  ** NEW CHECKPOINT: {current_latest} **")
                last_latest = current_latest
            else:
                # Just print a heartbeat
                now = datetime.now().strftime('%H:%M:%S')
                print(f"  [{now}] No change — latest: {current_latest}", end='\r')

            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            break
        except Exception as e:
            print(f"\n  Poll error: {e}")
            time.sleep(args.interval)

if __name__ == '__main__':
    main()
