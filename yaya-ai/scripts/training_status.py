"""Show current training status for Yaya.

Reads the training log and checkpoint directory to show progress.
"""

import re
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Auto-detect active log/checkpoint — 125M is the default model
_RUNS = [
    ("logs/sft_125m.log",      "checkpoints/yaya-125m-sft",           30000),
    ("logs/reasoning.log",     "checkpoints/yaya-125m-reasoning",      8000),
    ("logs/pretrain_125m.log", "checkpoints/yaya-125m",               20000),
    ("logs/math_combined.log", "checkpoints/yaya-tiny-math-combined", 3000),
]

import time as _time
def _pick_run():
    best = None
    best_mtime = 0
    for log, ckpt, steps in _RUNS:
        if os.path.exists(log):
            mtime = os.path.getmtime(log)
            if mtime > best_mtime:
                best_mtime = mtime
                best = (log, ckpt, steps)
    return best or _RUNS[0]

LOG_FILE, CKPT_DIR, MAX_STEPS = _pick_run()


def main():
    print("=" * 50)
    print("  Yaya Training Status")
    print("=" * 50)

    # Parse log for latest step
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
        steps = re.findall(r"Step\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\d.e+-]+)", data)
        if steps:
            step, loss, lr = max(steps, key=lambda x: int(x[0]))
            step = int(step)
            loss = float(loss)
            pct = 100 * step / MAX_STEPS
            remaining = MAX_STEPS - step
            mtime = os.path.getmtime(LOG_FILE)
            age = int(time.time() - mtime)
            print(f"\n  Log file: {LOG_FILE}")
            print(f"  Latest step:  {step:,} / {MAX_STEPS:,} ({pct:.1f}%)")
            print(f"  Loss:         {loss:.4f}")
            print(f"  LR:           {lr}")
            print(f"  Remaining:    {remaining:,} steps")
            print(f"  Log updated:  {age}s ago")
        else:
            print(f"\n  Log exists but no steps logged yet.")
            mtime = os.path.getmtime(LOG_FILE)
            age = int(time.time() - mtime)
            print(f"  Log updated: {age}s ago (startup in progress?)")
    else:
        print(f"\n  No log file at {LOG_FILE}")

    # Show checkpoints
    print(f"\n  Checkpoints in {CKPT_DIR}:")
    if os.path.isdir(CKPT_DIR):
        entries = [e for e in os.listdir(CKPT_DIR) if e.startswith("checkpoint-")]
        entries.sort()
        for e in entries:
            path = os.path.join(CKPT_DIR, e)
            meta = os.path.join(path, "metadata.json")
            step_info = ""
            if os.path.exists(meta):
                import json
                with open(meta) as f:
                    m = json.load(f)
                step_info = f"step={m.get('step', '?')}, loss={m.get('loss', '?'):.4f}"
            mtime = os.path.getmtime(path)
            age_min = int((time.time() - mtime) / 60)
            print(f"    {e}  ({step_info}, {age_min}m ago)")
        latest_file = os.path.join(CKPT_DIR, "latest")
        if os.path.exists(latest_file):
            with open(latest_file) as f:
                latest = f.read().strip()
            print(f"    latest -> {latest}")
    else:
        print(f"    (directory not found)")

    print()


if __name__ == "__main__":
    main()
