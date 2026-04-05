"""Live training monitor — polls HF Hub every 60s and prints step/loss.

Usage:
    python scripts/monitor.py              # reads HF_TOKEN from .env or env
    python scripts/monitor.py --token hf_xxx
    python scripts/monitor.py --interval 30   # poll every 30s
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB_REPO  = "Jaylink-coder/yaya-125m"


def load_token():
    # 1. env var
    t = os.environ.get("HF_TOKEN", "")
    if t:
        return t
    # 2. .env file
    env_file = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    t = line.split("=", 1)[1].strip()
                    if t:
                        return t
    # 3. HF cache
    cache = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache):
        with open(cache) as f:
            t = f.read().strip()
        if t:
            return t
    return ""


def get_hub_status(token):
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        # get latest pointer
        try:
            lp = hf_hub_download(
                repo_id=HUB_REPO, filename="latest.json",
                repo_type="model", token=token, local_dir="/tmp/yaya_monitor",
                force_download=True,
            )
            with open(lp) as f:
                latest_name = json.load(f)["latest"]
        except Exception:
            files = list(list_repo_files(repo_id=HUB_REPO, repo_type="model", token=token))
            ckpts = sorted({f.split("/")[0] for f in files
                           if f.split("/")[0].startswith(("checkpoint-", "recovery-"))})
            if not ckpts:
                return None
            latest_name = ckpts[-1]

        # fetch metadata
        try:
            mp = hf_hub_download(
                repo_id=HUB_REPO, filename=f"{latest_name}/metadata.json",
                repo_type="model", token=token, local_dir="/tmp/yaya_monitor",
                force_download=True,
            )
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        return {"ckpt": latest_name, "step": meta.get("step"), "loss": meta.get("loss")}
    except Exception as e:
        return {"error": str(e)}


def bar(pct, width=20):
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",    default="")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once",     action="store_true")
    args = parser.parse_args()

    token = args.token or load_token()
    if not token:
        print("No HF_TOKEN found. Add it to .env or pass --token hf_xxx")
        print(f"  Edit: {os.path.join(REPO_ROOT, '.env')}")
        sys.exit(1)

    print(f"Monitoring: {HUB_REPO}")
    print(f"Polling every {args.interval}s  (Ctrl+C to stop)")
    print("-" * 50)

    prev_step = None
    while True:
        status = get_hub_status(token)
        ts = time.strftime("%H:%M:%S")

        if status is None:
            print(f"[{ts}] No checkpoints on Hub yet...")
        elif "error" in status:
            print(f"[{ts}] Hub error: {status['error']}")
        else:
            ckpt = status["ckpt"]
            step = status["step"]
            loss = status["loss"]

            # Determine phase
            is_recovery = ckpt.startswith("recovery-")
            is_dpo      = "dpo" in ckpt.lower() and not is_recovery
            phase_str   = "Recovery" if is_recovery else ("DPO" if is_dpo else "SFT")

            if step is not None:
                if is_recovery:
                    pct = min(step / 3000 * 100, 100)
                    prog = f"Step {step}/3000 {bar(pct)} {pct:.0f}%"
                elif is_dpo:
                    pct = min(step / 2500 * 100, 100)
                    prog = f"Step {step}/2500 {bar(pct)} {pct:.0f}%"
                else:
                    pct = min(step / 40000 * 100, 100)
                    prog = f"Step {step}/40000 {bar(pct)} {pct:.0f}%"
            else:
                prog = ckpt

            loss_str = f"Loss: {loss:.4f}" if loss is not None else "Loss: ?"
            delta = ""
            if prev_step is not None and step is not None and step != prev_step:
                delta = f" (+{step - prev_step} steps)"
            prev_step = step

            print(f"[{ts}] [{phase_str}] {prog}  {loss_str}{delta}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
