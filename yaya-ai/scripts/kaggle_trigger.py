"""Trigger Kaggle notebook to resume Yaya training.

Checks GPU quota, waits if needed, then triggers the notebook to run.
Can be scheduled as a cron job or run manually.

Usage:
    # Run once — trigger now if quota available
    python scripts/kaggle_trigger.py

    # Watch mode — poll until quota resets, then trigger
    python scripts/kaggle_trigger.py --watch

    # Check status only
    python scripts/kaggle_trigger.py --status
"""

import sys
import os
import time
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NOTEBOOK_REF = "jaylinkcoder/notebookfe483550f6"
HF_TOKEN = "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"
POLL_INTERVAL = 600  # 10 min between checks in watch mode


def get_api():
    import kaggle
    kaggle.api.authenticate()
    return kaggle.api


def get_notebook_status(api):
    try:
        status = api.kernels_status(NOTEBOOK_REF)
        return status
    except Exception as e:
        return None


def get_training_step():
    """Get current training step from HF Hub."""
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        files = list(list_repo_files(repo_id="Jaylink-coder/yaya-125m", repo_type="model", token=HF_TOKEN))
        ckpts = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-") and "_temp" not in f})
        if not ckpts:
            return 0, None
        latest = ckpts[-1]
        p = hf_hub_download(
            repo_id="Jaylink-coder/yaya-125m", filename=f"{latest}/metadata.json",
            repo_type="model", token=HF_TOKEN, local_dir="/tmp/kt_meta", force_download=True,
        )
        with open(p) as f:
            meta = json.load(f)
        return meta.get("step", 0), meta.get("loss", 0)
    except Exception:
        return 0, None


def trigger_notebook(api):
    """Push a new run of the notebook."""
    try:
        api.kernels_push(NOTEBOOK_REF)
        return True
    except Exception as e:
        # kernels_push may not exist in all versions — try pull instead
        try:
            result = api.kernel_pull(NOTEBOOK_REF, path="/tmp/yaya_kernel")
            return True
        except Exception as e2:
            print(f"  Trigger failed: {e} / {e2}")
            return False


def print_status(api):
    step, loss = get_training_step()
    pct = step / 40000 * 100
    bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))

    print(f"\n{'='*55}")
    print(f"  Yaya Training Status  ({time.strftime('%H:%M:%S')})")
    print(f"{'='*55}")
    print(f"  Step:  {step:,} / 40,000  ({pct:.1f}%)")
    if loss:
        print(f"  Loss:  {loss:.4f}")
    print(f"  [{bar}]")

    nb_status = get_notebook_status(api)
    if nb_status:
        print(f"\n  Kaggle notebook: {nb_status}")
    print(f"{'='*55}\n")

    return step, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch",  action="store_true", help="Watch and auto-trigger when quota resets")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--force",  action="store_true", help="Trigger notebook without checks")
    args = parser.parse_args()

    api = get_api()

    if args.status:
        print_status(api)
        return

    if args.force:
        print("Force-triggering notebook...")
        step, loss = get_training_step()
        print(f"  Current step: {step:,}  Loss: {loss}")
        if trigger_notebook(api):
            print("  Notebook triggered successfully.")
        else:
            print("  Trigger failed — run the notebook manually on kaggle.com")
        return

    # Default / watch mode
    while True:
        step, loss = print_status(api)[0], None

        if step >= 40000:
            print("Training complete! Step 40,000 reached.")
            print("DPO should have auto-launched. Check HF Hub for final checkpoint.")
            break

        nb_status = get_notebook_status(api)
        nb_running = nb_status and str(nb_status).lower() in ("running", "queued")

        if nb_running:
            print(f"Notebook already running. Current step: {step:,}")
        else:
            print(f"Notebook not running. Attempting to trigger...")
            if trigger_notebook(api):
                print("Triggered! Training will resume shortly.")
            else:
                print("Could not auto-trigger. Open kaggle.com and run the notebook manually.")

        if not args.watch:
            break

        print(f"Next check in {POLL_INTERVAL//60} min...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
