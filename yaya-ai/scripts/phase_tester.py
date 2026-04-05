"""Phase auto-tester — watches HF Hub and tests Yaya after each completed phase.

Usage:
    # One-shot: test latest checkpoint now
    python scripts/phase_tester.py --once

    # Watch mode: poll every 5 min, test when new phase checkpoint appears
    python scripts/phase_tester.py --watch

    # Test a specific checkpoint
    python scripts/phase_tester.py --checkpoint checkpoints/yaya-125m-sft/checkpoint-00015000
"""

import sys
import os
import time
import json
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN
from src.inference.generator import TextGenerator, GenerationConfig

HF_REPO = "Jaylink-coder/yaya-125m"
PHASE_STEPS = 2500  # steps per phase
TOTAL_STEPS = 40000
LOCAL_CKPT_DIR = "checkpoints/yaya-125m-sft"
RESULTS_FILE = "docs/phase_test_results.jsonl"

PHASE_NAMES = {
    1: "First Words", 2: "Warm-up", 3: "Basic Answers", 4: "Arithmetic",
    5: "Foundation I", 6: "Foundation II", 7: "Word Problems", 8: "Reasoning I",
    9: "Reasoning II", 10: "Reasoning III", 11: "Deep Math I", 12: "Deep Math II",
    13: "Deep Math III", 14: "Advanced I", 15: "Advanced II", 16: "Advanced III",
    17: "DPO",
}

TEST_QUESTIONS = [
    ("2 + 2",          "What is 2 + 2?",                           "4"),
    ("12 x 12",        "What is 12 x 12?",                         "144"),
    ("10 - 4",         "What is 10 - 4?",                          "6"),
    ("15% of 200",     "What is 15% of 200?",                      "30"),
    ("7 x 8",          "What is 7 times 8?",                       "56"),
    ("apples",         "If I have 10 apples and give away 3, how many remain?", "7"),
    ("france capital", "What is the capital of France?",           "paris"),
    ("sky color",      "What color is the sky?",                   "blue"),
    ("yaya identity",  "Who are you?",                             "yaya"),
    ("12 x 10",        "What is 12 x 10?",                         "120"),
]

SYSTEM = "You are Yaya, a helpful, honest, and friendly AI assistant. You answer questions clearly and thoughtfully."


def get_latest_phase_checkpoint(token):
    """Return the latest checkpoint that falls on a phase boundary (multiple of PHASE_STEPS)."""
    files = list(list_repo_files(repo_id=HF_REPO, repo_type="model", token=token))
    ckpts = sorted({
        f.split("/")[0] for f in files
        if f.startswith("checkpoint-") and "_temp" not in f
    })
    # Only phase-boundary checkpoints
    phase_ckpts = [c for c in ckpts if int(c.split("-")[1]) % PHASE_STEPS == 0]
    return phase_ckpts[-1] if phase_ckpts else None


def download_checkpoint(ckpt_name, token):
    """Download model.pt for a checkpoint from HF Hub."""
    local_dir = os.path.join(LOCAL_CKPT_DIR, ckpt_name)
    model_path = os.path.join(local_dir, "model.pt")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 100_000_000:
        print(f"  Already have {ckpt_name}/model.pt locally.")
        return model_path
    os.makedirs(local_dir, exist_ok=True)
    print(f"  Downloading {ckpt_name}/model.pt from HF Hub...")
    hf_hub_download(
        repo_id=HF_REPO, filename=f"{ckpt_name}/model.pt",
        repo_type="model", token=token, local_dir=LOCAL_CKPT_DIR,
    )
    # Also grab metadata
    try:
        hf_hub_download(
            repo_id=HF_REPO, filename=f"{ckpt_name}/metadata.json",
            repo_type="model", token=token, local_dir=LOCAL_CKPT_DIR,
        )
    except Exception:
        pass
    return model_path


def load_model(model_path):
    model_config = load_model_config("configs/model/yaya_125m.yaml")
    model = YayaForCausalLM(model_config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def run_tests(model, tokenizer):
    gen = TextGenerator(model, tokenizer, device="cpu")
    cfg = GenerationConfig(
        max_new_tokens=60, temperature=0.7, top_p=0.9,
        do_sample=True, repetition_penalty=1.5,
    )
    results = []
    passed = 0
    for label, question, expected in TEST_QUESTIONS:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        prompt = tokenizer.format_chat(messages) + "\n" + ASSISTANT_TOKEN + "\n"
        answer = gen.generate(prompt, config=cfg).strip()
        correct = expected.lower() in answer.lower()
        if correct:
            passed += 1
        results.append({
            "label": label,
            "question": question,
            "expected": expected,
            "answer": answer,
            "correct": correct,
        })
        mark = "PASS" if correct else "FAIL"
        print(f"  [{mark}] {question}")
        print(f"         → {answer[:80]!r}")
    return results, passed


def save_results(ckpt_name, step, loss, phase, results, passed):
    os.makedirs("docs", exist_ok=True)
    record = {
        "checkpoint": ckpt_name,
        "step": step,
        "loss": loss,
        "phase": phase,
        "phase_name": PHASE_NAMES.get(phase, ""),
        "score": f"{passed}/{len(results)}",
        "pct": round(passed / len(results) * 100, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n  Results saved → {RESULTS_FILE}")
    return record


def test_checkpoint(ckpt_name, tokenizer, token=None, step=None, loss=None):
    phase = (int(ckpt_name.split("-")[1]) // PHASE_STEPS)
    print(f"\n{'='*60}")
    print(f"  Phase {phase}/17 — {PHASE_NAMES.get(phase, '')}  ({ckpt_name})")
    print(f"  Step: {step}  Loss: {loss:.4f}" if loss else f"  Step: {step}")
    print(f"{'='*60}")

    if token:
        model_path = download_checkpoint(ckpt_name, token)
    else:
        model_path = os.path.join(LOCAL_CKPT_DIR, ckpt_name, "model.pt")

    print("  Loading model...")
    model = load_model(model_path)
    print("  Running tests...")
    results, passed = run_tests(model, tokenizer)

    pct = passed / len(results) * 100
    print(f"\n  Score: {passed}/{len(results)} ({pct:.0f}%)")
    if loss is not None:
        record = save_results(ckpt_name, step, loss, phase, results, passed)
    print(f"{'='*60}\n")
    return passed, len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",       action="store_true", help="Test latest checkpoint once")
    parser.add_argument("--watch",      action="store_true", help="Watch for new phase checkpoints")
    parser.add_argument("--checkpoint", type=str,            help="Test a specific local checkpoint path")
    parser.add_argument("--token",      type=str,            default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--interval",   type=int,            default=300, help="Poll interval seconds (watch mode)")
    args = parser.parse_args()

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")

    if args.checkpoint:
        # Test a local checkpoint directly
        ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
        test_checkpoint(ckpt_name, tokenizer, token=None)
        return

    if not args.token:
        print("Error: set HF_TOKEN env var or pass --token")
        sys.exit(1)

    if args.once or args.watch:
        tested = set()

        while True:
            try:
                latest = get_latest_phase_checkpoint(args.token)
                if latest and latest not in tested:
                    # Get metadata
                    try:
                        p = hf_hub_download(
                            repo_id=HF_REPO, filename=f"{latest}/metadata.json",
                            repo_type="model", token=args.token,
                            local_dir=f"/tmp/phase_meta_{latest}", force_download=True,
                        )
                        with open(p) as f:
                            meta = json.load(f)
                        step = meta.get("step", 0)
                        loss = meta.get("loss", 0.0)
                    except Exception:
                        step = int(latest.split("-")[1])
                        loss = 0.0

                    test_checkpoint(latest, tokenizer, token=args.token, step=step, loss=loss)
                    tested.add(latest)
                else:
                    files = list(list_repo_files(repo_id=HF_REPO, repo_type="model", token=args.token))
                    all_ckpts = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-") and "_temp" not in f})
                    cur_step = int(all_ckpts[-1].split("-")[1]) if all_ckpts else 0
                    pct = cur_step / TOTAL_STEPS * 100
                    print(f"[{time.strftime('%H:%M:%S')}] No new phase checkpoint. Current: step {cur_step:,} ({pct:.1f}%)  Latest tested: {latest or 'none'}")

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")

            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
