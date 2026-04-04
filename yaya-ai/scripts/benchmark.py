"""Comprehensive Yaya benchmark — tests across all capability areas.

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/yaya-125m-sft/checkpoint-00030000
    python scripts/benchmark.py --token hf_xxx  # auto-downloads latest from HF Hub

Outputs a summary table and saves to docs/benchmark_results.jsonl
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN
from src.inference.generator import TextGenerator, GenerationConfig

HF_REPO = "Jaylink-coder/yaya-125m"
RESULTS_FILE = "docs/benchmark_results.jsonl"
SYSTEM = "You are Yaya, a helpful, honest, and friendly AI assistant. You answer questions clearly and thoughtfully."

# ── Test suite ────────────────────────────────────────────────────────────────
SUITE = {
    "Basic Arithmetic": [
        ("2 + 2",     "What is 2 + 2?",          ["4"]),
        ("3 + 3",     "What is 3 + 3?",          ["6"]),
        ("10 - 4",    "What is 10 - 4?",         ["6"]),
        ("5 x 5",     "What is 5 x 5?",          ["25"]),
        ("8 x 9",     "What is 8 x 9?",          ["72"]),
        ("12 x 12",   "What is 12 x 12?",        ["144"]),
        ("100 / 4",   "What is 100 divided by 4?", ["25"]),
        ("sqrt(16)",  "What is the square root of 16?", ["4"]),
    ],
    "Word Problems": [
        ("apples",    "If I have 10 apples and give away 3, how many do I have left?", ["7"]),
        ("shirt",     "A shirt costs $15. I pay with $20. How much change do I get?", ["5", "$5"]),
        ("distance",  "A car travels at 60 km/h for 2 hours. How far does it go?", ["120"]),
        ("eggs",      "A dozen eggs costs $3. How much do 2 dozen cost?", ["6", "$6"]),
        ("percent",   "What is 15% of 200?", ["30"]),
        ("half",      "What is half of 84?", ["42"]),
    ],
    "Factual Knowledge": [
        ("france",    "What is the capital of France?",          ["paris"]),
        ("kenya",     "What is the capital of Kenya?",           ["nairobi"]),
        ("sky",       "What color is the sky?",                  ["blue"]),
        ("week",      "How many days are in a week?",            ["7"]),
        ("year",      "How many months are in a year?",          ["12"]),
        ("water",     "What is the chemical formula for water?", ["h2o", "h₂o"]),
        ("planet",    "What planet do we live on?",              ["earth"]),
        ("boiling",   "At what temperature does water boil?",    ["100", "212"]),
    ],
    "Identity": [
        ("name",      "What is your name?",                      ["yaya"]),
        ("who",       "Who are you?",                            ["yaya"]),
        ("not gpt",   "Are you ChatGPT?",                        ["no", "not"]),
        ("not human", "Are you a human?",                        ["no", "not", "ai", "assistant"]),
    ],
    "Reasoning": [
        ("logic",     "All dogs are animals. Rex is a dog. Is Rex an animal?", ["yes"]),
        ("lily pad",  "A lily pad doubles every day. After 48 days it covers the lake. When was it half covered?", ["47"]),
        ("bat ball",  "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?", ["0.05", "5 cent"]),
        ("prime",     "Is 17 a prime number?",                   ["yes"]),
        ("not prime", "Is 15 a prime number?",                   ["no"]),
    ],
    "Language": [
        ("opposite",  "What is the opposite of hot?",            ["cold"]),
        ("bigger",    "Which is bigger: the Sun or the Earth?",  ["sun"]),
        ("faster",    "Which is faster: a car or a bicycle?",    ["car"]),
        ("language",  "What language is this sentence written in?", ["english"]),
    ],
}


def load_model(checkpoint_path):
    model_config = load_model_config("configs/model/yaya_125m.yaml")
    model = YayaForCausalLM(model_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def run_benchmark(model, tokenizer, step=None, loss=None):
    gen = TextGenerator(model, tokenizer, device="cpu")
    cfg = GenerationConfig(
        max_new_tokens=60, temperature=0.7, top_p=0.9,
        do_sample=True, repetition_penalty=1.5,
    )

    all_results = {}
    total_pass = total_fail = 0

    for category, tests in SUITE.items():
        cat_pass = cat_fail = 0
        cat_results = []
        for label, question, accepted in tests:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
            prompt = tokenizer.format_chat(messages) + "\n" + ASSISTANT_TOKEN + "\n"
            answer = gen.generate(prompt, config=cfg).strip()
            correct = any(a.lower() in answer.lower() for a in accepted)
            if correct:
                cat_pass += 1
                total_pass += 1
            else:
                cat_fail += 1
                total_fail += 1
            cat_results.append({"label": label, "question": question, "answer": answer, "correct": correct})

        pct = cat_pass / len(tests) * 100
        all_results[category] = {"pass": cat_pass, "total": len(tests), "pct": pct, "results": cat_results}

    total = total_pass + total_fail
    overall_pct = total_pass / total * 100

    return all_results, total_pass, total, overall_pct


def print_report(all_results, total_pass, total, overall_pct, step=None, loss=None):
    print()
    print("=" * 55)
    if step:
        print(f"  Yaya Benchmark  (step {step:,}  loss {loss:.4f})" if loss else f"  Yaya Benchmark  (step {step:,})")
    else:
        print("  Yaya Benchmark")
    print("=" * 55)
    print(f"  {'Category':<22} {'Score':>8}  {'Bar'}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*20}")
    for cat, r in all_results.items():
        bar = "#" * r["pass"] + "." * (r["total"] - r["pass"])
        print(f"  {cat:<22} {r['pass']}/{r['total']} ({r['pct']:4.0f}%)  [{bar}]")
    print(f"  {'-'*22}  {'-'*8}")
    print(f"  {'OVERALL':<22} {total_pass}/{total} ({overall_pct:4.0f}%)")
    print("=" * 55)

    # Show failures for context
    print("\n  Failures:")
    any_fail = False
    for cat, r in all_results.items():
        for res in r["results"]:
            if not res["correct"]:
                any_fail = True
                print(f"  [{cat}] {res['question']}")
                print(f"    → {res['answer'][:80]!r}")
    if not any_fail:
        print("  None!")
    print()


def save_results(all_results, total_pass, total, overall_pct, ckpt_name, step, loss):
    os.makedirs("docs", exist_ok=True)
    record = {
        "checkpoint": ckpt_name,
        "step": step,
        "loss": loss,
        "overall": f"{total_pass}/{total}",
        "overall_pct": round(overall_pct, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "categories": {k: {"pass": v["pass"], "total": v["total"], "pct": round(v["pct"], 1)}
                       for k, v in all_results.items()},
    }
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Results saved → {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--token",      type=str, default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")

    if args.checkpoint:
        ckpt_path = os.path.join(args.checkpoint, "model.pt")
        ckpt_name = os.path.basename(args.checkpoint.rstrip("/\\"))
        step = None
        loss = None
        meta_path = os.path.join(args.checkpoint, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            step = meta.get("step")
            loss = meta.get("loss")
    elif args.token:
        from huggingface_hub import list_repo_files, hf_hub_download
        files = list(list_repo_files(repo_id=HF_REPO, repo_type="model", token=args.token))
        ckpts = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-") and "_temp" not in f})
        ckpt_name = ckpts[-1]
        print(f"Using latest: {ckpt_name}")
        local_dir = os.path.join("checkpoints/yaya-125m-sft", ckpt_name)
        ckpt_path = os.path.join(local_dir, "model.pt")
        if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 100_000_000:
            os.makedirs(local_dir, exist_ok=True)
            hf_hub_download(repo_id=HF_REPO, filename=f"{ckpt_name}/model.pt",
                            repo_type="model", token=args.token, local_dir="checkpoints/yaya-125m-sft")
        try:
            p = hf_hub_download(repo_id=HF_REPO, filename=f"{ckpt_name}/metadata.json",
                                repo_type="model", token=args.token,
                                local_dir=f"/tmp/bench_{ckpt_name}", force_download=True)
            with open(p) as f:
                meta = json.load(f)
            step = meta.get("step")
            loss = meta.get("loss")
        except Exception:
            step = int(ckpt_name.split("-")[1])
            loss = None
    else:
        print("Error: provide --checkpoint or --token")
        sys.exit(1)

    print(f"Loading model from {ckpt_path}...")
    model = load_model(ckpt_path)
    print("Running benchmark...")
    all_results, total_pass, total, overall_pct = run_benchmark(model, tokenizer, step, loss)
    print_report(all_results, total_pass, total, overall_pct, step, loss)
    save_results(all_results, total_pass, total, overall_pct, ckpt_name, step, loss)


if __name__ == "__main__":
    main()
