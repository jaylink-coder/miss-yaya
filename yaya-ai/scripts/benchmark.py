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

# ── Test suite (105 questions across 9 categories) ────────────────────────────
SUITE = {
    "Basic Arithmetic": [
        ("2 + 2",       "What is 2 + 2?",                          ["4"]),
        ("3 + 3",       "What is 3 + 3?",                          ["6"]),
        ("10 - 4",      "What is 10 - 4?",                         ["6"]),
        ("15 - 9",      "What is 15 - 9?",                         ["6"]),
        ("5 x 5",       "What is 5 x 5?",                          ["25"]),
        ("7 x 8",       "What is 7 x 8?",                          ["56"]),
        ("8 x 9",       "What is 8 x 9?",                          ["72"]),
        ("12 x 12",     "What is 12 x 12?",                        ["144"]),
        ("100 / 4",     "What is 100 divided by 4?",               ["25"]),
        ("144 / 12",    "What is 144 divided by 12?",              ["12"]),
        ("sqrt(16)",    "What is the square root of 16?",          ["4"]),
        ("sqrt(25)",    "What is the square root of 25?",          ["5"]),
        ("sqrt(81)",    "What is the square root of 81?",          ["9"]),
    ],
    "Word Problems": [
        ("apples",      "If I have 10 apples and give away 3, how many do I have left?",  ["7"]),
        ("shirt",       "A shirt costs $15. I pay with $20. How much change do I get?",   ["5", "$5"]),
        ("distance",    "A car travels at 60 km/h for 2 hours. How far does it go?",      ["120"]),
        ("eggs",        "A dozen eggs costs $3. How much do 2 dozen cost?",               ["6", "$6"]),
        ("percent",     "What is 15% of 200?",                                            ["30"]),
        ("half",        "What is half of 84?",                                            ["42"]),
        ("books",       "A book costs 250 shillings. I buy 4 books. How much do I pay?",  ["1000", "1,000"]),
        ("speed",       "A train travels 300 km in 3 hours. What is its speed in km/h?",  ["100"]),
        ("savings",     "I save 500 shillings per week. How much do I save in 4 weeks?",  ["2000", "2,000"]),
        ("discount",    "An item costs 800 shillings. There is a 25% discount. How much do I pay?", ["600"]),
    ],
    "Factual Knowledge": [
        ("france",      "What is the capital of France?",           ["paris"]),
        ("kenya",       "What is the capital of Kenya?",            ["nairobi"]),
        ("japan",       "What is the capital of Japan?",            ["tokyo"]),
        ("nigeria",     "What is the capital of Nigeria?",          ["abuja"]),
        ("sky",         "What color is the sky?",                   ["blue"]),
        ("week",        "How many days are in a week?",             ["7"]),
        ("year",        "How many months are in a year?",           ["12"]),
        ("day",         "How many hours are in a day?",             ["24"]),
        ("hour",        "How many minutes are in an hour?",         ["60"]),
        ("water",       "What is the chemical formula for water?",  ["h2o", "h₂o"]),
        ("planet",      "What planet do we live on?",               ["earth"]),
        ("boiling",     "At what temperature does water boil?",     ["100", "212"]),
        ("sun",         "What is the largest object in our solar system?", ["sun"]),
        ("continents",  "How many continents are there on Earth?",  ["7"]),
        ("gravity",     "What pulls objects toward Earth?",         ["gravity"]),
    ],
    "Identity": [
        ("name",        "What is your name?",                       ["yaya"]),
        ("who",         "Who are you?",                             ["yaya"]),
        ("not gpt",     "Are you ChatGPT?",                         ["no", "not"]),
        ("not human",   "Are you a human?",                         ["no", "not", "ai", "assistant"]),
        ("not claude",  "Are you Claude?",                          ["no", "not"]),
        ("what model",  "What AI model are you?",                   ["yaya"]),
    ],
    "Reasoning": [
        ("logic",       "All dogs are animals. Rex is a dog. Is Rex an animal?",           ["yes"]),
        ("lily pad",    "A lily pad doubles every day. After 48 days it covers the lake. When was it half covered?", ["47"]),
        ("bat ball",    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?", ["0.05", "5 cent"]),
        ("prime 17",    "Is 17 a prime number?",                    ["yes"]),
        ("not prime",   "Is 15 a prime number?",                    ["no"]),
        ("prime 2",     "Is 2 a prime number?",                     ["yes"]),
        ("not prime 9", "Is 9 a prime number?",                     ["no"]),
        ("cats logic",  "All cats have four legs. Whiskers is a cat. Does Whiskers have four legs?", ["yes"]),
        ("sequence",    "What comes next: 2, 4, 6, 8, ...?",        ["10"]),
        ("odd one out", "Which does not belong: apple, banana, carrot, mango?", ["carrot"]),
    ],
    "Language": [
        ("opp hot",     "What is the opposite of hot?",             ["cold"]),
        ("opp cold",    "What is the opposite of cold?",            ["hot"]),
        ("opp big",     "What is the opposite of big?",             ["small", "little", "tiny"]),
        ("opp day",     "What is the opposite of day?",             ["night"]),
        ("bigger",      "Which is bigger: the Sun or the Earth?",   ["sun"]),
        ("faster",      "Which is faster: a car or a bicycle?",     ["car"]),
        ("heavier",     "Which is heavier: a kilogram of iron or a kilogram of feathers?", ["same", "equal", "both"]),
        ("language",    "What language is this sentence written in?", ["english"]),
        ("plural",      "What is the plural of 'child'?",           ["children"]),
        ("sentence",    "What type of word is 'run' in 'I run every day'?", ["verb"]),
    ],
    "Kenya & East Africa": [
        ("mount kenya",  "What is the highest mountain in Kenya?",         ["mount kenya", "kenya"]),
        ("lake vic",     "What is the largest lake in Africa?",            ["victoria"]),
        ("rift valley",  "What is the name of the great geological feature running through Kenya?", ["rift", "valley"]),
        ("currency",     "What is the currency of Kenya?",                  ["shilling", "kes"]),
        ("independence", "In what year did Kenya gain independence?",        ["1963"]),
        ("nairobi",      "In which country is Nairobi located?",           ["kenya"]),
        ("swahili",      "What is the national language of Kenya alongside English?", ["swahili", "kiswahili"]),
        ("east africa",  "Name one country that borders Kenya.",            ["ethiopia", "somalia", "tanzania", "uganda", "south sudan"]),
        ("continent",    "Which continent is Kenya on?",                    ["africa"]),
        ("flag",         "What colors are on the Kenyan flag?",             ["black", "red", "green", "white"]),
    ],
    "Swahili Basics": [
        ("hello",       "What does 'Jambo' mean in English?",              ["hello", "hi", "greet"]),
        ("thank you",   "What does 'Asante' mean in English?",             ["thank", "thanks"]),
        ("welcome",     "What does 'Karibu' mean in English?",             ["welcome", "come in"]),
        ("water sw",    "What is the Swahili word for water?",             ["maji"]),
        ("yes sw",      "What is the Swahili word for yes?",               ["ndiyo"]),
        ("no sw",       "What is the Swahili word for no?",                ["hapana", "la"]),
        ("one sw",      "What is the Swahili word for one?",               ["moja"]),
        ("good sw",     "What does 'Nzuri' mean in English?",              ["good", "fine", "nice", "great"]),
    ],
    "Conversational": [
        ("greet",       "Hello! How are you?",                             ["good", "well", "fine", "great", "hello", "hi", "yaya"]),
        ("help",        "Can you help me with a question?",                ["yes", "sure", "of course", "happy", "glad"]),
        ("thank",       "Thank you for your help!",                        ["welcome", "pleasure", "glad", "happy"]),
        ("what do",     "What can you do?",                                ["help", "answer", "assist", "question"]),
        ("good",        "That is a great answer!",                         ["thank", "glad", "happy", "welcome"]),
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

    # Update dashboard HTML with new results
    dash_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_dashboard.py")
    if os.path.exists(dash_script):
        import subprocess
        subprocess.run([sys.executable, dash_script], check=False)


if __name__ == "__main__":
    main()
