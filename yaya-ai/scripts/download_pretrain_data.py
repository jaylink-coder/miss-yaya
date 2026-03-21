"""
Download pretrain data for multi-source domain mixing into data/raw/.

Domains and sources:
  academic      → allenai/peS2o (scientific papers)
  books         → bookcorpus (fiction/non-fiction books)
  business      → financial_phrasebank + SEC-like public filings subset
  code          → bigcode/the-stack-smol (permissively licensed code)
  conversational→ OpenAssistant/oasst1 (human-written conversations)
  math          → lighteval/MATH + gsm8k (math word problems)
  multilingual  → Helsinki-NLP/tatoeba-translation-challenge (en↔multilingual)
  web           → allenai/c4 (filtered Common Crawl English)

Usage:
    python scripts/download_pretrain_data.py --all
    python scripts/download_pretrain_data.py --domain code math
    python scripts/download_pretrain_data.py --all --max_examples 50000
"""

import argparse
import json
import os
import re
import sys

RAW_DIR = "data/raw"

DOMAINS = ["academic", "books", "business", "code", "conversational", "math", "multilingual", "web"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def save_texts(texts: list[str], domain: str, tag: str) -> int:
    """Save a list of text strings to data/raw/<domain>/<tag>.txt, one doc per line."""
    out_dir = os.path.join(RAW_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.txt")
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            text = text.strip()
            if not text:
                continue
            # Replace newlines within a doc with a space, separate docs with newline
            f.write(text.replace("\n", " ") + "\n")
            written += 1
    return written


def load_hf(dataset_id: str, split: str = "train", **kwargs):
    """Load a HuggingFace dataset, fail clearly if datasets not installed."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)
    return load_dataset(dataset_id, split=split, trust_remote_code=True, **kwargs)


# ── Domain downloaders ─────────────────────────────────────────────────────────

def download_academic(max_examples: int) -> int:
    """allenai/peS2o — scientific papers (title + abstract)."""
    print("[academic] Downloading allenai/peS2o (scientific papers)...")
    ds = load_hf("allenai/peS2o", split="train", streaming=True)
    texts = []
    for item in ds:
        title = item.get("title", "").strip()
        abstract = item.get("abstract", "").strip()
        if title and abstract:
            texts.append(f"{title}. {abstract}")
        if len(texts) >= max_examples:
            break
    written = save_texts(texts, "academic", "pes2o")
    print(f"  [academic] {written:,} examples saved.")
    return written


def download_books(max_examples: int) -> int:
    """bookcorpusopen — open-access books."""
    print("[books] Downloading bookcorpusopen...")
    ds = load_hf("bookcorpusopen", split="train", streaming=True)
    texts = []
    for item in ds:
        text = item.get("text", "").strip()
        if len(text) > 200:
            texts.append(text[:2000])  # cap to 2K chars per book chunk
        if len(texts) >= max_examples:
            break
    written = save_texts(texts, "books", "bookcorpus")
    print(f"  [books] {written:,} examples saved.")
    return written


def download_business(max_examples: int) -> int:
    """financial_phrasebank + FiQA for business/financial domain text."""
    print("[business] Downloading financial_phrasebank + fiqa...")
    texts = []

    # financial_phrasebank: sentence-level financial sentiment text
    try:
        ds = load_hf("financial_phrasebank", "sentences_allagree", split="train", streaming=False)
        for item in ds:
            sentence = item.get("sentence", "").strip()
            if sentence:
                texts.append(sentence)
    except Exception as e:
        print(f"  [business] financial_phrasebank failed: {e}")

    # FiQA: finance Q&A
    try:
        ds2 = load_hf("BeIR/fiqa", split="corpus", streaming=True)
        for item in ds2:
            text = item.get("text", "").strip()
            if len(text) > 50:
                texts.append(text[:1000])
            if len(texts) >= max_examples:
                break
    except Exception as e:
        print(f"  [business] fiqa failed: {e}")

    texts = texts[:max_examples]
    written = save_texts(texts, "business", "finance")
    print(f"  [business] {written:,} examples saved.")
    return written


def download_code(max_examples: int) -> int:
    """bigcode/the-stack-smol — permissively licensed code (Python subset)."""
    print("[code] Downloading bigcode/the-stack-smol (Python)...")
    ds = load_hf("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("content", "").strip()
        if len(content) > 100:
            texts.append(content[:3000])  # cap long files
        if len(texts) >= max_examples:
            break
    written = save_texts(texts, "code", "python")
    print(f"  [code] {written:,} examples saved.")
    return written


def download_conversational(max_examples: int) -> int:
    """OpenAssistant/oasst1 — human-written multi-turn conversations."""
    print("[conversational] Downloading OpenAssistant/oasst1...")
    ds = load_hf("OpenAssistant/oasst1", split="train", streaming=False)

    # Build conversation threads: group by message_tree_id, reconstruct turns
    from collections import defaultdict
    trees: dict = defaultdict(list)
    for item in ds:
        trees[item["message_tree_id"]].append(item)

    texts = []
    for tree_id, messages in trees.items():
        # Sort by created_at to get order
        messages.sort(key=lambda x: x.get("created_at", ""))
        thread = []
        for msg in messages:
            role = "Human" if msg.get("role") == "prompter" else "Assistant"
            text = msg.get("text", "").strip()
            if text:
                thread.append(f"{role}: {text}")
        if len(thread) >= 2:
            texts.append("\n".join(thread))
        if len(texts) >= max_examples:
            break

    written = save_texts(texts, "conversational", "oasst1")
    print(f"  [conversational] {written:,} examples saved.")
    return written


def download_math(max_examples: int) -> int:
    """lighteval/MATH + gsm8k — math problems with solutions."""
    print("[math] Downloading lighteval/MATH + gsm8k...")
    texts = []

    # MATH dataset
    try:
        ds = load_hf("lighteval/MATH", split="train", streaming=False)
        for item in ds:
            problem = item.get("problem", "").strip()
            solution = item.get("solution", "").strip()
            if problem and solution:
                texts.append(f"Problem: {problem}\nSolution: {solution}")
            if len(texts) >= max_examples // 2:
                break
    except Exception as e:
        print(f"  [math] MATH dataset failed: {e}")

    # GSM8K
    try:
        ds2 = load_hf("gsm8k", "main", split="train", streaming=False)
        for item in ds2:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            if question and answer:
                texts.append(f"Problem: {question}\nSolution: {answer}")
            if len(texts) >= max_examples:
                break
    except Exception as e:
        print(f"  [math] gsm8k failed: {e}")

    texts = texts[:max_examples]
    written = save_texts(texts, "math", "math_gsm8k")
    print(f"  [math] {written:,} examples saved.")
    return written


def download_multilingual(max_examples: int) -> int:
    """Helsinki-NLP/tatoeba-translation-challenge — parallel sentence pairs."""
    print("[multilingual] Downloading Helsinki-NLP/tatoeba-translation-challenge (eng-*)...")
    # Sample a few language pairs
    lang_pairs = ["eng-fra", "eng-deu", "eng-spa", "eng-zho", "eng-ara"]
    texts = []
    per_pair = max(1, max_examples // len(lang_pairs))

    for pair in lang_pairs:
        try:
            ds = load_hf(
                "Helsinki-NLP/tatoeba-translation-challenge",
                pair,
                split="test",
                streaming=True,
            )
            count = 0
            for item in ds:
                src = item.get("sourceString", "").strip()
                tgt = item.get("targetString", "").strip()
                if src and tgt:
                    texts.append(f"{src} | {tgt}")
                    count += 1
                if count >= per_pair:
                    break
            print(f"  [{pair}] {count} pairs")
        except Exception as e:
            print(f"  [multilingual] {pair} failed: {e}")

    texts = texts[:max_examples]
    written = save_texts(texts, "multilingual", "tatoeba")
    print(f"  [multilingual] {written:,} examples saved.")
    return written


def download_web(max_examples: int) -> int:
    """allenai/c4 — filtered Common Crawl English web text."""
    print("[web] Downloading allenai/c4 (English subset)...")
    ds = load_hf("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for item in ds:
        text = item.get("text", "").strip()
        if len(text) > 200:
            texts.append(text[:2000])
        if len(texts) >= max_examples:
            break
    written = save_texts(texts, "web", "c4_en")
    print(f"  [web] {written:,} examples saved.")
    return written


# ── Dispatch ───────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "academic":      download_academic,
    "books":         download_books,
    "business":      download_business,
    "code":          download_code,
    "conversational": download_conversational,
    "math":          download_math,
    "multilingual":  download_multilingual,
    "web":           download_web,
}


def main():
    parser = argparse.ArgumentParser(description="Download pretrain domain data into data/raw/")
    parser.add_argument("--all", action="store_true", help="Download all domains")
    parser.add_argument("--domain", nargs="+", choices=DOMAINS, help="Specific domain(s)")
    parser.add_argument(
        "--max_examples", type=int, default=100_000,
        help="Max examples per domain (default 100K)",
    )
    args = parser.parse_args()

    if not args.all and not args.domain:
        parser.print_help()
        sys.exit(1)

    domains = DOMAINS if args.all else args.domain
    total = 0
    failed = []

    for domain in domains:
        print(f"\n{'='*60}")
        try:
            n = DOWNLOADERS[domain](args.max_examples)
            total += n
        except Exception as e:
            print(f"  ERROR downloading {domain}: {e}")
            failed.append(domain)

    print(f"\n{'='*60}")
    print(f"Done. Total examples written: {total:,}")
    if failed:
        print(f"Failed domains: {failed}")
    print(f"\nNext step: run 'make train-tokenizer' to rebuild tokenizer, then 'make prepare-data'")


if __name__ == "__main__":
    main()
