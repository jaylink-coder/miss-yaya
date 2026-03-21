"""Mix multi-domain pretrain data into final training shards.

Pipeline:
  1. Tokenize each data/raw/{domain}/*.txt into data/processed/shards/{domain}/
  2. Run ShardMixer to combine domains with mix ratios from sources.yaml
  3. Output final shards to data/processed/mixed/

Usage:
    # Full run (all domains present in data/raw/)
    python scripts/mix_pretrain_data.py

    # Limit total tokens (useful for Kaggle/Colab budget)
    python scripts/mix_pretrain_data.py --total_tokens 3_000_000_000

    # Dry run — show what would be mixed without writing output
    python scripts/mix_pretrain_data.py --dry_run
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

RAW_DIR        = "data/raw"
SHARDS_DIR     = "data/processed/shards"
MIXED_DIR      = "data/processed/mixed"
SOURCES_CONFIG = "configs/data/sources.yaml"
TOKENIZER_PATH = "data/tokenizer/yaya_tokenizer.model"

# Maps data/raw/ directory names → sources.yaml category names
DOMAIN_TO_CATEGORY = {
    "academic":      "academic",
    "books":         "books",
    "business":      "domain",
    "code":          "code",
    "conversational":"conversation",
    "math":          "math",
    "multilingual":  "web_text",   # fold into web_text
    "web":           "web_text",
    "wikipedia":     "wikipedia",  # if present from kaggle pipeline
}


def load_mix_ratios(sources_config: str) -> dict:
    """Load mix ratios from sources.yaml."""
    with open(sources_config) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("mix_ratios", {})


def tokenize_domain(domain: str, raw_dir: str, shard_dir: str, tokenizer) -> int:
    """Tokenize all .txt files in raw_dir into binary shards in shard_dir.

    Each line in a .txt file is treated as one document.
    Returns total token count written.
    """
    txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not txt_files:
        print(f"  [{domain}] No .txt files found — skipping.")
        return 0

    os.makedirs(shard_dir, exist_ok=True)

    # Check if already tokenized
    existing = sorted(glob.glob(os.path.join(shard_dir, "*.bin")))
    if existing:
        total_bytes = sum(os.path.getsize(f) for f in existing)
        total_tokens = total_bytes // 2
        print(f"  [{domain}] Already tokenized: {total_tokens/1e6:.0f}M tokens — skipping.")
        return total_tokens

    SHARD_SIZE = 50_000_000  # 50M tokens per shard
    buf = []
    shard_idx = 0
    total_tokens = 0
    docs_processed = 0

    for txt_path in txt_files:
        fname = os.path.basename(txt_path)
        print(f"  [{domain}] Tokenizing {fname}...", flush=True)
        with open(txt_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if len(text) < 50:
                    continue
                toks = tokenizer.encode(text)
                buf.extend(toks)
                docs_processed += 1

                while len(buf) >= SHARD_SIZE:
                    slab = buf[:SHARD_SIZE]
                    buf = buf[SHARD_SIZE:]
                    arr = np.array(slab, dtype=np.uint16)
                    path = os.path.join(shard_dir, f"shard_{shard_idx:05d}.bin")
                    arr.tofile(path)
                    total_tokens += len(arr)
                    print(f"    shard_{shard_idx:05d}: {len(arr)/1e6:.0f}M tokens", flush=True)
                    shard_idx += 1

    # Write remaining tokens
    if buf:
        arr = np.array(buf, dtype=np.uint16)
        path = os.path.join(shard_dir, f"shard_{shard_idx:05d}.bin")
        arr.tofile(path)
        total_tokens += len(arr)
        print(f"    shard_{shard_idx:05d} (final): {len(arr)/1e6:.0f}M tokens", flush=True)

    print(f"  [{domain}] Done: {total_tokens/1e6:.0f}M tokens from {docs_processed:,} docs")
    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="Tokenize domains + mix into final training shards")
    parser.add_argument("--raw_dir", default=RAW_DIR)
    parser.add_argument("--shards_dir", default=SHARDS_DIR)
    parser.add_argument("--output_dir", default=MIXED_DIR)
    parser.add_argument("--tokenizer_path", default=TOKENIZER_PATH)
    parser.add_argument("--sources_config", default=SOURCES_CONFIG)
    parser.add_argument(
        "--total_tokens", type=int, default=None,
        help="Target total tokens in mixed output (default: all available)"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without writing output shards")
    args = parser.parse_args()

    # ── 1. Discover available domains ─────────────────────────────────────────
    print("\n[1/3] Scanning data/raw/ for domain directories...\n")
    available_domains = {}
    for domain in sorted(os.listdir(args.raw_dir)):
        domain_path = os.path.join(args.raw_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        txt_files = glob.glob(os.path.join(domain_path, "*.txt"))
        if txt_files:
            available_domains[domain] = domain_path
            size_mb = sum(os.path.getsize(f) for f in txt_files) / 1e6
            print(f"  {domain:20s} {len(txt_files)} file(s)  {size_mb:.0f} MB")

    if not available_domains:
        print(f"No domain directories with .txt files found in {args.raw_dir}")
        print("Run: make download-pretrain-data")
        sys.exit(1)

    # ── 2. Tokenize each domain ────────────────────────────────────────────────
    print(f"\n[2/3] Tokenizing domains → {args.shards_dir}/\n")

    from src.tokenizer.tokenizer import YayaTokenizer
    tokenizer = YayaTokenizer(args.tokenizer_path)
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}\n")

    domain_shard_dirs = {}
    domain_token_counts = {}

    for domain, raw_path in available_domains.items():
        shard_dir = os.path.join(args.shards_dir, domain)
        if not args.dry_run:
            n = tokenize_domain(domain, raw_path, shard_dir, tokenizer)
        else:
            txt_files = glob.glob(os.path.join(raw_path, "*.txt"))
            n = sum(os.path.getsize(f) for f in txt_files) // 4  # rough estimate
            print(f"  [dry_run] {domain}: ~{n/1e6:.0f}M tokens")
        domain_shard_dirs[domain] = shard_dir
        domain_token_counts[domain] = n

    total_available = sum(domain_token_counts.values())
    print(f"\n  Total available: {total_available/1e9:.2f}B tokens across {len(domain_shard_dirs)} domains")

    # ── 3. Build category → shards mapping with mix ratios ────────────────────
    print(f"\n[3/3] Mixing into {args.output_dir}/\n")

    # Load mix ratios
    try:
        all_ratios = load_mix_ratios(args.sources_config)
    except Exception:
        # Fallback uniform ratios
        categories = set(DOMAIN_TO_CATEGORY[d] for d in available_domains if d in DOMAIN_TO_CATEGORY)
        all_ratios = {c: 1.0 / len(categories) for c in categories}

    # Build category_dirs: map category → merged shard dir
    # Multiple domains can map to the same category (e.g. web + multilingual → web_text)
    category_shards: dict[str, list[str]] = {}
    for domain, shard_dir in domain_shard_dirs.items():
        category = DOMAIN_TO_CATEGORY.get(domain, "web_text")
        if domain_token_counts.get(domain, 0) == 0:
            continue
        category_shards.setdefault(category, []).append(shard_dir)

    if not category_shards:
        print("No tokenized shards to mix.")
        sys.exit(1)

    # For categories with multiple source dirs, merge them conceptually
    # ShardMixer expects one dir per category — use first dir and symlink others
    # Simpler: create merged category dirs under shards_dir/_merged/
    merged_dir = os.path.join(args.shards_dir, "_merged")
    os.makedirs(merged_dir, exist_ok=True)

    category_dirs: dict[str, str] = {}
    for cat, src_dirs in category_shards.items():
        cat_merged = os.path.join(merged_dir, cat)
        os.makedirs(cat_merged, exist_ok=True)

        if not args.dry_run:
            # Symlink or copy shards from all source dirs into merged dir
            shard_counter = 0
            for src_dir in src_dirs:
                for bin_file in sorted(glob.glob(os.path.join(src_dir, "*.bin"))):
                    link = os.path.join(cat_merged, f"shard_{shard_counter:05d}.bin")
                    if not os.path.exists(link):
                        try:
                            os.symlink(os.path.abspath(bin_file), link)
                        except (OSError, NotImplementedError):
                            # Windows: copy instead of symlink
                            import shutil
                            shutil.copy2(bin_file, link)
                    shard_counter += 1

        category_dirs[cat] = cat_merged

    # Only include ratios for categories we have data for
    active_ratios = {cat: all_ratios.get(cat, 0.05) for cat in category_dirs}
    print("  Mix ratios (active categories):")
    total_ratio = sum(active_ratios.values())
    for cat, ratio in sorted(active_ratios.items(), key=lambda x: -x[1]):
        norm = ratio / max(total_ratio, 1e-8)
        tokens_est = domain_token_counts.get(cat, 0)
        print(f"    {cat:20s}  {norm*100:5.1f}%")

    if args.dry_run:
        print("\n  [dry_run] Skipping shard writing.")
        return

    from src.data.shard_mixer import ShardMixer
    mixer = ShardMixer(
        category_dirs=category_dirs,
        mix_ratios=active_ratios,
        output_dir=args.output_dir,
        shard_size=100_000_000,
        seed=42,
    )

    stats = mixer.mix(total_tokens=args.total_tokens)

    print(f"\n{'='*60}")
    print(f"  MIXING COMPLETE")
    print(f"  Total tokens: {stats['total_tokens']/1e9:.2f}B")
    print(f"  Output shards: {stats['num_shards']} in {args.output_dir}")
    print(f"\nNext step: update train_1b.yaml to point at {args.output_dir}")
    print(f"  Or run: make train-1b (it will read from data/processed/mixed/ automatically)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
