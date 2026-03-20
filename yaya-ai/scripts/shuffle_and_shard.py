"""
Shuffle and re-shard an existing shard_00000.bin into multiple shuffled shards.

Solves the training instability caused by alphabetically-ordered Wikipedia data.
Loads the binary file, shuffles sequences randomly, writes N output shards.

Usage:
    python scripts/shuffle_and_shard.py \
        --input  data/processed/wikipedia/train/shard_00000.bin \
        --outdir data/processed/wikipedia/train \
        --n_shards 10 \
        --seq_len 1025 \
        --seed 42

After this, shard_00000.bin is renamed to .bak so StreamingTextDataset
picks up only the new shuffled shards (shard_00001.bin ... shard_00010.bin).
"""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True, help="Path to existing shard_00000.bin")
    parser.add_argument("--outdir",   required=True, help="Output directory for new shards")
    parser.add_argument("--n_shards", type=int, default=10)
    parser.add_argument("--seq_len",  type=int, default=1025,
                        help="Tokens per sequence (context_len + 1)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--delete_original", action="store_true",
                        help="Delete original file instead of renaming to .bak")
    args = parser.parse_args()

    print(f"Loading {args.input} ...", flush=True)
    tokens = np.fromfile(args.input, dtype=np.uint16)
    total_tokens = len(tokens)
    print(f"  {total_tokens:,} tokens  ({total_tokens * 2 / 1e6:.1f} MB)", flush=True)

    # Trim to whole number of sequences
    n_complete = total_tokens // args.seq_len
    tokens = tokens[: n_complete * args.seq_len]
    print(f"  {n_complete:,} complete sequences of {args.seq_len} tokens", flush=True)

    # Reshape and shuffle rows
    seqs = tokens.reshape(n_complete, args.seq_len)
    rng  = np.random.default_rng(args.seed)
    idx  = rng.permutation(n_complete)
    seqs = seqs[idx].copy()   # copy so we own the memory cleanly
    print(f"  Shuffled {n_complete:,} sequences (seed={args.seed})", flush=True)

    # Write shards (1-based index to avoid collision with original shard_00000)
    os.makedirs(args.outdir, exist_ok=True)
    seqs_per_shard = n_complete // args.n_shards

    for i in range(args.n_shards):
        start = i * seqs_per_shard
        end   = (i + 1) * seqs_per_shard if i < args.n_shards - 1 else n_complete
        chunk = seqs[start:end].flatten()

        shard_name = f"shard_{i + 1:05d}.bin"
        out_path   = os.path.join(args.outdir, shard_name)
        chunk.tofile(out_path)

        n_tok = len(chunk)
        print(f"  Wrote {shard_name}: {n_tok:,} tokens  ({n_tok * 2 / 1e6:.1f} MB)", flush=True)

    # Handle original file
    if args.delete_original:
        os.remove(args.input)
        print(f"\n  Deleted original {args.input}", flush=True)
    else:
        bak = args.input + ".bak"
        os.rename(args.input, bak)
        print(f"\n  Renamed original → {bak}", flush=True)

    print(f"\nDone. {args.n_shards} shuffled shards in {args.outdir}", flush=True)
    print("StreamingTextDataset will now shuffle between shards each epoch.", flush=True)


if __name__ == "__main__":
    main()
