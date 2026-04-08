# Yaya AI — Pre-Training Guide

## Why Pre-Train?

The current yaya-125m model was trained **directly with SFT from random initialization**. This means:

- The model has no general world knowledge — it only knows what's in the ~205K SFT examples
- Factual recall is weak and inconsistent (compensated by runtime guards)
- Reasoning is pattern-matching on SFT data, not true generalization
- Scaling to 350M+ without pre-training will not fix this

**Pre-training on a text corpus is the single biggest improvement available.**

Even a modest pre-training run (5–10B tokens) would give the model a foundation of:
- Language structure and grammar
- World knowledge (geography, science, history)
- Basic reasoning patterns
- Vocabulary and concept associations

## Recommended Pre-Training Data Mix

| Source | Tokens (est.) | Purpose | Download |
|---|---|---|---|
| Wikipedia (en) | ~4B | World knowledge, facts | `datasets.load_dataset("wikipedia", "20220301.en")` |
| TinyStories | ~500M | Coherent narrative generation | `datasets.load_dataset("roneneldan/TinyStories")` |
| OpenWebText2 | ~8B | Web text diversity | `datasets.load_dataset("openwebtext2")` |
| The Pile (subset) | ~5B | Code, books, academic | `datasets.load_dataset("EleutherAI/pile", split="train", streaming=True)` |
| Swahili Wikipedia | ~50M | Bilingual capability | `datasets.load_dataset("wikipedia", "20220301.sw")` |

**Recommended total: 5–10B tokens for 125M model, 15–30B tokens for 350M model.**

Chinchilla-optimal: ~20x model params in tokens (125M → 2.5B tokens, 350M → 7B tokens).

## Step-by-Step Pipeline

### 1. Download Raw Data

```bash
# Use existing scripts
python scripts/download_pretrain_data.py --sources wikipedia,tinystories,openwebtext
```

Or manually:

```python
from datasets import load_dataset

# Wikipedia
wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
for article in wiki:
    # Save to data/raw/wikipedia/
    pass

# TinyStories (small, good for bootstrapping)
stories = load_dataset("roneneldan/TinyStories", split="train")
```

### 2. Process and Filter

```bash
# Dedup, quality filter, tokenize
python scripts/process_pretrain_data.py \
    --input data/raw/ \
    --output data/processed/pretrain/ \
    --tokenizer data/tokenizer/yaya_tokenizer.model \
    --min_length 64 \
    --max_length 2048
```

Key quality filters:
- **Min length**: 64 tokens (drop very short docs)
- **Max length**: 2048 tokens (truncate or split long docs)
- **Language filter**: Keep English + Swahili only
- **Deduplication**: MinHash dedup at document level
- **Quality**: Perplexity filter using a small reference model

### 3. Create Token Mix

```bash
python scripts/mix_pretrain_data.py \
    --sources data/processed/pretrain/ \
    --output data/processed/pretrain_mixed.bin \
    --mix "wikipedia:0.4,openwebtext:0.3,tinystories:0.1,pile:0.15,swahili:0.05"
```

### 4. Pre-Train

```bash
# 125M model — feasible on a single T4 with gradient checkpointing
python scripts/train.py \
    --model_config configs/model/yaya_125m.yaml \
    --train_config configs/training/train_125m.yaml \
    --data data/processed/pretrain_mixed.bin

# 350M model — needs A100 or L4 GPU
python scripts/train.py \
    --model_config configs/model/yaya_350m.yaml \
    --train_config configs/training/train_350m.yaml \
    --data data/processed/pretrain_mixed.bin
```

### 5. Then SFT

```bash
# SFT from pre-trained checkpoint (NOT from scratch)
python scripts/train_sft.py \
    --model_config configs/model/yaya_125m.yaml \
    --train_config configs/training/sft_125m.yaml \
    --pretrain_checkpoint checkpoints/yaya-125m-pretrain/latest
```

## Compute Estimates

| Model | Tokens | GPU | Time (est.) | Cost (cloud) |
|---|---|---|---|---|
| 125M | 2.5B | T4 (Kaggle free) | ~20 hrs | Free |
| 125M | 5B | T4 | ~40 hrs | Free (multiple sessions) |
| 350M | 7B | A100 (40GB) | ~12 hrs | ~$25 |
| 350M | 7B | T4 | ~80 hrs | Free (many sessions) |

## Quick Start (Minimal Pre-Training)

If compute is limited, even a small pre-training run helps:

```bash
# Download just Wikipedia (~4B tokens)
python -c "
from datasets import load_dataset
ds = load_dataset('wikipedia', '20220301.en', split='train')
ds.to_json('data/raw/wikipedia_en.jsonl')
"

# Tokenize and pack
python scripts/process_pretrain_data.py \
    --input data/raw/wikipedia_en.jsonl \
    --output data/processed/wiki_pretrain.bin

# Pre-train for 10K steps (~500M tokens at batch 32, seq 512)
# This alone will dramatically improve factual knowledge
python scripts/train.py \
    --model_config configs/model/yaya_125m.yaml \
    --train_config configs/training/train_125m.yaml \
    --data data/processed/wiki_pretrain.bin \
    --max_steps 10000
```

## Expected Impact

Based on scaling laws and empirical results from similar projects:

| Capability | Before Pre-Training | After 5B Token Pre-Training |
|---|---|---|
| Capital of France? | Needs runtime guard | Model answers correctly |
| Basic arithmetic | Needs calculator guard | ~60% correct (still weak) |
| "Who are you?" | Needs identity guard | ~70% correct after SFT |
| Multi-step reasoning | Very poor | Moderate improvement |
| Factual knowledge | Memorized from SFT only | Broad coverage |
| Guard lift (benchmark) | ~50+ percentage points | ~5-15 percentage points |

The goal: reduce guard lift from 50+pp to <15pp, then eventually <5pp.
