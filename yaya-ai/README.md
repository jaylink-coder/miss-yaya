# Yaya AI — Multimodal Foundation Model

A multimodal AI system trained from scratch, combining language understanding, vision processing, and domain-specific capabilities for general-purpose, business, and creative applications.

## Project Structure

```
yaya-ai/
├── configs/                    # All configuration files
│   ├── model/                  # Model architecture configs (1.5B, 7B, 13B, etc.)
│   ├── training/               # Training hyperparameter configs
│   ├── data/                   # Data pipeline configs
│   └── serving/                # Inference & serving configs
│
├── src/                        # Core source code
│   ├── model/                  # Model architecture
│   │   ├── transformer.py      # Core transformer (blocks, layers)
│   │   ├── attention.py        # GQA, FlashAttention, KV-cache
│   │   ├── embeddings.py       # Token + positional embeddings (RoPE)
│   │   ├── feedforward.py      # SwiGLU feed-forward network
│   │   ├── normalization.py    # RMSNorm
│   │   ├── vision_encoder.py   # Vision Transformer (ViT) encoder
│   │   ├── multimodal.py       # Multimodal fusion (projector, combined model)
│   │   └── yaya_model.py       # Top-level model class
│   │
│   ├── tokenizer/              # Tokenizer
│   │   ├── trainer.py          # BPE tokenizer training
│   │   ├── tokenizer.py        # Tokenizer wrapper with special tokens
│   │   └── vocab.py            # Vocabulary management
│   │
│   ├── data/                   # Data pipeline
│   │   ├── dataset.py          # Dataset classes (text, vision, multimodal)
│   │   ├── dataloader.py       # Streaming dataloader with distributed support
│   │   ├── processing.py       # Text cleaning, filtering, dedup
│   │   ├── image_processing.py # Image preprocessing and augmentation
│   │   └── mixing.py           # Data mixing and sampling strategy
│   │
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Main training loop
│   │   ├── distributed.py      # Distributed training setup (DeepSpeed, FSDP)
│   │   ├── optimizer.py        # AdamW + learning rate schedulers
│   │   ├── checkpointing.py    # Save/load checkpoints
│   │   ├── logging_utils.py    # W&B and console logging
│   │   └── loss.py             # Loss functions
│   │
│   ├── evaluation/             # Evaluation & benchmarking
│   │   ├── evaluator.py        # Main evaluation orchestrator
│   │   ├── benchmarks.py       # Benchmark runners (MMLU, HellaSwag, etc.)
│   │   ├── metrics.py          # Metric computations
│   │   └── safety.py           # Safety and toxicity evaluation
│   │
│   ├── inference/              # Inference & serving
│   │   ├── generator.py        # Text generation (greedy, sampling, beam)
│   │   ├── kv_cache.py         # KV-cache for efficient inference
│   │   ├── quantization.py     # Post-training quantization (INT4, INT8, FP8)
│   │   └── server.py           # FastAPI serving endpoint
│   │
│   └── utils/                  # Shared utilities
│       ├── config.py           # Configuration loading and validation
│       ├── io_utils.py         # File I/O helpers
│       ├── distributed_utils.py # Distributed computing helpers
│       └── profiling.py        # Performance profiling tools
│
├── scripts/                    # Executable scripts
│   ├── train.py                # Main training entry point
│   ├── train_tokenizer.py      # Tokenizer training script
│   ├── evaluate.py             # Evaluation entry point
│   ├── generate.py             # Interactive generation script
│   ├── serve.py                # Launch serving endpoint
│   ├── prepare_data.py         # Data preparation pipeline
│   └── convert_checkpoint.py   # Checkpoint format conversion
│
├── tests/                      # Unit and integration tests
│   ├── test_model.py           # Model architecture tests
│   ├── test_attention.py       # Attention mechanism tests
│   ├── test_tokenizer.py       # Tokenizer tests
│   ├── test_data.py            # Data pipeline tests
│   ├── test_training.py        # Training loop tests
│   └── test_inference.py       # Inference tests
│
├── docs/                       # Documentation
│   ├── architecture.md         # Architecture design document
│   ├── training_guide.md       # Training guide
│   └── api.md                  # API documentation
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── pyproject.toml              # Modern Python project config
├── Dockerfile                  # Container for training/serving
├── docker-compose.yml          # Multi-container orchestration
├── Makefile                    # Common commands
└── .gitignore                  # Git ignore rules
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Train tokenizer
python scripts/train_tokenizer.py --config configs/data/tokenizer.yaml

# Train model (single GPU)
python scripts/train.py --config configs/training/train_1.5b.yaml

# Train model (distributed)
torchrun --nproc_per_node=8 scripts/train.py --config configs/training/train_1.5b.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/latest --benchmarks mmlu,hellaswag

# Generate text
python scripts/generate.py --checkpoint checkpoints/latest --prompt "Hello, I am Yaya"

# Serve API
python scripts/serve.py --checkpoint checkpoints/latest --port 8000
```

## Architecture

- **LLM Core:** Dense transformer with RMSNorm, GQA, RoPE, SwiGLU
- **Vision:** Vision Transformer (ViT) encoder with learned patch embeddings
- **Multimodal Fusion:** Unified embedding decoder architecture (Method A)
- **Training:** BF16 mixed precision, DeepSpeed ZeRO-2/3, gradient checkpointing
- **Inference:** KV-cache, INT4/INT8 quantization, continuous batching

## Model Configurations

| Config | Params | Layers | Hidden | Heads | KV Heads |
|--------|--------|--------|--------|-------|----------|
| Yaya-1.5B | 1.5B | 24 | 2048 | 16 | 4 |
| Yaya-7B | 7B | 32 | 4096 | 32 | 8 |
| Yaya-13B | 13B | 40 | 5120 | 40 | 8 |

## License

Proprietary — All rights reserved.
