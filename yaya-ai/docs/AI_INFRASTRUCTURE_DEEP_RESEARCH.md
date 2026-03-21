# Deep Research: Complete AI Infrastructure Guide

> **Compiled for Yaya AI Project — March 2026**
> Covers architecture, hardware, software, procedures, requirements, costs, and everything needed to build, train, deploy, and operate an AI system from scratch.

---

## Table of Contents

1. [Overview: What is AI Infrastructure?](#1-overview)
2. [Hardware Infrastructure](#2-hardware-infrastructure)
3. [Software Stack](#3-software-stack)
4. [Model Architecture](#4-model-architecture)
5. [Data Infrastructure](#5-data-infrastructure)
6. [Training Pipeline: Step-by-Step Procedure](#6-training-pipeline)
7. [Scaling Laws & Compute Requirements](#7-scaling-laws)
8. [Distributed Training](#8-distributed-training)
9. [Post-Training: SFT, RLHF, DPO](#9-post-training)
10. [Evaluation & Benchmarking](#10-evaluation)
11. [Deployment & Serving Infrastructure](#11-deployment)
12. [MLOps & Production Operations](#12-mlops)
13. [Safety & Alignment Infrastructure](#13-safety)
14. [Cost Analysis](#14-cost-analysis)
15. [Yaya AI: How Our Architecture Maps to This](#15-yaya-mapping)

---

## 1. Overview: What is AI Infrastructure?

AI infrastructure is the **complete set of technologies, tools, hardware, and platforms** required to build, train, deploy, and maintain AI systems. Unlike traditional software, AI infrastructure must handle:

- **Massive datasets** (terabytes to petabytes)
- **Extreme computational workloads** (thousands of GPU-hours)
- **Model versioning and experiment tracking**
- **Continuous monitoring and retraining**
- **Safety, alignment, and content filtering**

### The Three Pillars

| Pillar | Components |
|--------|-----------|
| **Data** | Collection, cleaning, deduplication, tokenization, storage, versioning |
| **Compute** | GPUs/TPUs, clusters, networking, distributed training frameworks |
| **Software** | Frameworks, model code, training loops, serving, monitoring |

### AI Tech Stack Layers (2025-2026)

```
┌─────────────────────────────────────────────────┐
│           Applications & User Interfaces         │
├─────────────────────────────────────────────────┤
│           API Layer / Model Serving              │
│     (FastAPI, vLLM, TGI, Triton, TorchServe)    │
├─────────────────────────────────────────────────┤
│           Safety & Alignment Layer               │
│     (Guardrails, Content Filtering, RLHF)        │
├─────────────────────────────────────────────────┤
│           Model Layer                            │
│     (Transformer, SSM, MoE architectures)        │
├─────────────────────────────────────────────────┤
│           Training & Optimization Layer          │
│     (PyTorch, DeepSpeed, Megatron, FSDP)         │
├─────────────────────────────────────────────────┤
│           Data Layer                             │
│     (Pipelines, Tokenizers, Feature Stores)      │
├─────────────────────────────────────────────────┤
│           Compute Layer                          │
│     (GPUs, Clusters, Networking, Storage)         │
├─────────────────────────────────────────────────┤
│           Infrastructure Layer                   │
│     (Cloud/On-prem, Kubernetes, Docker)           │
└─────────────────────────────────────────────────┘
```

---

## 2. Hardware Infrastructure

### 2.1 GPU Comparison (2024-2026)

| GPU | Architecture | VRAM | Memory BW | FP16 TFLOPS | FP8 TFLOPS | Interconnect | Price (est.) |
|-----|-------------|------|-----------|-------------|------------|-------------|-------------|
| **A100 80GB** | Ampere (2020) | 80 GB HBM2e | 2.0 TB/s | 312 | N/A | NVLink 600GB/s | $15,000-20,000 |
| **H100 SXM** | Hopper (2022) | 80 GB HBM3 | 3.35 TB/s | 990 | 1,979 | NVLink 900GB/s | $27,000-40,000 |
| **H200 SXM** | Hopper (2024) | 141 GB HBM3e | 4.8 TB/s | 990 | 1,979 | NVLink 900GB/s | ~$35,000-45,000 |
| **B200** | Blackwell (2025) | 192 GB HBM3e | 8.0 TB/s | 2,250 | 4,500 | NVLink 1.8TB/s | ~$40,000-60,000 |
| **B300** | Blackwell Ultra (2025) | 288 GB HBM3e | 12.0 TB/s | 2,250+ | 4,500+ | NVLink 1.8TB/s | TBD |
| **GB200 NVL72** | Blackwell (2025) | 72x 192GB | Aggregate | Massive | Massive | NVLink Domain | ~$3M/rack |

### 2.2 Choosing the Right GPU

- **Small models (< 3B params)**: A100 40GB or L4 — sufficient VRAM, cost-effective
- **Medium models (3B-13B)**: A100 80GB or H100 — good balance of memory and compute
- **Large models (13B-70B)**: H100/H200 clusters — need high memory bandwidth and NVLink
- **Frontier models (70B+)**: B200/GB200 clusters — maximum compute density

### 2.3 Cluster Architecture

```
                    ┌──────────────────┐
                    │   Head Node      │
                    │  (Job Scheduler) │
                    └────────┬─────────┘
                             │ Ethernet (management)
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼─────┐  ┌────▼────────┐
     │  Node 0    │  │  Node 1    │  │  Node N     │
     │ 8x H100    │  │ 8x H100   │  │ 8x H100    │
     │ NVLink     │  │ NVLink    │  │ NVLink     │
     │ (intra)    │  │ (intra)   │  │ (intra)    │
     └────────┬───┘  └──────┬─────┘  └────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    InfiniBand / RoCE
                    (400Gbps inter-node)
```

**Key cluster components:**
- **Intra-node**: NVLink/NVSwitch connects GPUs within a single server (600-1800 GB/s)
- **Inter-node**: InfiniBand or RoCE v2 connects servers (200-400 Gbps per link)
- **Storage**: Parallel file systems (Lustre, GPFS, WekaFS) — must sustain 100+ GB/s aggregate read throughput to keep GPUs fed
- **Networking fabric**: Fat-tree or dragonfly topology for minimal latency
- **Cooling**: 40-70 kW per rack for GPU nodes; liquid cooling preferred for H100/B200

### 2.4 Storage Requirements

| Model Size | Training Data | Storage Needed | I/O Throughput |
|-----------|--------------|---------------|---------------|
| 1.5B | ~30B tokens (~60 GB) | 500 GB total | 1-5 GB/s |
| 7B | ~140B tokens (~280 GB) | 2 TB total | 5-20 GB/s |
| 70B | ~1.4T tokens (~2.8 TB) | 20 TB total | 50-100 GB/s |
| 405B (Llama 3) | ~15T tokens (~30 TB) | 200+ TB total | 200+ GB/s |

### 2.5 Networking Requirements

- **Intra-node**: NVLink (essential, built into GPU servers)
- **Inter-node**: InfiniBand HDR (200Gbps) or NDR (400Gbps) — **critical** for distributed training
  - Ethernet alternatives exist (RoCE) but InfiniBand has lower latency
- **Latency target**: < 5 microseconds between nodes for gradient all-reduce
- **Bandwidth**: Scales linearly with cluster size. 256-GPU cluster needs ~100 Tbps aggregate bisection bandwidth

---

## 3. Software Stack

### 3.1 Core Frameworks (2025-2026 Market Share)

| Framework | Use Case | Market Share | Notes |
|-----------|---------|-------------|-------|
| **PyTorch** | Training + research | ~55% production | De facto standard for LLM training |
| **JAX** | Google ecosystem, research | ~15% | Used by Google (Gemini), DeepMind |
| **TensorFlow** | Legacy + production serving | ~20% | Declining for new projects |
| **Triton (OpenAI)** | Custom GPU kernels | Growing | Write GPU kernels in Python |

### 3.2 Distributed Training Frameworks

| Framework | Developer | Key Feature |
|-----------|----------|------------|
| **DeepSpeed** | Microsoft | ZeRO optimizer, pipeline parallelism, MoE support |
| **Megatron-LM** | NVIDIA | Tensor + pipeline parallelism for massive models |
| **FSDP** | Meta/PyTorch | Fully Sharded Data Parallel (native PyTorch) |
| **Alpa** | UC Berkeley | Automated inter/intra-op parallelism |
| **ColossalAI** | HPC-AI Tech | Unified parallelism + heterogeneous training |

### 3.3 Serving & Inference Frameworks

| Framework | Key Feature | Best For |
|-----------|-----------|---------|
| **vLLM** | PagedAttention, continuous batching | High-throughput LLM serving |
| **TGI (Hugging Face)** | Production-ready, streaming | Easy deployment |
| **TensorRT-LLM** | NVIDIA-optimized inference | Maximum GPU utilization |
| **Triton Inference Server** | Multi-model, multi-framework | Enterprise serving |
| **llama.cpp / GGML** | CPU + quantized inference | Edge/local deployment |
| **ONNX Runtime** | Cross-platform | Portable inference |

### 3.4 MLOps & Experiment Tracking

| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment tracking, model registry, deployment |
| **Weights & Biases (W&B)** | Experiment tracking, visualization, sweeps |
| **DVC** | Data versioning, pipeline orchestration |
| **Kubeflow** | ML pipelines on Kubernetes |
| **Airflow** | Workflow orchestration |
| **Label Studio** | Data annotation |

### 3.5 Key Libraries

| Library | Purpose |
|---------|---------|
| **Hugging Face Transformers** | Pre-trained models, tokenizers, pipelines |
| **Hugging Face Datasets** | Dataset loading and processing |
| **SentencePiece** | Tokenizer training (BPE, Unigram) |
| **tiktoken** | Fast BPE tokenizer (OpenAI) |
| **Flash Attention** | Memory-efficient attention (2-4x speedup) |
| **bitsandbytes** | Quantization (4-bit, 8-bit) |
| **PEFT / LoRA** | Parameter-efficient fine-tuning |
| **TRL** | RLHF/DPO training |

---

## 4. Model Architecture

### 4.1 The Transformer (Still Dominant in 2026)

The Transformer architecture, introduced in "Attention Is All You Need" (2017), remains the foundation of virtually all frontier LLMs. Key variants:

| Variant | Models | Description |
|---------|--------|------------|
| **Decoder-only** | GPT-4, Llama, Mistral | Causal (autoregressive) — dominant for generation |
| **Encoder-only** | BERT, RoBERTa | Bidirectional — for embeddings/classification |
| **Encoder-Decoder** | T5, BART | Seq2seq — for translation/summarization |

**Decoder-only is the standard for modern LLMs.**

### 4.2 Modern Transformer Block (2024-2026 Best Practices)

```
Input Token IDs
      │
      ▼
┌─────────────────┐
│  Token Embedding │ + Positional Encoding (RoPE)
└────────┬────────┘
         │
         ▼  (repeat N layers)
┌─────────────────────────────────────────┐
│                                         │
│  ┌─────────────┐                        │
│  │  RMSNorm    │  (Pre-norm, not Post)  │
│  └──────┬──────┘                        │
│         │                               │
│  ┌──────▼──────────────┐                │
│  │ Grouped-Query       │                │
│  │ Attention (GQA)     │                │
│  │ + RoPE             │                │
│  │ + Flash Attention  │                │
│  └──────┬──────────────┘                │
│         │ + Residual                    │
│         ▼                               │
│  ┌─────────────┐                        │
│  │  RMSNorm    │                        │
│  └──────┬──────┘                        │
│         │                               │
│  ┌──────▼──────────────┐                │
│  │  SwiGLU FFN         │                │
│  │  (w1·x ⊙ silu(w3·x))│                │
│  └──────┬──────────────┘                │
│         │ + Residual                    │
│         ▼                               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final RMSNorm   │
├─────────────────┤
│  LM Head (Linear)│ → Logits → Token prediction
└─────────────────┘
```

### 4.3 Key Architecture Decisions

#### Attention Mechanism
| Type | KV Heads | Memory | Speed | Used By |
|------|----------|--------|-------|---------|
| **Multi-Head Attention (MHA)** | h heads | High | Baseline | GPT-3 |
| **Multi-Query Attention (MQA)** | 1 head | Lowest | Fastest | PaLM, Falcon |
| **Grouped-Query Attention (GQA)** | h/g groups | Medium | Fast | Llama 2/3, Mistral |

**GQA is the current standard** — balances quality and efficiency.

#### Positional Encoding
| Type | Max Length | Quality | Used By |
|------|-----------|---------|---------|
| **Absolute (sinusoidal)** | Fixed | Baseline | Original Transformer |
| **ALiBi** | Extrapolates | Good | BLOOM, MPT |
| **RoPE** | Extrapolable | Best | Llama, Mistral, Qwen, Gemma |

**RoPE is the standard** — encodes relative position via rotation matrices, supports context extension.

#### Normalization
| Type | Pros | Used By |
|------|------|---------|
| **LayerNorm** | Original, well-understood | GPT-2, BERT |
| **RMSNorm** | 10-15% faster, equally effective | Llama, Mistral, Gemma |
| **Pre-norm** | More stable training than post-norm | All modern LLMs |

**Pre-norm + RMSNorm is the standard.**

#### Activation Function
| Type | Formula | Used By |
|------|---------|---------|
| **ReLU** | max(0, x) | Legacy |
| **GELU** | x·Φ(x) | GPT-2, BERT |
| **SwiGLU** | (Wx ⊙ silu(Vx)) | Llama, Mistral, PaLM |

**SwiGLU is the standard** — ~1% better perplexity, slightly more parameters.

### 4.4 Emerging Architectures (Beyond Transformers)

| Architecture | Key Innovation | Status (2026) |
|-------------|---------------|--------------|
| **Mamba (SSM)** | Linear-time sequence modeling via Selective State Spaces | Promising, used in Jamba (AI21) |
| **RWKV** | Linear attention RNN-Transformer hybrid | Active development, v6+ |
| **Mixture of Experts (MoE)** | Sparse activation — only subset of params used per token | Production (Mixtral, DeepSeek, GPT-4) |
| **Hybrid SSM+Attention** | Combines Mamba blocks with attention layers | Emerging (Jamba, Zamba) |
| **Diffusion Transformers (DiT)** | For image/video generation | Standard for vision (Sora, Flux) |

**MoE is production-ready and dominant for large models.** Pure SSMs haven't replaced transformers yet but are closing the gap.

### 4.5 Model Size Configurations

| Model | Params | Layers | Hidden | Heads | KV Heads | FFN | Vocab |
|-------|--------|--------|--------|-------|----------|-----|-------|
| Llama 3.2 1B | 1.24B | 16 | 2048 | 32 | 8 | 8192 | 128K |
| Llama 3.1 8B | 8.03B | 32 | 4096 | 32 | 8 | 14336 | 128K |
| Llama 3.1 70B | 70.6B | 80 | 8192 | 64 | 8 | 28672 | 128K |
| Mistral 7B | 7.3B | 32 | 4096 | 32 | 8 | 14336 | 32K |
| DeepSeek-V3 | 671B MoE | 61 | 7168 | 128 | — | MoE | 128K |
| Qwen 2.5 72B | 72.7B | 80 | 8192 | 64 | 8 | 29568 | 152K |

---

## 5. Data Infrastructure

### 5.1 Data Sources for Pretraining

| Source | Size | Quality | Notes |
|--------|------|---------|-------|
| **Common Crawl** | 250+ TB raw | Medium | Web crawl, needs heavy filtering |
| **RefinedWeb (Falcon)** | 5TB clean | High | Pre-filtered Common Crawl |
| **The Pile** | 800 GB | High | Curated mix (EleutherAI) |
| **RedPajama** | 1.2 TB | High | Open reproduction of LLaMA data |
| **StarCoder data** | 250 GB | High | Code from GitHub |
| **Wikipedia** | ~20 GB | Very High | Encyclopedic knowledge |
| **Books (PG-19, BookCorpus)** | ~10 GB | High | Long-form text |
| **ArXiv** | ~50 GB | Very High | Scientific papers |
| **StackExchange** | ~30 GB | High | Q&A format |
| **GitHub Code** | Variable | Medium-High | Programming languages |

### 5.2 Data Processing Pipeline

```
Raw Data (Web Crawl, Books, Code, etc.)
    │
    ▼
┌─────────────────────────┐
│ 1. DOWNLOAD & EXTRACT   │  Wget, AWS CLI, HF datasets
│    - Decompress          │  
│    - Format normalize    │  
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. CLEAN                │  
│    - Remove HTML/markup  │  
│    - Fix encoding        │  
│    - Normalize unicode   │  
│    - Remove boilerplate  │  
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. FILTER                │  
│    - Language detection   │  (fasttext, lingua)
│    - Quality scoring      │  (perplexity, classifier)
│    - Length filtering      │  (min/max tokens)
│    - Toxicity filtering   │  (regex + classifier)
│    - PII removal          │  (regex patterns)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. DEDUPLICATE           │  
│    - Exact dedup          │  (hash-based)
│    - Near-dedup           │  (MinHash / SimHash)
│    - URL-level dedup      │  
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 5. MIX & SAMPLE          │  
│    - Domain weighting     │  (web 60%, books 15%,
│    - Curriculum schedule  │   code 15%, wiki 5%,
│    - Epoch management     │   academic 5%)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 6. TOKENIZE              │  
│    - BPE (SentencePiece)  │  
│    - Pack into sequences  │  (context_length tokens)
│    - Write binary shards  │  (.bin or .arrow)
└───────────┬─────────────┘
            ▼
      Training-ready data
```

### 5.3 Tokenization

| Algorithm | Description | Used By |
|-----------|------------|---------|
| **BPE (Byte Pair Encoding)** | Iteratively merges frequent byte pairs | GPT-4, Llama, Mistral |
| **WordPiece** | Similar to BPE, maximizes likelihood | BERT |
| **Unigram** | Statistical model, prunes from large vocab | T5, ALBERT |
| **SentencePiece** | Language-agnostic, processes raw bytes | Llama, Gemma |
| **tiktoken** | Fast BPE with byte-level fallback | GPT-3.5/4 |

**Typical vocabulary sizes:**
- Small models: 32K-64K tokens
- Large models: 128K-256K tokens
- Larger vocab = better compression but larger embedding matrix

### 5.4 Data Quality Metrics

| Metric | Description | Tool |
|--------|------------|------|
| **Perplexity filtering** | Remove text that a reference LM finds surprising | KenLM, small GPT-2 |
| **Repetition ratio** | Reject documents with excessive n-gram repetition | Custom |
| **Language score** | Confidence of language detection | fastText lid |
| **Toxicity score** | Classifier-based harmful content detection | Perspective API, custom |
| **Compression ratio** | Documents that compress poorly are often gibberish | gzip heuristic |

---

## 6. Training Pipeline: Step-by-Step Procedure

### Complete Procedure to Build an AI System from Scratch

```
Phase 0: PLANNING & DESIGN (1-2 months)
  │
  ├── Define objectives and capabilities
  ├── Choose model size based on compute budget
  ├── Select architecture (Transformer variant)
  ├── Plan data strategy (sources, quality, mix)
  ├── Set up infrastructure (cloud vs on-prem)
  └── Establish evaluation criteria
  │
Phase 1: DATA ENGINEERING (2-4 months)
  │
  ├── Build data download pipeline
  ├── Implement cleaning & filtering
  ├── Build deduplication system
  ├── Train tokenizer on representative sample
  ├── Implement data mixing & curriculum
  ├── Create tokenized training shards
  └── Validate data quality (spot checks, statistics)
  │
Phase 2: MODEL IMPLEMENTATION (1-2 months)
  │
  ├── Implement model architecture
  │   ├── Embedding layer
  │   ├── Transformer blocks (attention + FFN)
  │   ├── Positional encoding (RoPE)
  │   ├── Normalization (RMSNorm)
  │   └── LM head
  ├── Implement training loop
  │   ├── AdamW optimizer
  │   ├── Learning rate schedule (warmup + cosine decay)
  │   ├── Gradient clipping
  │   ├── Mixed precision (bf16)
  │   └── Checkpointing
  ├── Implement distributed training
  │   ├── Data parallelism (DDP/FSDP)
  │   ├── Tensor parallelism (if needed)
  │   └── Pipeline parallelism (if needed)
  └── Unit tests for all components
  │
Phase 3: PRETRAINING (weeks to months)
  │
  ├── Small-scale validation runs (1B params, subset of data)
  ├── Monitor loss curves, learning rate, gradient norms
  ├── Hyperparameter tuning on small scale
  ├── Full-scale training run
  │   ├── Checkpoint every N steps
  │   ├── Evaluate on held-out data periodically
  │   ├── Monitor hardware utilization (GPU MFU)
  │   └── Handle failures and resume from checkpoints
  └── Select best checkpoint based on validation loss
  │
Phase 4: POST-TRAINING (2-4 weeks)
  │
  ├── Supervised Fine-Tuning (SFT)
  │   ├── Curate instruction-following dataset
  │   ├── Include tool-use and structured output examples
  │   ├── Train for 1-3 epochs on SFT data
  │   └── Evaluate on instruction-following benchmarks
  ├── Alignment (RLHF or DPO)
  │   ├── Collect human preference data (or synthetic)
  │   ├── Train reward model (for RLHF) or use DPO directly
  │   ├── Run PPO/DPO optimization
  │   └── Evaluate helpfulness vs. harmlessness
  └── Safety fine-tuning
      ├── Red-teaming to find vulnerabilities
      ├── Train on refusal examples for harmful queries
      └── Validate safety benchmarks
  │
Phase 5: EVALUATION (1-2 weeks)
  │
  ├── Benchmark suite (MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval)
  ├── Safety evaluations (toxicity, bias, jailbreak resistance)
  ├── Human evaluation (helpfulness, coherence, factuality)
  ├── Domain-specific evaluations
  └── Compare against baselines
  │
Phase 6: DEPLOYMENT (2-4 weeks)
  │
  ├── Model optimization
  │   ├── Quantization (GPTQ, AWQ, GGUF)
  │   ├── Export formats (SafeTensors, ONNX)
  │   └── KV-cache optimization
  ├── Serving infrastructure
  │   ├── API server (OpenAI-compatible)
  │   ├── Rate limiting, authentication
  │   ├── Safety guardrails
  │   └── Monitoring & metrics
  ├── Scaling
  │   ├── Load balancing
  │   ├── Auto-scaling
  │   └── Multi-region deployment
  └── CI/CD pipeline for model updates
  │
Phase 7: OPERATIONS (Ongoing)
  │
  ├── Monitor model quality (drift detection)
  ├── Collect user feedback
  ├── Periodic retraining/fine-tuning
  ├── Safety monitoring and incident response
  └── Cost optimization
```

### 6.1 Pretraining Details

**Objective**: Causal Language Modeling (CLM) — predict next token
```
Loss = -Σ log P(token_t | token_1, ..., token_{t-1})
```

**Typical hyperparameters for a 1.5B model:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 3e-4 (peak) | With warmup + cosine decay |
| Warmup Steps | 2,000 | Linear warmup |
| Batch Size | 512-2048 sequences | Gradient accumulation if needed |
| Sequence Length | 2048-4096 tokens | Longer = better but more memory |
| Weight Decay | 0.1 | AdamW |
| Adam β1, β2 | 0.9, 0.95 | Standard for LLMs |
| Gradient Clip | 1.0 | By global norm |
| Precision | bf16 | Brain float 16 (preferred over fp16) |
| Optimizer | AdamW | Fused implementation preferred |

**Training duration estimates:**

| Model Size | Tokens | GPUs (H100) | Time | Cost (cloud) |
|-----------|--------|-------------|------|-------------|
| 1.5B | 30B tokens | 8x H100 | ~3 days | ~$3,000-5,000 |
| 7B | 1T tokens | 64x H100 | ~2 weeks | ~$100,000-200,000 |
| 13B | 2T tokens | 128x H100 | ~3 weeks | ~$300,000-500,000 |
| 70B | 15T tokens | 512x H100 | ~2 months | ~$2,000,000-5,000,000 |

---

## 7. Scaling Laws & Compute Requirements

### 7.1 Chinchilla Scaling Law (DeepMind, 2022)

The foundational insight: **for a fixed compute budget, there is an optimal balance between model size and training data.**

**Original Chinchilla ratio**: ~20 tokens per parameter
- A 1B parameter model should see ~20B tokens
- A 70B parameter model should see ~1.4T tokens

### 7.2 Modern Scaling Ratios (2024-2026)

Industry has moved far beyond Chinchilla-optimal, "over-training" smaller models on much more data:

| Source | Tokens:Parameters Ratio | Year |
|--------|------------------------|------|
| **Chinchilla (DeepMind)** | 20:1 | 2022 |
| **DeepSeek scaling** | ~30:1 | 2024 |
| **Mosaic/Databricks** | ~190:1 | 2023 |
| **Tsinghua** | 192:1 | 2024 |
| **Llama 3** | **1,875:1** | 2024 |

**Why over-train?** Smaller models trained on more data are cheaper to serve at inference time while achieving similar quality to larger models trained on less data. Inference cost dominates in production.

### 7.3 Compute Estimation

**FLOPs for training** ≈ 6 × N × D
- N = number of parameters
- D = number of training tokens
- 6 = accounts for forward + backward pass

| Model | Params (N) | Tokens (D) | FLOPs | H100-hours |
|-------|-----------|-----------|-------|-----------|
| 1.5B | 1.5×10⁹ | 30×10⁹ | 2.7×10²⁰ | ~300 |
| 7B | 7×10⁹ | 1×10¹² | 4.2×10²² | ~20,000 |
| 70B | 70×10⁹ | 15×10¹² | 6.3×10²⁴ | ~3,000,000 |

**Model FLOPs Utilization (MFU):**
- Theoretical peak H100: ~990 TFLOPS (bf16)
- Practical MFU: 30-55% (depends on model size, batch size, optimization)
- Good target: 40-50% MFU

---

## 8. Distributed Training

### 8.1 Parallelism Strategies

When a model doesn't fit on one GPU, or training is too slow, use parallelism:

```
┌────────────────────────────────────────────────────────┐
│                 PARALLELISM STRATEGIES                   │
├─────────────────┬──────────────────┬───────────────────┤
│ Data Parallel   │ Tensor Parallel  │ Pipeline Parallel │
│                 │                  │                   │
│ Same model on   │ Split layers     │ Split layers      │
│ each GPU,       │ across GPUs      │ into stages       │
│ different data  │ (within a layer) │ (between layers)  │
│                 │                  │                   │
│ Scale: Easy     │ Scale: Moderate  │ Scale: Complex    │
│ Comm: AllReduce │ Comm: AllGather  │ Comm: P2P         │
│ Limit: Memory   │ Limit: Bandwidth │ Limit: Bubbles   │
└─────────────────┴──────────────────┴───────────────────┘
```

### 8.2 When to Use What

| Scenario | Strategy |
|---------|---------|
| Model fits on 1 GPU | Data Parallel (DDP) |
| Model fits on 1 GPU with optimization | FSDP (ZeRO-3) — shards optimizer states |
| Model doesn't fit on 1 GPU | Tensor Parallel + Data Parallel |
| Very large model (70B+) | All three (3D parallelism) |
| MoE model | Expert Parallel + Data Parallel |

### 8.3 ZeRO (Zero Redundancy Optimizer)

DeepSpeed's ZeRO eliminates memory redundancy across GPUs:

| Stage | What's Partitioned | Memory Savings |
|-------|-------------------|---------------|
| **ZeRO-1** | Optimizer states | ~4x |
| **ZeRO-2** | + Gradients | ~8x |
| **ZeRO-3** | + Parameters | ~N_gpu × (linear scaling) |

**FSDP** (PyTorch native) is equivalent to ZeRO-3.

### 8.4 Mixed Precision Training

| Precision | Bits | Memory | Speed | Quality |
|-----------|------|--------|-------|---------|
| **FP32** | 32 | Baseline | 1x | Perfect |
| **FP16** | 16 | 0.5x | 2x | Needs loss scaling |
| **BF16** | 16 | 0.5x | 2x | Better range, no loss scaling needed |
| **FP8** | 8 | 0.25x | 4x | Hopper/Blackwell only, emerging |

**BF16 is the standard for training.** FP8 is emerging with Blackwell GPUs.

### 8.5 Key Optimizations

- **Flash Attention**: Fuses attention computation, reduces memory from O(N²) to O(N). **Essential.**
- **Gradient Checkpointing**: Recompute activations during backward pass instead of storing. Trades compute for memory.
- **Activation Offloading**: Move activations to CPU memory when not needed. Slower but saves GPU memory.
- **Fused Kernels**: Combine multiple operations into single GPU kernels (e.g., fused AdamW, fused LayerNorm).

---

## 9. Post-Training: SFT, RLHF, DPO

### 9.1 The Post-Training Pipeline

```
Pretrained Base Model
        │
        ▼
┌─────────────────────────┐
│ Supervised Fine-Tuning   │  Instruction data: (prompt, response) pairs
│ (SFT)                    │  1-3 epochs, low LR (1e-5 to 5e-5)
│                          │  50K-500K high-quality examples
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│ Alignment                │
│ Option A: RLHF           │  Train reward model → PPO
│ Option B: DPO            │  Direct preference optimization (simpler)
│ Option C: KTO            │  Binary feedback (like/dislike)
│ Option D: ORPO           │  Odds ratio preference optimization
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│ Safety Fine-Tuning       │  Refusal training, red-team adversarial
│                          │  examples, constitutional AI constraints
└───────────┬──────────────┘
            ▼
    Aligned, Safe Chat Model
```

### 9.2 SFT Data Types

| Category | Examples | Count (typical) |
|----------|---------|----------------|
| **General instruction** | "Explain quantum physics simply" | 10K-50K |
| **Conversational** | Multi-turn dialogue | 10K-30K |
| **Tool use** | Calculator, search, code execution | 5K-20K |
| **Structured output** | JSON responses, data extraction | 5K-10K |
| **Safety refusals** | "I can't help with that because..." | 5K-10K |
| **Code** | Generate, explain, debug code | 10K-30K |
| **Math/reasoning** | Step-by-step problem solving | 10K-20K |
| **Creative writing** | Stories, poems, essays | 5K-10K |
| **RAG-grounded** | Answer based on provided context | 5K-10K |

### 9.3 RLHF vs. DPO

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| **Complexity** | High (reward model + RL) | Low (direct optimization) |
| **Compute** | 3-4x SFT cost | 1.5-2x SFT cost |
| **Stability** | Tricky (reward hacking) | More stable |
| **Quality** | Slightly better at scale | Comparable for most use cases |
| **Used By** | OpenAI (early ChatGPT) | Llama, Mistral, most open models |

**DPO is the current industry standard** for alignment due to simplicity and competitive quality.

### 9.4 Parameter-Efficient Fine-Tuning (PEFT)

When you can't afford full fine-tuning:

| Method | Trainable Params | Quality | Speed |
|--------|-----------------|---------|-------|
| **Full fine-tuning** | 100% | Best | Slowest |
| **LoRA** | 0.1-1% | ~95% of full | 3-5x faster |
| **QLoRA** | 0.1-1% (4-bit base) | ~93% of full | 5-10x faster |
| **Prefix Tuning** | < 0.1% | 85-90% | Fast |
| **Adapters** | 1-5% | ~92% | Moderate |

**LoRA / QLoRA is the standard** for fine-tuning when compute is limited.

---

## 10. Evaluation & Benchmarking

### 10.1 Standard Benchmarks

| Benchmark | Measures | Metric | Notes |
|-----------|---------|--------|-------|
| **MMLU** | Knowledge (57 subjects) | Accuracy (5-shot) | Most widely reported |
| **HellaSwag** | Commonsense reasoning | Accuracy (10-shot) | Sentence completion |
| **ARC-Challenge** | Science reasoning | Accuracy (25-shot) | Grade school science |
| **TruthfulQA** | Truthfulness | MC accuracy | Resistance to common misconceptions |
| **GSM8K** | Math reasoning | Accuracy (5-shot) | Grade school math |
| **HumanEval** | Code generation | pass@k | Python function completion |
| **MBPP** | Code generation | pass@k | Mostly basic programming |
| **Winogrande** | Commonsense | Accuracy | Pronoun resolution |
| **BBH** | Hard reasoning | Accuracy | BIG-Bench Hard subset |
| **MATH** | Competition math | Accuracy | Harder than GSM8K |
| **IFEval** | Instruction following | Accuracy | Can the model follow precise instructions |
| **MT-Bench** | Chat quality | LLM-as-judge score (1-10) | Multi-turn conversation |
| **AlpacaEval** | Helpfulness | Win rate vs reference | GPT-4 as judge |

### 10.2 Safety Benchmarks

| Benchmark | Measures |
|-----------|---------|
| **ToxiGen** | Toxic language generation |
| **RealToxicityPrompts** | Toxicity in continuations |
| **BBQ** | Bias across demographics |
| **WinoBias** | Gender bias |
| **CrowS-Pairs** | Stereotypical bias |
| **Red-team evaluation** | Adversarial jailbreak resistance |

### 10.3 Evaluation Best Practices

1. **Always report few-shot settings** (0-shot, 5-shot, etc.)
2. **Use standardized harnesses** (lm-evaluation-harness by EleutherAI)
3. **Report confidence intervals** where possible
4. **Include both automated and human evaluation**
5. **Test safety before deployment** (red-teaming)
6. **Monitor post-deployment** (drift, new failure modes)

---

## 11. Deployment & Serving Infrastructure

### 11.1 Model Optimization for Deployment

| Technique | Memory Savings | Speed Impact | Quality Impact |
|-----------|---------------|-------------|---------------|
| **FP16 inference** | 2x | ~Same | None |
| **INT8 quantization** | 4x | 1.5-2x faster | < 1% degradation |
| **INT4 quantization (GPTQ/AWQ)** | 8x | 2-3x faster | 1-3% degradation |
| **GGUF (llama.cpp)** | Variable | Good on CPU | 1-5% depending on quant |
| **KV-cache optimization** | Significant | Faster decoding | None |
| **Speculative decoding** | None | 2-3x faster | None |
| **Continuous batching** | None | 5-10x throughput | None |
| **PagedAttention (vLLM)** | ~4x KV-cache | Better batching | None |

### 11.2 Serving Architecture

```
┌────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Client    │────▶│  Load        │────▶│  API Gateway    │
│  (Browser,  │     │  Balancer    │     │  - Auth          │
│   Mobile,   │     │  (NGINX,     │     │  - Rate Limit    │
│   API)      │     │   HAProxy)   │     │  - Logging       │
└────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                                    ┌─────────────┼─────────────┐
                                    │             │             │
                              ┌─────▼────┐  ┌────▼─────┐  ┌───▼──────┐
                              │ Worker 0  │  │ Worker 1 │  │ Worker N │
                              │ GPU 0-3   │  │ GPU 4-7  │  │ GPU ...  │
                              │ vLLM/TGI  │  │ vLLM/TGI │  │ vLLM/TGI │
                              └──────────┘  └──────────┘  └──────────┘
                                    │             │             │
                              ┌─────▼─────────────▼─────────────▼─────┐
                              │        Model Weights (shared NFS/S3)  │
                              └───────────────────────────────────────┘
```

### 11.3 Key Serving Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **TTFT** | Time to First Token | < 500ms |
| **TPS** | Tokens Per Second (per user) | 30-80 tokens/s |
| **Throughput** | Total tokens/second (all users) | Maximize |
| **p99 Latency** | 99th percentile response time | < 5 seconds |
| **Availability** | Uptime percentage | > 99.9% |
| **Cost/token** | Infrastructure cost per generated token | Minimize |

### 11.4 Export Formats

| Format | Use Case | Tools |
|--------|---------|-------|
| **SafeTensors** | Safe, fast model loading | Hugging Face |
| **ONNX** | Cross-platform inference | ONNX Runtime |
| **GGUF** | llama.cpp / local inference | llama.cpp |
| **TensorRT** | NVIDIA-optimized inference | TensorRT-LLM |
| **CoreML** | Apple devices | coremltools |
| **TorchScript** | PyTorch production | torch.jit |

---

## 12. MLOps & Production Operations

### 12.1 MLOps Components

```
┌─────────────────────────────────────────────────────────┐
│                    MLOps Platform                         │
├──────────┬──────────┬───────────┬──────────┬────────────┤
│ Experiment│  Model   │ Pipeline  │ Serving  │ Monitoring │
│ Tracking  │ Registry │ Orchestr. │ Infra    │ & Alerts   │
│           │          │           │          │            │
│ W&B       │ MLflow   │ Airflow   │ K8s      │ Prometheus │
│ MLflow    │ HF Hub   │ Kubeflow  │ Docker   │ Grafana    │
│ Neptune   │ Custom   │ Prefect   │ Triton   │ Datadog    │
└──────────┴──────────┴───────────┴──────────┴────────────┘
```

### 12.2 CI/CD for ML

| Stage | What It Does | Tools |
|-------|-------------|-------|
| **Code CI** | Lint, type check, unit tests | GitHub Actions, pytest, ruff |
| **Data CI** | Validate data quality, schema | Great Expectations, DVC |
| **Model CI** | Train on subset, check metrics | Custom, MLflow |
| **Model CD** | Deploy if metrics pass thresholds | ArgoCD, Seldon |
| **Monitoring** | Track drift, performance, cost | Prometheus, custom |

### 12.3 Model Versioning

Track everything:
- **Code version**: Git commit hash
- **Data version**: DVC hash or data snapshot ID
- **Model weights**: Checkpoint path + metadata
- **Config**: Hyperparameters, architecture config
- **Metrics**: All benchmark scores at time of deployment
- **Environment**: Docker image hash, dependency versions

---

## 13. Safety & Alignment Infrastructure

### 13.1 Safety Layers

```
User Input
    │
    ▼
┌──────────────────────┐
│ Input Safety Filter   │  Toxicity detection, prompt injection
│                       │  detection, PII detection
└──────────┬───────────┘
           │ (blocked → refusal)
           ▼
┌──────────────────────┐
│ Model Generation      │  With safety-tuned weights
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Output Safety Filter  │  PII redaction, toxicity check,
│                       │  hallucination detection
└──────────┬───────────┘
           │
           ▼
    Safe Response to User
```

### 13.2 Safety Components

| Component | Purpose | Implementation |
|-----------|---------|---------------|
| **Toxicity Detector** | Detect harmful content categories | Regex + classifier |
| **Prompt Injection Detector** | Detect jailbreak attempts | Pattern matching + classifier |
| **PII Detector** | Find personal info (email, phone, SSN) | Regex + NER model |
| **Output Validator** | Ensure safe, appropriate output | Rule-based + classifier |
| **Refusal Generator** | Produce helpful refusals with resources | Template-based |
| **Guardrails Engine** | Orchestrate all safety checks | Pipeline coordinator |

### 13.3 Alignment Techniques

| Technique | Description |
|-----------|------------|
| **RLHF** | Train reward model from human preferences, use PPO |
| **DPO** | Directly optimize on preference pairs (simpler) |
| **Constitutional AI** | Self-critique using principles |
| **RLAIF** | Use AI feedback instead of human feedback |
| **Red-teaming** | Adversarial testing by humans or models |
| **Safety fine-tuning** | Train on (harmful_query → refusal) pairs |

---

## 14. Cost Analysis

### 14.1 Training Costs (Cloud, 2025-2026 prices)

| Resource | Cost/hour | Notes |
|----------|----------|-------|
| H100 SXM (cloud) | $2.50-4.00 | Lambda, CoreWeave, RunPod |
| H200 SXM (cloud) | $3.50-5.00 | Limited availability |
| A100 80GB (cloud) | $1.50-2.50 | Widely available |
| B200 (cloud) | $5.00-8.00 | Emerging, 2025+ |

### 14.2 Total Project Cost Estimates

| Model Size | Training | Post-Training | Infra/Ops | Total |
|-----------|---------|--------------|----------|-------|
| **1.5B** | $3K-10K | $1K-3K | $2K-5K | **$6K-18K** |
| **7B** | $100K-200K | $20K-50K | $30K-50K | **$150K-300K** |
| **13B** | $300K-500K | $50K-100K | $50K-100K | **$400K-700K** |
| **70B** | $2M-5M | $200K-500K | $200K-500K | **$2.5M-6M** |
| **400B+** | $50M-100M+ | $5M-10M | $5M-10M | **$60M-120M+** |

*Note: DeepSeek-V3 (671B MoE) was reported at ~$5.5M — a landmark in cost efficiency through MoE architecture and engineering optimization.*

### 14.3 Inference Costs

| Model | Quantization | GPU | Cost/1M tokens |
|-------|-------------|-----|---------------|
| 7B | INT4 | 1x A100 | ~$0.03-0.10 |
| 13B | INT4 | 1x A100 | ~$0.05-0.15 |
| 70B | INT4 | 2x A100 | ~$0.30-0.80 |
| 70B | INT4 | 1x H200 | ~$0.15-0.40 |

### 14.4 Cost Optimization Strategies

1. **Use spot/preemptible instances** (50-70% savings, need robust checkpointing)
2. **Quantize for inference** (INT4 = 8x memory reduction)
3. **Use smaller, over-trained models** (1.5B trained on 100B tokens ≈ 7B trained on 20B tokens in some tasks)
4. **Batch inference requests** (continuous batching via vLLM)
5. **Speculative decoding** (2-3x speedup at inference)
6. **KV-cache optimization** (PagedAttention)
7. **MoE architecture** (only activates subset of params per token)

---

## 15. Yaya AI: How Our Architecture Maps to This

### What We've Built (394 tests passing)

| Industry Component | Yaya Implementation | Status |
|-------------------|---------------------|--------|
| **Model Architecture** | Transformer + GQA + RoPE + SwiGLU + RMSNorm | ✅ Complete |
| **Vision** | ViT encoder + projector (multimodal) | ✅ Complete |
| **Tokenizer** | SentencePiece BPE trainer | ✅ Complete |
| **Data Pipeline** | Download → Clean → Filter → Dedup → Tokenize → Mix | ✅ Complete |
| **Training Loop** | AdamW, cosine LR, grad clip, checkpointing | ✅ Complete |
| **Distributed Training** | DeepSpeed integration | ✅ Complete |
| **Agent Framework** | Tool registry, chat template, agent loop | ✅ Complete |
| **RAG System** | Doc store, dense/sparse/hybrid retrieval, pipeline | ✅ Complete |
| **Safety System** | Toxicity, injection, PII, guardrails engine | ✅ Complete |
| **Structured Output** | JSON schema validation, constrained generation | ✅ Complete |
| **Serving** | Rate limiting, auth, metrics, production server | ✅ Complete |
| **SFT Data Pipeline** | Tool-use + RAG + safety + structured output data gen | ✅ Complete |
| **Evaluation** | MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval | ✅ Complete |
| **Export** | SafeTensors, ONNX, GGUF | ✅ Complete |
| **API Server** | OpenAI-compatible (FastAPI) | ✅ Complete |

### What's Next: Phase 3 — Training

To train the 1.5B PoC, we need:

| Requirement | Specification | Estimated Cost |
|-------------|--------------|---------------|
| **GPUs** | 8x H100 (or equivalent) | $2.50-4.00/GPU/hr |
| **Training tokens** | 30B tokens (Chinchilla-optimal) | — |
| **Training time** | ~3 days on 8x H100 | ~$2,000-4,000 |
| **SFT data** | ~50K examples (our pipeline generates this) | — |
| **DPO data** | ~10K preference pairs | — |
| **Total budget** | | **~$5,000-15,000** |

### Recommended Compute Providers (2025-2026)

| Provider | H100/hr | Notes |
|----------|---------|-------|
| **Lambda Cloud** | ~$2.49 | Good availability |
| **RunPod** | ~$2.69 | Serverless GPU option |
| **CoreWeave** | ~$2.21 | Enterprise, great networking |
| **Vast.ai** | ~$2.00 | Marketplace, variable quality |
| **Google Cloud (A3)** | ~$3.50 | Managed, reliable |
| **AWS (p5)** | ~$4.00 | Most features, highest price |

---

## Appendix A: Key Papers

| Paper | Year | Contribution |
|-------|------|-------------|
| "Attention Is All You Need" | 2017 | Transformer architecture |
| "BERT: Pre-training of Deep Bidirectional Transformers" | 2018 | Masked language modeling |
| "Language Models are Unsupervised Multitask Learners" (GPT-2) | 2019 | Scaling decoder-only models |
| "Scaling Laws for Neural Language Models" (Kaplan) | 2020 | Neural scaling laws |
| "Training Compute-Optimal LLMs" (Chinchilla) | 2022 | Data-optimal scaling |
| "LLaMA: Open and Efficient Foundation Language Models" | 2023 | Efficient open LLMs |
| "Direct Preference Optimization" (DPO) | 2023 | Simplified alignment |
| "Mixtral of Experts" | 2024 | Sparse MoE for LLMs |
| "Mamba: Linear-Time Sequence Modeling" | 2024 | State-space alternative |
| "DeepSeek-V3 Technical Report" | 2024 | Cost-efficient MoE at scale |
| "Llama 3.1 Technical Report" | 2024 | Over-training scaling |
| "FlashAttention-2" | 2023 | IO-aware exact attention |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **BPE** | Byte Pair Encoding — tokenization algorithm |
| **CLM** | Causal Language Modeling — predict next token |
| **DDP** | Distributed Data Parallel — replicate model across GPUs |
| **DPO** | Direct Preference Optimization — alignment without reward model |
| **FSDP** | Fully Sharded Data Parallel — shard model across GPUs |
| **GQA** | Grouped-Query Attention — KV-head sharing |
| **KV-cache** | Cached key-value pairs for efficient autoregressive decoding |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning |
| **MFU** | Model FLOPs Utilization — actual vs theoretical compute |
| **MLM** | Masked Language Modeling — BERT-style pretraining |
| **MoE** | Mixture of Experts — sparse model activation |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **RMSNorm** | Root Mean Square Normalization |
| **RoPE** | Rotary Position Embedding |
| **SFT** | Supervised Fine-Tuning |
| **SSM** | State Space Model — linear-time sequence model |
| **SwiGLU** | Swish-Gated Linear Unit — activation function |
| **TTFT** | Time to First Token — latency metric |
| **ZeRO** | Zero Redundancy Optimizer — DeepSpeed memory optimization |

---

*This document is a living reference for the Yaya AI project. Last updated: March 2026.*
