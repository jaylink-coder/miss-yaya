# AI Pre-Development Master Plan
# Everything On The Table Before We Build

> **Project Scope:** Train from scratch — a multimodal, general-purpose AI system covering Language (LLM), Vision (CV), and Creative/Business applications.
>
> **Date:** March 2026

---

## Table of Contents

1. [Project Vision & Scope](#1-project-vision--scope)
2. [Critical Decisions To Make First](#2-critical-decisions-to-make-first)
3. [Architecture Design](#3-architecture-design)
4. [Data Strategy](#4-data-strategy)
5. [Compute & Hardware Infrastructure](#5-compute--hardware-infrastructure)
6. [Software Stack & Tooling](#6-software-stack--tooling)
7. [Team & Organization](#7-team--organization)
8. [Training Strategy](#8-training-strategy)
9. [Evaluation & Benchmarking](#9-evaluation--benchmarking)
10. [Deployment & Serving Infrastructure](#10-deployment--serving-infrastructure)
11. [Cost Estimates (Realistic)](#11-cost-estimates)
12. [Phased Development Roadmap](#12-phased-development-roadmap)
13. [Risks & Mitigation](#13-risks--mitigation)
14. [Legal, Licensing & Compliance](#14-legal-licensing--compliance)
15. [Decision Checklist](#15-decision-checklist)

---

## 1. Project Vision & Scope

### What We're Building

A **multimodal AI system** trained from scratch with three core capabilities:

```
┌──────────────────────────────────────────────────────────────┐
│                    MULTIMODAL AI SYSTEM                       │
├──────────────┬──────────────────┬────────────────────────────┤
│  LANGUAGE    │  VISION          │  CREATIVE / BUSINESS       │
│  (LLM Core) │  (Vision Encoder)│  (Domain Specialization)   │
├──────────────┼──────────────────┼────────────────────────────┤
│ • Text gen   │ • Image under-   │ • Content generation       │
│ • Reasoning  │   standing       │ • Business analytics       │
│ • Code gen   │ • Object detect  │ • Document processing      │
│ • Q&A        │ • Image gen      │ • Customer service         │
│ • Translation│ • Video analysis │ • Creative writing/art     │
└──────────────┴──────────────────┴────────────────────────────┘
```

### What "From Scratch" Means

Training from scratch means we:
- **Design our own model architecture** (transformer-based, with our own decisions on layers, heads, etc.)
- **Build or choose our own tokenizer** (BPE, SentencePiece, or custom)
- **Collect and curate our own training data** (trillions of tokens for text, billions of images)
- **Run pre-training from randomly initialized weights** (no starting from someone else's checkpoint)
- **Handle all post-training ourselves** (instruction tuning, RLHF, safety alignment)

### What It Does NOT Mean

- We don't need to invent new math — we'll use proven transformer architecture with our modifications
- We can still use open-source tools (PyTorch, tokenizer libraries, data processing tools)
- We can study and learn from published architectures (LLaMA, GPT, Gemini papers)

---

## 2. Critical Decisions To Make First

These decisions cascade through everything else. They must be locked before writing code.

### DECISION 1: Model Scale

| Scale Tier | Parameters | Training Tokens | GPU-Hours (est.) | Cost (est.) |
|-----------|-----------|----------------|-------------------|-------------|
| **Small (proof of concept)** | 1B–3B | 1–3T tokens | 10K–50K H100-hrs | $30K–$150K |
| **Medium (competitive)** | 7B–13B | 2–5T tokens | 100K–500K H100-hrs | $300K–$1.5M |
| **Large (frontier-class)** | 30B–70B | 5–15T tokens | 1M–5M H100-hrs | $3M–$15M |
| **Massive (GPT-4 class)** | 200B+ / MoE | 10T+ tokens | 10M+ H100-hrs | $50M–$100M+ |

**Recommendation:** Start with a **1B–3B parameter proof-of-concept**, validate architecture decisions, then scale to 7B–13B, then beyond.

### DECISION 2: Architecture Pattern

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| **Dense Transformer** | Every parameter active for every token (like LLaMA) | Simple, proven, easy to debug | Expensive at scale |
| **Mixture of Experts (MoE)** | Only subset of params active per token (like Mixtral, likely GPT-4) | More capacity per FLOP, cheaper inference | Harder to train, load balancing issues |
| **Hybrid (start dense, add MoE later)** | Begin dense, convert to MoE after validating | Best of both worlds | Added complexity in transition |

### DECISION 3: Multimodal Approach

Two proven methods for connecting vision to language:

**Method A: Unified Embedding Decoder Architecture**
```
Image → Vision Encoder → Linear Projection → [Image Tokens] + [Text Tokens] → LLM Decoder → Output
```
- Images converted to tokens with same embedding size as text
- Single decoder processes both modalities after concatenation
- **Easier to implement**, no LLM architecture changes needed
- Used by: LLaVA, Qwen-VL, Llama 3.2 Vision

**Method B: Cross-Modality Attention Architecture**
```
Image → Vision Encoder → Cross-Attention Layers inside LLM → Output
Text  → Text Embeddings ↗
```
- Image features injected via cross-attention in transformer layers
- **More computationally efficient** (doesn't overload input context with image tokens)
- Preserves text-only performance if LLM weights are frozen
- Used by: Flamingo, Aria, some NVLM variants

**Recommendation:** Start with **Method A** (unified embedding) — simpler to implement and debug. Consider Method B for v2.

### DECISION 4: Tokenizer Strategy

| Option | Description | Notes |
|--------|------------|-------|
| **BPE (Byte-Pair Encoding)** | Most common. Used by GPT, LLaMA. | Proven, fast, good compression |
| **SentencePiece (Unigram)** | Used by T5, mBART. Language-agnostic. | Good for multilingual |
| **Byte-level BPE** | Operates on raw bytes. Used by GPT-2+. | No unknown tokens, handles any input |
| **Custom multi-modal tokenizer** | Separate tokenizers for text + vision + audio | Allows modality-specific optimization |

**Key decisions:**
- **Vocabulary size:** 32K (small/fast) → 64K (balanced) → 128K+ (multilingual)
- **Languages supported:** English-only vs. multilingual (affects data needs 5–10×)
- **Special tokens:** Define tokens for modality boundaries, tool use, function calls, etc.

### DECISION 5: Vision Encoder

| Option | Description | Notes |
|--------|------------|-------|
| **Train from scratch (ViT)** | Build our own Vision Transformer | Full control, highest cost, needs huge image dataset |
| **Train from scratch (custom CNN)** | Custom convolutional backbone | Simpler for specific vision tasks |
| **Use pre-trained then unfreeze** | Start from CLIP/SigLIP, fine-tune during multimodal training | Pragmatic middle ground (most common even in "from scratch" projects) |

**Reality check:** Even frontier labs (Meta, Google) often use pre-trained vision encoders (CLIP, SigLIP) as starting points. Training a competitive vision encoder from scratch requires ~400M+ image-text pairs.

### DECISION 6: Training Precision

| Precision | Memory per Param | Speed | Quality |
|-----------|-----------------|-------|---------|
| **FP32** | 4 bytes | Baseline | Best (but unnecessary) |
| **BF16** | 2 bytes | ~2× faster | Excellent (industry standard) |
| **FP16** | 2 bytes | ~2× faster | Good (needs loss scaling) |
| **FP8** | 1 byte | ~4× faster | Good for inference, emerging for training |

**Recommendation:** **BF16 for training** (industry standard), FP8/INT4 for inference deployment.

---

## 3. Architecture Design

### 3.1 LLM Core Architecture

Based on modern transformer best practices (LLaMA 3, Mistral, Qwen patterns):

```
┌─────────────────────────────────────────┐
│            MODEL ARCHITECTURE            │
├─────────────────────────────────────────┤
│                                          │
│  Input → Tokenizer → Embedding Layer     │
│              ↓                           │
│  ┌─────────────────────────────┐        │
│  │   Transformer Block × N     │        │
│  │  ┌───────────────────────┐  │        │
│  │  │ RMSNorm (Pre-Norm)    │  │        │
│  │  │ Grouped-Query Attention│  │        │
│  │  │ + RoPE Positional Enc  │  │        │
│  │  │ + KV-Cache for Infer.  │  │        │
│  │  ├───────────────────────┤  │        │
│  │  │ RMSNorm (Pre-Norm)    │  │        │
│  │  │ SwiGLU Feed-Forward    │  │        │
│  │  └───────────────────────┘  │        │
│  └─────────────────────────────┘        │
│              ↓                           │
│  RMSNorm → Linear → Softmax → Output    │
│                                          │
└─────────────────────────────────────────┘
```

### Key Architecture Components

| Component | Recommended Choice | Why |
|-----------|-------------------|-----|
| **Normalization** | RMSNorm (pre-norm) | Faster than LayerNorm, more stable training |
| **Attention** | Grouped-Query Attention (GQA) | Balance between MHA quality and MQA speed. Reduces KV-cache memory. |
| **Positional Encoding** | RoPE (Rotary Position Embeddings) | Handles variable sequence lengths, good extrapolation |
| **Activation Function** | SwiGLU | Better than ReLU/GELU, proven in LLaMA/PaLM |
| **Attention Implementation** | FlashAttention-2/3 | 2–4× faster attention, memory efficient |
| **Bias terms** | No bias in attention/FFN | Simplifies architecture, matches LLaMA/Mistral |
| **Tied embeddings** | Optional (input = output embedding) | Saves parameters at small scale |

### Architecture Configurations by Scale

| Config | 1.5B | 7B | 13B | 30B | 70B |
|--------|------|-----|------|------|------|
| **Layers** | 24 | 32 | 40 | 48 | 80 |
| **Hidden dim** | 2048 | 4096 | 5120 | 6144 | 8192 |
| **Attention heads** | 16 | 32 | 40 | 48 | 64 |
| **KV heads (GQA)** | 4 | 8 | 8 | 8 | 8 |
| **FFN dim** | 5632 | 11008 | 13824 | 16384 | 28672 |
| **Vocab size** | 64K | 64K | 64K | 64K | 64K |
| **Context length** | 4K–8K | 8K–32K | 8K–32K | 32K–128K | 32K–128K |

### 3.2 Vision Encoder Architecture

```
Image (variable size)
    ↓
Resize/Pad to patches (e.g., 14×14 or 16×16 pixels per patch)
    ↓
Linear patch embedding (flatten patch → vector)
    ↓
+ Positional embeddings (2D sinusoidal or learned)
    ↓
Vision Transformer blocks × M
    ↓
Vision features (sequence of patch embeddings)
    ↓
Projection layer (map to LLM embedding dimension)
    ↓
→ Feed into LLM as visual tokens
```

| Component | Spec |
|-----------|------|
| **Patch size** | 14×14 pixels (standard) |
| **Image resolution** | Dynamic (224–1024+, with tiling for high-res) |
| **ViT size** | ViT-Large (300M) → ViT-Huge (600M+) |
| **Projection** | 2-layer MLP (vision dim → LLM hidden dim) |

### 3.3 Multimodal Fusion

```
                    ┌──────────────┐
     Text ────────→ │  Tokenizer   │──→ Text Embeddings ──┐
                    └──────────────┘                       │
                                                           ├──→ [Combined Sequence] ──→ LLM Decoder
                    ┌──────────────┐    ┌────────────┐    │
     Image ───────→ │ Vision Enc.  │──→ │ Projector  │──→ Image Embeddings
                    └──────────────┘    └────────────┘

Special tokens:  <image_start> [visual tokens...] <image_end> [text tokens...]
```

---

## 4. Data Strategy

### 4.1 Data Requirements (Training from Scratch)

This is the **most critical and often underestimated** part.

#### Text Data

| Dataset Type | Target Volume | Sources | Notes |
|-------------|--------------|---------|-------|
| **Web crawl (filtered)** | 5–10T tokens | CommonCrawl, custom crawls | Needs heavy filtering and dedup |
| **Books** | 100B–500B tokens | Public domain, licensed | High quality prose |
| **Academic papers** | 50B–200B tokens | arXiv, PubMed, Semantic Scholar | Technical knowledge |
| **Code** | 500B–1T tokens | GitHub (permissive licenses), StackOverflow | Programming capability |
| **Wikipedia** | ~4B tokens (all languages) | Wikimedia dumps | Factual knowledge backbone |
| **Conversational data** | 50B–200B tokens | Forums, Q&A sites, dialog datasets | Conversational ability |
| **Domain-specific (business)** | 50B–200B tokens | Financial reports, legal docs, enterprise data | Business specialization |
| **Math & reasoning** | 50B–100B tokens | Math datasets, proofs, textbooks | Reasoning capability |
| **Multilingual** | 1T–5T tokens (if multilingual) | Per-language web crawls | 5–10× data if multilingual |
| **Synthetic data** | 500B–2T tokens | Generated by existing models | Augment weak areas |

**Total target: 5–15 trillion tokens** for a competitive general-purpose model.

#### Image/Vision Data

| Dataset Type | Target Volume | Sources |
|-------------|--------------|---------|
| **Image-text pairs** | 1B–5B pairs | LAION, CC12M, custom crawls |
| **Classification images** | 100M–1B images | ImageNet, OpenImages, custom |
| **Object detection** | 10M–100M annotated images | COCO, OpenImages, custom |
| **Video frames** | 100M–1B frames | WebVid, custom |
| **Document images** | 10M–100M | PDFs, screenshots, scans |
| **Synthetic images** | 100M–1B | Generated for augmentation |

### 4.2 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA PIPELINE                              │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │  Crawl/  │   │  Filter  │   │  Dedup   │   │  Quality   │  │
│  │  Collect  │──→│  & Clean │──→│  (MinHash│──→│  Score     │  │
│  │          │   │          │   │  /Exact)  │   │  & Rank    │  │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘  │
│                                                      │          │
│                                               ┌──────▼───────┐  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐ │  Tokenize    │  │
│  │  Store   │◀──│  Version  │◀──│  Mix &   │◀│  & Format    │  │
│  │  (Final) │   │  (DVC)   │   │  Sample  │ │              │  │
│  └──────────┘   └──────────┘   └──────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Data Quality Pipeline (Critical)

| Step | What It Does | Tools |
|------|-------------|-------|
| **Language detection** | Filter to target languages | fastText language ID |
| **Deduplication** | Remove exact and near-duplicate documents | MinHash LSH, exact hash |
| **Quality filtering** | Score text quality (perplexity, grammar, coherence) | KenLM, custom classifiers |
| **Toxicity filtering** | Remove harmful/toxic content | Perspective API, custom classifiers |
| **PII removal** | Strip personal information | Regex + NER models |
| **Domain classification** | Tag documents by domain for balanced mixing | Fasttext classifiers |
| **Content filtering** | Remove low-value content (boilerplate, ads, navigation) | trafilatura, custom rules |

**Data quality > data quantity.** A well-filtered 3T token dataset will outperform a noisy 10T dataset.

### 4.4 Data Mixing Strategy

Not all data is equal. The training data mix ratio profoundly affects model behavior:

| Category | Suggested Mix % | Purpose |
|----------|----------------|---------|
| Web text (filtered) | 40–50% | General knowledge |
| Code | 10–15% | Coding + logical reasoning |
| Books + long-form | 10–15% | Coherent long-form generation |
| Academic/technical | 5–10% | Technical knowledge |
| Math & reasoning | 5–8% | Mathematical reasoning |
| Conversational | 5–8% | Dialog capability |
| Domain-specific | 5–10% | Business/enterprise specialization |
| Multilingual | 5–15% | Non-English capabilities |

This mix should be **tuned experimentally** with small-scale ablation runs.

---

## 5. Compute & Hardware Infrastructure

### 5.1 Compute Requirements by Scale

| Model Size | Min GPUs | Recommended GPUs | Training Time (est.) | Total GPU-Hours |
|-----------|---------|------------------|---------------------|----------------|
| **1.5B (PoC)** | 8× H100 | 32× H100 | 1–2 weeks | 5K–15K |
| **7B** | 64× H100 | 128–256× H100 | 2–4 weeks | 50K–200K |
| **13B** | 128× H100 | 256–512× H100 | 3–6 weeks | 150K–500K |
| **30B** | 256× H100 | 512–1024× H100 | 4–8 weeks | 500K–1.5M |
| **70B** | 512× H100 | 1024–2048× H100 | 6–12 weeks | 1.5M–5M |

**Note:** These assume 2–5T training tokens. Actual time depends on batch size, sequence length, and cluster efficiency.

### 5.2 Hardware Shopping List (for a serious 7B–13B training cluster)

| Component | Spec | Quantity | Est. Cost |
|-----------|------|----------|-----------|
| **GPUs** | NVIDIA H100 80GB SXM5 | 128–256 | $25K–$35K each |
| **GPU Servers** | 8× H100 per node (DGX H100 or similar) | 16–32 nodes | $300K–$400K/node |
| **Networking** | InfiniBand NDR 400Gbps | Full bisection | $5K–$10K per port |
| **Network switches** | InfiniBand leaf/spine | 4–8 switches | $50K–$100K each |
| **Storage (fast)** | NVMe SSD array or parallel FS (Lustre/GPFS/Weka) | 500TB–1PB | $200K–$500K |
| **Storage (bulk)** | Object storage for raw data | 2–10PB | $100K–$300K |
| **CPU servers** | Data preprocessing nodes | 10–20 nodes | $10K–$20K each |
| **Cooling** | Liquid cooling (rear-door or direct-to-chip) | Full cluster | $200K–$500K |
| **Power** | UPS + PDUs | 500kW–2MW capacity | $100K–$300K |
| **Rack infrastructure** | Racks, cabling, management | Full cluster | $100K–$200K |

**Total on-prem estimate (128 H100s):** ~$8M–$15M

### 5.3 Cloud Alternative (Recommended to Start)

| Provider | GPU | On-Demand $/hr | Spot/Preemptible $/hr | Notes |
|----------|-----|---------------|----------------------|-------|
| **AWS** | H100 (p5.48xlarge, 8× H100) | ~$98/hr | ~$40–$60/hr | Broadest ecosystem |
| **GCP** | H100, A3 instances | ~$98/hr | ~$30–$50/hr | TPU alternative available |
| **Azure** | H100 (ND H100 v5) | ~$98/hr | ~$40–$60/hr | Best if using Microsoft tools |
| **Lambda Labs** | H100 | ~$2.49/GPU/hr | N/A | Simpler, AI-focused |
| **CoreWeave** | H100 | ~$2.06/GPU/hr | Reserved pricing | AI-native cloud |
| **RunPod** | H100 | ~$2.39/GPU/hr | ~$1.74/GPU/hr | Budget-friendly |
| **Vast.ai** | H100 | Variable | Auction-based | Cheapest, less reliable |

**Cloud cost estimate for 7B model training (128 H100s, 4 weeks):**
- On-demand: 128 × $2.50/hr × 24hrs × 28 days = ~$215K
- Spot/reserved: ~$100K–$150K
- Add 2–3× for failed runs, experiments, and iteration = **$300K–$650K total**

### 5.4 Recommended Approach: Phased Infrastructure

```
Phase 1 (Proof of Concept):   8–32 H100s on cloud (Lambda/RunPod/CoreWeave)
Phase 2 (7B Training):        128–256 H100s on cloud (reserved instances)
Phase 3 (13B+ Training):      256–512+ H100s (negotiate enterprise cloud deal or build on-prem)
Phase 4 (Production):         Inference cluster (can be smaller GPUs, quantized models)
```

---

## 6. Software Stack & Tooling

### 6.1 Complete Software Stack

```
┌─────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                      │
│  Custom API, Web UI, SDK, Agent Framework                │
├─────────────────────────────────────────────────────────┤
│                    SERVING LAYER                         │
│  vLLM, TensorRT-LLM, Triton Server, FastAPI             │
├─────────────────────────────────────────────────────────┤
│                    MLOps LAYER                           │
│  MLflow, W&B, Docker, Kubernetes, GitHub Actions         │
├─────────────────────────────────────────────────────────┤
│                    TRAINING LAYER                        │
│  PyTorch 2.x, DeepSpeed ZeRO, Megatron-LM, FSDP        │
│  FlashAttention-2, Torch Compile, Mixed Precision        │
├─────────────────────────────────────────────────────────┤
│                    DATA LAYER                            │
│  Apache Spark/Dask, HuggingFace Datasets, DVC           │
│  MinHash dedup, fastText, KenLM, custom filters          │
├─────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                   │
│  Linux (Ubuntu 22.04/24.04), CUDA 12.x, NCCL            │
│  InfiniBand drivers, Container runtime                    │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Detailed Tool Selection

| Category | Primary Choice | Alternative | Why |
|----------|---------------|-------------|-----|
| **Framework** | PyTorch 2.x | JAX | PyTorch has largest ecosystem, best debugging |
| **Distributed Training** | DeepSpeed ZeRO-3 + Megatron-LM | PyTorch FSDP | Best scale performance; Megatron for tensor parallelism |
| **Fast Attention** | FlashAttention-3 | xformers | 2–4× attention speedup, memory efficient |
| **Tokenizer** | SentencePiece (BPE) | tiktoken, HF tokenizers | Battle-tested, supports multilingual |
| **Experiment Tracking** | Weights & Biases | MLflow | Best visualization, team collaboration |
| **Data Processing** | Apache Spark + custom Python | Dask, Ray | Handles petabyte-scale processing |
| **Data Versioning** | DVC + Git | LakeFS, Delta Lake | Simple, integrates with Git |
| **Containerization** | Docker | Singularity/Apptainer | Industry standard |
| **Orchestration** | Kubernetes + SLURM | Kubernetes only | SLURM for training jobs, K8s for serving |
| **CI/CD** | GitHub Actions | GitLab CI | Widely used, good integrations |
| **Model Registry** | W&B Artifacts + HuggingFace Hub | MLflow Registry | Easy sharing and versioning |
| **Evaluation** | lm-evaluation-harness | custom | Standard LLM benchmark suite |
| **RLHF** | TRL (Transformers RL) | OpenRLHF, DeepSpeed-Chat | HuggingFace ecosystem, well-maintained |
| **Serving (LLM)** | vLLM | TensorRT-LLM | PagedAttention, easy to use, fast |
| **Serving (Vision)** | Triton Inference Server | BentoML | Multi-model, dynamic batching |

### 6.3 Key Libraries to Install

```
# Core training
pytorch >= 2.3
deepspeed >= 0.14
flash-attn >= 2.5
transformers >= 4.40
tokenizers >= 0.19
sentencepiece >= 0.2

# Data processing
datasets >= 2.18
apache-spark >= 3.5
trafilatura >= 1.8
fasttext
pycld3

# Experiment tracking
wandb
mlflow

# Evaluation
lm-eval >= 0.4

# RLHF / Alignment
trl >= 0.8

# Serving
vllm >= 0.4
fastapi
uvicorn

# Vision
torchvision >= 0.18
timm >= 0.9
pillow
opencv-python

# Infrastructure
docker
kubernetes (client)
```

---

## 7. Team & Organization

### 7.1 Minimum Viable Team (Phase 1: Proof of Concept)

| Role | Count | Responsibility | Salary Range |
|------|-------|---------------|-------------|
| **Lead AI Researcher** | 1 | Architecture design, training strategy, research decisions | $180K–$300K |
| **ML Engineer (Training)** | 2 | Implement model, training loop, distributed training | $150K–$250K |
| **Data Engineer** | 1–2 | Data pipeline, crawling, cleaning, storage | $120K–$200K |
| **Infrastructure/MLOps** | 1 | GPU cluster setup, monitoring, CI/CD | $130K–$200K |
| **Total Phase 1** | **5–6 people** | | **$700K–$1.2M/year** |

### 7.2 Full Team (Phase 2–3: Production Model)

| Role | Count | Responsibility |
|------|-------|---------------|
| **Head of AI / CTO** | 1 | Overall technical strategy |
| **AI Researchers** | 2–4 | Architecture, scaling laws, novel techniques |
| **ML Engineers (Training)** | 3–5 | Model implementation, training at scale |
| **ML Engineers (Inference)** | 2–3 | Optimization, quantization, serving |
| **Data Engineers** | 3–5 | Data pipeline at scale |
| **Data Annotators / QA** | 5–10 | Labeling, RLHF data, quality assurance |
| **Infrastructure / SRE** | 2–3 | Cluster management, reliability |
| **MLOps Engineers** | 1–2 | CI/CD, deployment automation |
| **Safety / Alignment** | 1–2 | Red-teaming, safety evaluation |
| **Product / PM** | 1–2 | Product direction, user research |
| **Total Phase 2–3** | **20–35 people** | |

### 7.3 Key Skills Needed

- **Distributed systems** — Someone who understands multi-node GPU training, NCCL, network optimization
- **Transformer architecture expertise** — Deep understanding of attention, scaling laws, training dynamics
- **Large-scale data engineering** — Experience with petabyte-scale data processing
- **CUDA/GPU optimization** — Custom kernels, memory optimization, profiling
- **MLOps at scale** — Kubernetes, monitoring, automated pipelines
- **Safety and alignment** — RLHF, red-teaming, constitutional AI approaches

---

## 8. Training Strategy

### 8.1 Training Phases

```
Phase 1: PRE-TRAINING (Text Only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal: Learn language, knowledge, reasoning from massive text corpus
  Data: 2–15T tokens of filtered web text, books, code, etc.
  Duration: Weeks to months
  Cost: 70–80% of total training budget

      ↓

Phase 2: MULTIMODAL PRE-TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal: Connect vision encoder to language model
  Step 2a: Train projector only (LLM frozen, vision encoder frozen)
    Data: 500M–2B image-text pairs
    Duration: Days
  Step 2b: Unfreeze LLM, joint training
    Data: High-quality image-text data
    Duration: Days to weeks

      ↓

Phase 3: SUPERVISED FINE-TUNING (SFT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal: Teach the model to follow instructions
  Data: 1M–10M instruction-response pairs (text + multimodal)
  Duration: Days
  Includes: Chat, Q&A, coding, analysis, creative tasks

      ↓

Phase 4: ALIGNMENT (RLHF / DPO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal: Align model with human preferences, improve safety
  Method: RLHF (reward model + PPO) or DPO (Direct Preference Optimization)
  Data: 100K–1M human preference comparisons
  Duration: Days to weeks

      ↓

Phase 5: DOMAIN SPECIALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal: Specialize for business/enterprise/creative use cases
  Data: Domain-specific instruction data
  Duration: Days
  May include: Tool use, function calling, RAG integration
```

### 8.2 Training Hyperparameters (Starting Point)

| Parameter | Small (1.5B) | Medium (7B) | Large (13B+) |
|-----------|-------------|-------------|-------------|
| **Batch size (tokens)** | 2M–4M | 4M–8M | 4M–16M |
| **Learning rate (peak)** | 3e-4 | 3e-4 | 1.5e-4 |
| **LR schedule** | Cosine with warmup | Cosine with warmup | Cosine with warmup |
| **Warmup steps** | 2000 | 2000 | 2000 |
| **Weight decay** | 0.1 | 0.1 | 0.1 |
| **Gradient clipping** | 1.0 | 1.0 | 1.0 |
| **Optimizer** | AdamW (β1=0.9, β2=0.95) | AdamW | AdamW |
| **Dropout** | 0.0 (modern practice) | 0.0 | 0.0 |
| **Sequence length** | 4096–8192 | 4096–8192 | 4096–8192 |

### 8.3 Distributed Training Configuration

For a 7B model on 128 H100s (16 nodes × 8 GPUs):

| Parameter | Setting |
|-----------|---------|
| **Data parallelism** | 16-way (across nodes) |
| **Tensor parallelism** | 4-way (within node, NVLink) |
| **Pipeline parallelism** | 2-way (if needed) |
| **ZeRO stage** | Stage 1 or 2 (with tensor parallel) |
| **Activation checkpointing** | Selective (every other layer) |
| **Communication backend** | NCCL over InfiniBand |
| **Micro batch size** | 1–2 per GPU |
| **Gradient accumulation** | Adjusted to hit global batch size |

### 8.4 Critical Training Practices

- **Checkpointing:** Save every 500–1000 steps. Training crashes WILL happen.
- **Monitoring:** Track loss, gradient norms, learning rate, throughput (tokens/sec), GPU utilization in real-time via W&B.
- **Loss spikes:** Will happen. Have protocols to roll back to last good checkpoint and reduce learning rate.
- **Data curriculum:** May change data mix ratios during training (e.g., more code/math in later stages).
- **Ablation runs:** Do small-scale (1B param, 100B token) ablation experiments FIRST to validate architecture and data choices.

---

## 9. Evaluation & Benchmarking

### 9.1 Benchmark Suite

| Benchmark | What It Tests | Target (competitive 7B) |
|-----------|--------------|------------------------|
| **MMLU** | World knowledge (57 subjects) | 60–70% |
| **HellaSwag** | Common sense reasoning | 75–85% |
| **ARC-Challenge** | Science reasoning | 55–65% |
| **TruthfulQA** | Truthfulness | 45–55% |
| **GSM8K** | Math word problems | 40–60% |
| **HumanEval** | Code generation (Python) | 30–50% |
| **MBPP** | Basic Python programming | 40–60% |
| **MT-Bench** | Multi-turn conversation quality | 7.0–8.0 / 10 |
| **Winogrande** | Commonsense co-reference | 70–80% |
| **MATH** | Competition math | 20–40% |

### 9.2 Vision Benchmarks

| Benchmark | What It Tests |
|-----------|--------------|
| **VQAv2** | Visual question answering |
| **TextVQA** | Reading text in images |
| **DocVQA** | Document understanding |
| **MMMU** | Multimodal multi-discipline understanding |
| **ChartQA** | Chart/graph understanding |
| **OCRBench** | Optical character recognition |

### 9.3 Safety Evaluation

| Test | Purpose |
|------|---------|
| **ToxiGen** | Toxicity generation tendency |
| **BBQ** | Social bias measurement |
| **Red-teaming** | Adversarial prompt testing (manual + automated) |
| **Jailbreak resistance** | Testing safety guardrails |
| **Hallucination rate** | Factual accuracy on known questions |

### 9.4 Internal Evaluation (Continuous)

- **Perplexity on held-out data** — tracked during training
- **Few-shot performance** — periodic evaluation during training
- **Human evaluation** — regular manual quality checks
- **A/B testing** — compare checkpoints and configurations

---

## 10. Deployment & Serving Infrastructure

### 10.1 Inference Optimization Pipeline

```
Trained Model (BF16)
    ↓
Quantization (GPTQ / AWQ → INT4 or FP8)
    → 75–80% memory reduction, <2% quality loss
    ↓
Optimized Serving Engine
    → vLLM (PagedAttention, continuous batching)
    → OR TensorRT-LLM (NVIDIA optimized)
    ↓
API Layer (FastAPI / gRPC)
    → Authentication, rate limiting, logging
    ↓
Load Balancer + Auto-Scaling (Kubernetes)
    → Scale based on request volume
    ↓
CDN + Caching (for static/repeated queries)
```

### 10.2 Serving Infrastructure

| Component | Tool | Purpose |
|-----------|------|---------|
| **LLM Serving** | vLLM | High-throughput LLM inference |
| **Vision Serving** | Triton Inference Server | Multi-model serving |
| **API Gateway** | Kong / Nginx | Routing, auth, rate limiting |
| **Container Orchestration** | Kubernetes | Auto-scaling, health checks |
| **Monitoring** | Prometheus + Grafana | Latency, throughput, errors |
| **Logging** | ELK Stack / Loki | Request logging, debugging |
| **Queue** | Redis / RabbitMQ | Request buffering |

### 10.3 Inference Hardware

| Tier | Hardware | Use Case |
|------|----------|----------|
| **High throughput** | H100 / A100 | High-volume production |
| **Cost-efficient** | L40S / A10G | Medium traffic |
| **Edge / demo** | RTX 4090 / quantized on CPU | Low traffic, demos |

### 10.4 Latency Targets

| Metric | Target |
|--------|--------|
| **Time to first token (TTFT)** | < 500ms |
| **Token generation speed** | 30–80 tokens/sec per request |
| **Image processing** | < 2 seconds |
| **API response (short query)** | < 3 seconds |
| **Concurrent users** | Scale to 1000+ |

---

## 11. Cost Estimates

### 11.1 Total Budget Breakdown (Building a Competitive 7B Multimodal Model)

| Category | Low Estimate | High Estimate | Notes |
|----------|-------------|---------------|-------|
| **Compute (pre-training)** | $200K | $1M | Cloud GPU rental, includes failed runs |
| **Compute (fine-tuning + RLHF)** | $30K | $150K | Smaller scale than pre-training |
| **Compute (experiments/ablations)** | $50K | $200K | Critical for architecture decisions |
| **Data acquisition** | $20K | $200K | Crawling, storage, licensed datasets |
| **Data labeling (SFT + RLHF)** | $50K | $300K | Human annotators for instructions & preferences |
| **Team (1 year, 5–6 people)** | $700K | $1.5M | Salaries, benefits |
| **Infrastructure & tools** | $20K | $100K | W&B, cloud services, storage |
| **Legal & compliance** | $10K | $50K | Data licensing, IP review |
| **TOTAL (Year 1)** | **$1.1M** | **$3.5M** | For a competitive 7B multimodal model |

### 11.2 Cost by Phase

| Phase | % of Budget | Duration |
|-------|------------|----------|
| **Data collection & prep** | 15–20% | Months 1–4 |
| **Architecture research & ablations** | 10–15% | Months 2–5 |
| **Pre-training** | 35–45% | Months 5–8 |
| **Multimodal training** | 5–10% | Month 8–9 |
| **SFT + Alignment** | 5–10% | Months 9–10 |
| **Evaluation + iteration** | 5–10% | Months 10–11 |
| **Deployment** | 5–10% | Months 11–12 |

### 11.3 Scaling Cost (If We Go Bigger)

| Model | Compute Only | Total (w/ team & data) |
|-------|-------------|----------------------|
| **1.5B (PoC)** | $30K–$150K | $300K–$800K |
| **7B** | $300K–$1.5M | $1.1M–$3.5M |
| **13B** | $1M–$5M | $3M–$8M |
| **30B** | $3M–$15M | $8M–$25M |
| **70B** | $10M–$50M | $20M–$70M |

### 11.4 Ongoing Costs (Post-Launch, Monthly)

| Item | Monthly Cost |
|------|-------------|
| **Inference compute** | $10K–$200K (depends on traffic) |
| **Storage** | $2K–$20K |
| **Monitoring & tools** | $1K–$5K |
| **Team (maintenance)** | $50K–$150K |
| **Data refresh & retraining** | $10K–$50K |
| **Total monthly** | **$75K–$425K** |

---

## 12. Phased Development Roadmap

### Phase 0: Foundation (Weeks 1–4)
```
□ Lock all critical decisions (Section 2)
□ Hire/assemble core team (5–6 people minimum)
□ Set up development environment (Git, CI/CD, W&B, Docker)
□ Set up cloud GPU access (Lambda/CoreWeave/AWS)
□ Set up data storage (S3/GCS + fast scratch storage)
□ Begin data collection & crawling pipeline
```

### Phase 1: Data Engine (Weeks 3–12)
```
□ Build data crawling infrastructure
□ Implement filtering pipeline (dedup, quality, toxicity, PII)
□ Process and tokenize initial text corpus (1T+ tokens)
□ Collect image-text pairs (500M+)
□ Build data loading infrastructure (efficient streaming)
□ Create held-out evaluation sets
□ Implement data mixing and sampling strategy
```

### Phase 2: Architecture & Small-Scale Validation (Weeks 6–14)
```
□ Implement model architecture in PyTorch
□ Build tokenizer (train BPE on our data)
□ Implement training loop with DeepSpeed/FSDP
□ Train 150M–300M parameter ablation models
□ Test different architecture choices (GQA vs MHA, SwiGLU vs GELU, etc.)
□ Validate data mix ratios with small models
□ Select final architecture configuration
□ Build evaluation pipeline (lm-evaluation-harness)
```

### Phase 3: 1.5B Proof of Concept (Weeks 12–18)
```
□ Train 1.5B parameter model on 500B–1T tokens
□ Evaluate on standard benchmarks
□ Identify weaknesses and adjust data mix
□ Validate distributed training infrastructure at medium scale
□ Begin instruction tuning experiments
□ Begin vision encoder integration experiments
```

### Phase 4: 7B Full Training (Weeks 16–28)
```
□ Scale up to 128–256 GPU cluster
□ Train 7B parameter model on 2–5T tokens
□ Monitor training 24/7 (loss, gradients, throughput)
□ Handle training instabilities (loss spikes, NaN detection)
□ Perform periodic evaluations during training
```

### Phase 5: Multimodal Integration (Weeks 26–32)
```
□ Train/integrate vision encoder
□ Train projection layer (LLM frozen)
□ Joint multimodal fine-tuning
□ Evaluate on vision-language benchmarks
```

### Phase 6: Alignment & Specialization (Weeks 30–38)
```
□ Collect instruction-tuning data (or generate synthetic)
□ Supervised fine-tuning (SFT)
□ Collect human preference data
□ RLHF or DPO alignment training
□ Safety red-teaming and iteration
□ Domain specialization (business, creative)
□ Tool use / function calling training
```

### Phase 7: Deployment & Launch (Weeks 36–44)
```
□ Quantize model (INT4/FP8)
□ Set up vLLM serving infrastructure
□ Build API layer (FastAPI)
□ Set up Kubernetes auto-scaling
□ Load testing and optimization
□ Set up monitoring (Prometheus/Grafana)
□ Beta testing with selected users
□ Public launch
```

### Summary Timeline

```
Month:  1    2    3    4    5    6    7    8    9   10   11
        ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Data    ████████████████████                              
Arch    ·····██████████·····                              
1.5B PoC     ·····████████··                              
7B Train          ·····████████████████                   
Multimodal                    ·····████████               
Alignment                          ·····████████         
Deploy                                   ·····████████   
```

**Total estimated timeline: 10–12 months** for a competitive 7B multimodal model from scratch.

---

## 13. Risks & Mitigation

| Risk | Severity | Probability | Mitigation |
|------|----------|------------|-----------|
| **Training instability (NaN, loss spikes)** | High | High | Frequent checkpointing, gradient clipping, learning rate warmup, monitoring |
| **Data quality issues** | High | High | Invest heavily in data filtering pipeline, continuous quality audits |
| **GPU failures during training** | Medium | High | Checkpoint every 500 steps, elastic training (handle node failures) |
| **Cost overrun** | High | Medium | Start small (1.5B PoC), validate before scaling. Set hard budget limits. |
| **Model underperforms benchmarks** | High | Medium | Extensive ablation studies, study published architectures, iterate on data mix |
| **Data licensing/legal issues** | High | Medium | Legal review of all data sources, use permissive-licensed data, document provenance |
| **Key person leaves** | High | Medium | Document everything, cross-train team, competitive compensation |
| **Compute availability** | Medium | Medium | Multi-cloud strategy, reserve capacity early, relationships with cloud providers |
| **Model generates harmful content** | High | Medium | Red-teaming, safety fine-tuning, output filters, content policy |
| **Competitor releases better model** | Medium | High | Focus on unique strengths (domain specialization, multimodal), iterate fast |
| **Scaling doesn't work as expected** | High | Low-Med | Follow published scaling laws, validate at each scale before investing in next |
| **Hallucination in production** | Medium | High | RAG integration, confidence scoring, user feedback loops |

---

## 14. Legal, Licensing & Compliance

### 14.1 Data Licensing

| Data Source | License Consideration |
|-------------|----------------------|
| **Web crawl** | robots.txt compliance, fair use arguments, regional laws vary |
| **GitHub code** | Must respect license (Apache, MIT = OK; GPL = careful) |
| **Books** | Public domain only, or negotiate licenses |
| **Academic papers** | Often CC-BY or open access, verify per source |
| **Images** | Creative Commons, fair use, or licensed stock |
| **Synthetic data** | Check if generation model's ToS allows training |

### 14.2 Regulatory Frameworks

| Framework | Region | Key Requirements |
|-----------|--------|-----------------|
| **EU AI Act** | Europe | Risk classification, transparency, data governance |
| **NIST AI RMF** | USA | Risk management framework (voluntary but influential) |
| **GDPR** | Europe | Data privacy, right to erasure, consent |
| **CCPA** | California | Consumer data privacy |
| **Industry-specific** | Varies | HIPAA (healthcare), SOC2 (enterprise), PCI-DSS (finance) |

### 14.3 IP Protection

- **Patent** — Novel architecture innovations, training methods
- **Trade secret** — Data mix ratios, training recipes, proprietary data
- **Copyright** — Model weights may or may not be copyrightable (evolving law)
- **Trademark** — Model name, brand

---

## 15. Decision Checklist

Before writing a single line of training code, all of these must be answered:

### Architecture
- [ ] Model scale (start with 1.5B PoC? directly to 7B?)
- [ ] Dense vs MoE
- [ ] Number of layers, hidden dim, attention heads
- [ ] GQA configuration (how many KV heads)
- [ ] Context length target
- [ ] Activation function (SwiGLU recommended)
- [ ] Normalization (RMSNorm recommended)
- [ ] Positional encoding (RoPE recommended)

### Tokenizer
- [ ] Algorithm (BPE recommended)
- [ ] Vocabulary size (32K / 64K / 128K)
- [ ] Languages to support
- [ ] Special tokens definition (modality markers, tool use, etc.)
- [ ] Train tokenizer on our data or use existing?

### Multimodal
- [ ] Vision encoder approach (train from scratch vs. pre-trained)
- [ ] Fusion method (unified embedding vs. cross-attention)
- [ ] Image resolution and patching strategy
- [ ] Projection layer design
- [ ] Training phases for multimodal

### Data
- [ ] Target corpus size (tokens)
- [ ] Data sources approved (legal review)
- [ ] Quality filtering pipeline designed
- [ ] Data mix ratios defined
- [ ] Held-out evaluation sets created
- [ ] Data loading infrastructure built
- [ ] Labeling strategy for SFT/RLHF

### Compute
- [ ] Cloud vs. on-prem decision
- [ ] Cloud provider selected
- [ ] GPU type and count determined
- [ ] Budget approved with contingency
- [ ] Storage solution selected
- [ ] Networking verified (InfiniBand available?)

### Team
- [ ] Core team hired/assembled
- [ ] Roles and responsibilities defined
- [ ] Communication channels set up
- [ ] On-call rotation for training runs

### Tooling
- [ ] Framework (PyTorch) + version locked
- [ ] Distributed training stack chosen (DeepSpeed / Megatron / FSDP)
- [ ] Experiment tracking set up (W&B)
- [ ] CI/CD pipeline configured
- [ ] Evaluation harness ready

### Legal
- [ ] Data licensing reviewed
- [ ] Regulatory requirements identified
- [ ] IP strategy defined
- [ ] Terms of service drafted

---

## Next Steps

**Immediate actions to start the project:**

1. **Lock the model scale decision** — I recommend starting with 1.5B PoC
2. **Begin data collection** — this is the longest lead time item
3. **Implement the base architecture** — start coding the model in PyTorch
4. **Set up cloud GPU access** — get accounts on Lambda/CoreWeave/AWS
5. **Start small ablation experiments** — validate every decision at small scale first

---

*This plan is a living document. Update it as decisions are made and circumstances change.*
