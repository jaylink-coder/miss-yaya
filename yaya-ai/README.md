---
language: en
tags:
  - pytorch
  - transformer
  - causal-lm
  - from-scratch
license: apache-2.0
---

# Yaya-125M

A 129M parameter causal language model trained from scratch in PyTorch — no HuggingFace Transformers dependency.

## Model Details

| Property | Value |
|---|---|
| Parameters | 128,994,048 (~129M) |
| Architecture | Transformer (decoder-only) |
| Layers | 12 |
| Hidden size | 768 |
| FFN size | 3,072 |
| Attention heads | 12 (GQA: 4 KV heads) |
| Vocab size | 32,768 (SentencePiece) |
| Max sequence length | 1,024 |
| Positional encoding | RoPE |
| Activation | SwiGLU |
| Tied embeddings | Yes |

## Training

- **Hardware**: Kaggle T4 GPU (float16)
- **SFT**: 40,000 steps on ~205K examples (GSM8K + MetaMath + OpenHermes + custom Q&A)
- **DPO**: 2,500 steps on 4,225 preference pairs
- **Optimizer**: AdamW (lr=2e-5, β₁=0.9, β₂=0.95)
- **Batch size**: 32 effective (4 × 8 grad accum)

## Benchmark Results

The benchmark suite contains **135 questions** across 12 categories. It can be run in two modes:
- **Guarded**: runtime guards (calculator, identity, fact overrides) active — measures system-level performance
- **Model-only**: all guards disabled — measures raw model capability

```bash
python scripts/benchmark.py --checkpoint ... --dual    # side-by-side comparison
python scripts/benchmark.py --checkpoint ... --model-only  # raw model score
```

| Checkpoint | Guarded | Model-Only (est.) | Notes |
|---|---|---|---|
| Step 15k | 29% | ~29% | Early SFT, no guards yet |
| DPO final | 26% | ~26% | Before guards |
| Patch SFT + guards | ~95%+ | ~40-50% | Guards cover most questions |

## Usage

```python
import torch
from src.model.yaya_model import YayaForCausalLM
from src.utils.config import ModelConfig
from src.tokenizer.tokenizer import YayaTokenizer
from src.inference.generator import TextGenerator, GenerationConfig

# Load
tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
model = YayaForCausalLM(ModelConfig())
state = torch.load("checkpoint/model.pt", map_location="cpu")
model.load_state_dict(state["model"])
model.eval()

gen = TextGenerator(model, tokenizer)
cfg = GenerationConfig(max_new_tokens=200, temperature=0.7, repetition_penalty=1.5)

response = gen.generate("What is 2 + 2?", config=cfg)
print(response)  # "4"
```

## Repo Structure

```
yaya-ai/
├── src/
│   ├── model/          # Transformer architecture (GQA, RoPE, SwiGLU, MoE, LoRA, Vision)
│   ├── tokenizer/      # SentencePiece wrapper with special tokens
│   ├── training/       # Trainer, DPO, EWC, curriculum, reward model, online learning
│   ├── inference/      # TextGenerator with runtime guards
│   ├── agent/          # Tool-calling agent loop with persistent memory
│   ├── rag/            # Retrieval-augmented generation pipeline
│   ├── safety/         # Content filtering and toxicity detection
│   ├── serving/        # Production server with rate limiting, auth, metrics
│   └── data/           # Dataset classes and data processing
├── scripts/
│   ├── train_sft.py            # SFT training
│   ├── train_dpo.py            # DPO alignment
│   ├── benchmark.py            # 135-question eval suite (--model-only / --dual)
│   ├── chat.py / web_ui.py     # CLI and Gradio chat interfaces
│   └── quantize.py             # INT8 quantization
├── configs/
│   ├── model/                  # yaya_125m, yaya_350m, yaya_1b, yaya_1_5b, yaya_7b
│   └── training/               # SFT, DPO, LoRA configs per model scale
└── docs/
    ├── PRETRAINING_GUIDE.md    # How to pre-train before SFT
    ├── dashboard.html          # Training progress dashboard
    └── benchmark_results.jsonl
```

## Known Limitations

- **No pre-training**: The model was SFT-trained from random initialization (no pre-training on text corpus). Factual knowledge is limited to what appears in the ~205K SFT examples.
- **Short context window**: 1,024 tokens max — limits multi-turn chat, RAG, and long-form generation.
- **Runtime guard dependence**: Many benchmark answers are provided by regex-based runtime guards in the generator, not the model itself. Use `--model-only` to see true model capability.
- **Small scale**: At 129M parameters, the model is well below the ~1B threshold for general-purpose competence.
- **Vision not active**: Multimodal (vision encoder + projector) code exists but has never been trained.

## Scaling Roadmap

1. **Pre-train** yaya-125m on 2.5–5B tokens (Wikipedia + web text) — see `docs/PRETRAINING_GUIDE.md`
2. **Scale to yaya-350m** (config ready: `configs/model/yaya_350m.yaml`) with 2048 context
3. **Reduce guard lift** to <15 percentage points via better pre-training + SFT data
4. **Activate vision** pipeline for multimodal capabilities
5. **Scale to 1B+** with MoE and distributed training

## Notes

- Built entirely from scratch — no HuggingFace Transformers dependency
- Token format: `<|system|>`, `</|user|>`, `</|assistant|>`
- Checkpoints pushed to HF Hub every 90s during Kaggle training
- See `docs/dashboard.html` for training progress visualization
