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

| Checkpoint | Overall | Arithmetic | Word Problems | Facts | Identity | Reasoning | Language |
|---|---|---|---|---|---|---|---|
| Step 15k | 29% | 50% | 33% | 25% | 25% | 20% | 0% |
| Step 30k | 23% | 25% | 17% | 13% | 50% | 0% | 50% |
| DPO final | 26% | 38% | 50% | 13% | 50% | 0% | 0% |

## Usage

```python
import torch
from src.model.yaya_model import YayaForCausalLM
from src.utils.config import ModelConfig
from src.tokenizer.tokenizer import YayaTokenizer
from src.inference.generator import TextGenerator, GenerationConfig

# Load
tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
model = YayaTransformer(ModelConfig())
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
│   ├── model/          # Transformer architecture
│   ├── tokenizer/      # SentencePiece wrapper
│   ├── training/       # Trainer, DPO trainer
│   ├── inference/      # TextGenerator
│   └── data/           # Dataset classes
├── scripts/
│   ├── kaggle_run_sft.py       # Main Kaggle SFT runner (40k steps, DONE)
│   ├── kaggle_run_recovery.py  # Recovery fine-tune (anti-list-format)
│   ├── train_dpo.py            # DPO alignment (DONE)
│   ├── benchmark.py            # 35-question eval suite
│   ├── chat.py                 # CLI chat
│   ├── web_ui.py               # Gradio web UI
│   ├── quantize.py             # int8 quantization (492MB → 219MB)
│   └── update_dashboard.py     # Regenerate dashboard from benchmark data
├── configs/
│   ├── model/yaya_125m.yaml
│   └── training/milestones.yaml
└── docs/
    ├── dashboard.html           # Training progress dashboard
    └── benchmark_results.jsonl
```

## Notes

- Built entirely from scratch — no HuggingFace Transformers dependency
- Token format: `<|system|>`, `</|user|>`, `</|assistant|>`
- Checkpoints pushed to HF Hub every 90s during Kaggle training
- See `docs/dashboard.html` for training progress visualization
