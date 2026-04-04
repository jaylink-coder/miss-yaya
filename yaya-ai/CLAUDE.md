# Yaya AI — Claude Code Project Instructions

## Project overview
Yaya is an AI assistant built from scratch in PyTorch. Working directory: `yaya-ai/`.
**Primary model: yaya-125m** (129M params, 12 layers, hidden 768, GQA, RoPE, SwiGLU).
Training runs on Kaggle T4 GPU and persists checkpoints to HuggingFace Hub.

## Key commands
```
make chat            # Interactive CLI chat (auto-picks latest checkpoint)
make web-ui          # Gradio web chat at http://localhost:7860
make eval-math       # Math evaluation (20 questions, per-stage scores)
make dpo             # DPO alignment (auto-detects best SFT checkpoint)
make training-status # Show step/loss/checkpoint status
make benchmark       # Full 35-question benchmark across 6 categories
```

## Training (Kaggle)
Training runs via `scripts/kaggle_run_sft.py` on Kaggle.
- **Resume**: Just re-run the Kaggle notebook — auto-pulls latest checkpoint from HF Hub
- **Monitor**: `python scripts/phase_tester.py --once --token $HF_TOKEN`
- **Benchmark**: `python scripts/benchmark.py --checkpoint checkpoints/.../checkpoint-XXXXX`
- **Auto-test per phase**: `python scripts/phase_tester.py --watch --token $HF_TOKEN`

## Current training state (2026-04-04)
- **Step**: 32,500 / 40,000 (81%) — Phase 14 "Advanced I", running on Kaggle
- **Loss**: 2.76 (was 10.7 at start)
- **HF Hub**: `Jaylink-coder/yaya-125m` — checkpoints pushed every 90s
- **Latest local**: `checkpoints/yaya-125m-sft/checkpoint-00032500/`
- **Benchmark**: Step 15k=29%, Step 30k=23% (model in math rut — short Q&A fix pending)
- **After step 40,000**: DPO auto-launches, then pushes final checkpoint to Hub

## Model configs
- **Primary**: `configs/model/yaya_125m.yaml` — 129M params
- **Training**: `configs/training/sft_125m.yaml` — batch=4, grad_accum=8, lr=2e-5
- **Milestones**: `configs/training/milestones.yaml` — 16 SFT phases × 2500 steps + DPO

## Python environment
Always use `.venv/Scripts/python.exe`.
```
.venv/Scripts/python.exe scripts/...
```

## Launching training locally (IMPORTANT — use PowerShell, not nohup)
`nohup python.exe ... &` in WSL bash gets killed when the bash session ends.
```powershell
powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_125m.yaml --train_config configs/training/sft_125m.yaml' -RedirectStandardOutput 'logs\sft_125m.log' -RedirectStandardError 'logs\sft_125m_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$(\$proc.Id)"
```

## Important rules
- **All entry-point scripts must have UTF-8 reconfigure at the top** (Windows charmap issue):
  ```python
  if hasattr(sys.stdout, "reconfigure"):
      sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  ```
- **Generator returns response-only** — `TextGenerator.generate()` returns just the new tokens,
  do NOT slice by `len(prompt)`. This was fixed 2026-04-04.
- **Repetition penalty applies to response tokens only** — not prompt tokens (fixed 2026-04-04).
  Short answers like "4" or "Paris" were being suppressed before this fix.
- **Always use `repetition_penalty=1.5`** in GenerationConfig for inference
- **Training resume vs fresh start**: Use `--resume` for same dataset/config; `--pretrain_checkpoint` for new dataset
- **Never use `required=True`** for `--checkpoint` or `--model_config` — all scripts auto-detect
- **Always resume from latest checkpoint** — never restart from scratch

## Token format
- `SYSTEM_TOKEN = "<|system|>"`
- `USER_TOKEN = "</|user|>"` (closing-style, used as prefix)
- `ASSISTANT_TOKEN = "</|assistant|>"`
- Prompt: `tokenizer.format_chat(messages) + "\n" + ASSISTANT_TOKEN + "\n"`
  (The `"\n"` before ASSISTANT_TOKEN is required to match training format)

## Reasoning / tool-call tokens
- `<|think|>...</|think|>` — chain-of-thought reasoning block
- `<|calc|>EXPRESSION<|/calc|>=RESULT` — calculator tool call

## Data files
- `data/sft/yaya_reasoning_large.jsonl` — **~203K examples** (Kaggle-built: GSM8K + MetaMath + OpenHermes + Yaya)
- `data/sft/yaya_short_qa.jsonl` — **2,634 direct Q&A pairs** (arithmetic, facts, identity) — fixes math rut
- `data/sft/yaya_dpo_combined.jsonl` — **4,214 DPO preference pairs** (was 1,052)
- `data/sft/yaya_reasoning_combined.jsonl` — 3,455 local CoT examples
- `data/tokenizer/yaya_tokenizer.model` — SentencePiece tokenizer (vocab 32768)

## Known issues / recent fixes
- **2026-04-04**: Generator was returning full prompt+response; fixed to return response-only
- **2026-04-04**: Repetition penalty was penalizing prompt tokens; fixed to response-only
- **2026-04-04**: OpenHermes download was cutting off ~53K short; fixed to use streaming=True
- **2026-04-04**: Dataset cache check now forces rebuild when short_qa not yet included
- **2026-04-04**: GradScaler deprecated call fixed (torch.amp.GradScaler)
