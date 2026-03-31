# Yaya AI — Claude Code Project Instructions

## Project overview
Yaya is a multimodal AI model built from scratch in PyTorch. The working directory is `yaya-ai/`.
**Primary model: yaya-125m** (129M params, GPT-2 scale). The tiny model is retired.

## Key commands
```
make sft             # SFT training — yaya-125m on reasoning+math dataset (starts fresh or from pretrain)
make sft-resume      # Resume 125m SFT from latest checkpoint
make pretrain        # Pretrain yaya-125m from scratch
make eval            # Evaluate instruction-following (auto-detects best 125m checkpoint)
make eval-math       # Evaluate math ability
make eval-loop       # Eval + augment dataset for failures
make chat            # Interactive CLI chat (auto-picks best 125m checkpoint)
make chat-calc       # Chat with calculator tool enabled
make web-ui          # Gradio web chat
make dpo             # DPO alignment (auto-detects best 125m SFT checkpoint)
make training-status # Show step/loss/checkpoint status
make self-eval       # Broader benchmark
make self-improve    # Self-eval + data augmentation
make continuous-learn # Convert chat logs → SFT data
make reasoning       # Build reasoning dataset + train (starts from math-stage2)
make reasoning-resume # Resume reasoning training
```

## Model configs
- **Primary**: `configs/model/yaya_125m.yaml` — 129M params, 12 layers, hidden 768
- Archive: `configs/model/yaya_tiny.yaml` — 4.8M params (do not use for new runs)

## Python environment
Always use `.venv/Scripts/python.exe`.
```
.venv/Scripts/python.exe scripts/...
```

## Launching training (IMPORTANT — use PowerShell, not nohup)
`nohup python.exe ... &` in WSL bash gets killed when the bash session ends.
Use PowerShell Start-Process to launch as a native Windows process that survives session close:
```powershell
powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_125m.yaml --train_config configs/training/sft_125m.yaml' -RedirectStandardOutput 'logs\sft_125m.log' -RedirectStandardError 'logs\sft_125m_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$(\$proc.Id)"
```
`make sft` does this automatically.

## Important rules
- **All entry-point scripts must have UTF-8 reconfigure at the top** (Windows charmap issue):
  ```python
  if hasattr(sys.stdout, "reconfigure"):
      sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  ```
- **Always use `repetition_penalty=1.5`** in GenerationConfig for inference — prevents degenerate outputs
- **Training resume vs fresh start**: Use `--resume` for same dataset/config restarts; use `--pretrain_checkpoint` only when switching to a new dataset (weights-only, no optimizer state)
- **Never use `required=True`** for `--checkpoint` or `--model_config` — all scripts auto-detect
- **Always resume from latest checkpoint** when restarting the same run — never lose progress
- **Model config auto-detection**: scripts detect 125m vs tiny from checkpoint path (`"125m" in path`)

## Token format
- `SYSTEM_TOKEN = "<|system|>"`
- `USER_TOKEN = "</|user|>"`  (closing-style, used as prefix)
- `ASSISTANT_TOKEN = "</|assistant|>"`
- Prompt construction: `tokenizer.format_chat(history) + "\n" + ASSISTANT_TOKEN + "\n"`
  (The `"\n"` before ASSISTANT_TOKEN is required to match training format)

## Reasoning / tool-call tokens
- `<|think|>...</|think|>` — chain-of-thought reasoning block in assistant response
- `<|calc|>EXPRESSION<|/calc|>=RESULT` — calculator tool call; `ToolAugmentedGenerator` intercepts and evaluates live

## Current training state (2026-03-31)
- `checkpoints/yaya-125m-sft/` — primary target (run `make sft` to start)
- `checkpoints/yaya-tiny-math-stage2/checkpoint-00000500` — math curriculum base (used as reasoning start)
- `checkpoints/yaya-tiny-reasoning/` — reasoning run (8000 steps, tiny model — reference only)

## Data files
- `data/sft/yaya_reasoning_combined.jsonl` — **3,455 examples** (math + CoT + calc-tool) — primary training set
- `data/sft/yaya_reasoning.jsonl` — 41 CoT+calculator examples (generated)
- `data/sft/yaya_cot.jsonl` — 23 chain-of-thought examples
- `data/sft/math/yaya_math_combined.jsonl` — 3,391 math curriculum examples
- `data/sft/yaya_dpo_combined.jsonl` — 1,052 DPO preference pairs
- `data/tokenizer/yaya_tokenizer.model` — SentencePiece tokenizer (vocab 32768)

## Self-improvement pipeline
```
make sft → checkpoint → make eval → make eval-loop → augment data → make sft-resume
                     → make self-eval → make self-improve → augment data → make sft-resume
                     → make continuous-learn (from chat logs)
```
Data augmentation writes to `data/sft/yaya_reasoning_combined.jsonl`.
