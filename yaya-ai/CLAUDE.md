# Yaya AI — Claude Code Project Instructions

## Project overview
Yaya is a multimodal AI model built from scratch in PyTorch. The working directory is `yaya-ai/`.

## Key commands
```
make sft-tiny-filtered   # Resume/start filtered SFT training (chess-free, 12.7k examples)
make eval-filtered       # Evaluate latest filtered checkpoint
make eval-tiny           # Evaluate instruction-following (13 questions, auto-detects best)
make eval-loop           # Eval + augment dataset for failures + retrain prompt
make chat-tiny           # Interactive CLI chat
make web-ui              # Gradio web chat
make dpo-tiny            # DPO alignment (auto-detects best SFT checkpoint)
make training-status     # Show step/loss/checkpoint status (scripts/training_status.py)
make self-eval           # Broader 16-question benchmark
make self-improve        # Self-eval + data augmentation
make continuous-learn    # Convert chat logs → SFT data
```

## Python environment
Always use `.venv/Scripts/python.exe` or activate `.venv` first.
```
.venv/Scripts/python.exe scripts/...
```

## Launching training (IMPORTANT — use PowerShell, not nohup)
`nohup python.exe ... &` in WSL bash gets killed when the bash session ends.
Use PowerShell Start-Process to launch as a native Windows process that survives session close:
```powershell
powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/sft_tiny_filtered.yaml --pretrain_checkpoint checkpoints/yaya-tiny-sft-clean/checkpoint-00002000' -RedirectStandardOutput 'logs\sft_filtered.log' -RedirectStandardError 'logs\sft_filtered_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$(\$proc.Id)"
```
The `make sft-tiny-filtered` target does this automatically.

## Important rules
- **All entry-point scripts must have UTF-8 reconfigure at the top** (Windows charmap issue):
  ```python
  if hasattr(sys.stdout, "reconfigure"):
      sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  ```
- **Always use `repetition_penalty=1.5`** in GenerationConfig for inference — prevents degenerate "0000..." outputs
- **Training resume vs fresh start**: Use `--resume` for same dataset/config restarts; use `--pretrain_checkpoint` only when switching to a new dataset (weights-only, no optimizer state)
- **Never use `required=True`** for `--checkpoint` or `--model_config` — all scripts should auto-detect
- **Always resume from latest checkpoint** when restarting the same run — never lose progress

## Token format
- `SYSTEM_TOKEN = "<|system|>"`
- `USER_TOKEN = "</|user|>"`  (closing-style, used as prefix)
- `ASSISTANT_TOKEN = "</|assistant|>"`
- Prompt construction: `tokenizer.format_chat(history) + "\n" + ASSISTANT_TOKEN + "\n"`
  (The `"\n"` before ASSISTANT_TOKEN is required to match training format: `user_content\n</|assistant|>`)
  Server.py already does this correctly. Any script using the old format (without leading `\n`) is a bug.

## Current training state
- `checkpoints/yaya-tiny/checkpoint-00010000` — pretrain base
- `checkpoints/yaya-tiny-sft-clean/checkpoint-00002000` — SFT clean (starting point for filtered)
- `checkpoints/yaya-tiny-sft-filtered/` — IN PROGRESS (PID 16588, chess-free 12.7k dataset)
- `checkpoints/yaya-tiny-sft-focused/` — KILLED (dataset was 60% chess, unusable)
- `checkpoints/yaya-tiny-dpo/` — not yet created

## Data files
- `data/sft/yaya_instruct_filtered.jsonl` — 12,683 examples (1.6% chess — current training set)
- `data/sft/yaya_instruct_filtered_eval.jsonl` — 164 eval examples (6% chess)
- `data/sft/yaya_instruct_clean.jsonl` — 12,290 old clean examples (59.8% chess — do not use)
- `data/sft/yaya_instruct_backup.jsonl` — 19,023 external examples (source for filtered set)
- `data/sft/yaya_qa_sft_new.jsonl` — 71 high-quality Q&A examples (incorporated into filtered)
- `data/sft/yaya_dpo_combined.jsonl` — 1,052 DPO preference pairs
- `data/tokenizer/yaya_tokenizer.model` — SentencePiece tokenizer (vocab 32768)

## Self-improvement pipeline
```
Training → Checkpoint → eval_instruction.py → eval_loop.py → augment dataset → retrain
                     → self_eval.py → self_improve.py → augment dataset → retrain
                     → continuous_learn.py (from chat logs)
```
Data augmentation writes to `data/sft/yaya_instruct_filtered.jsonl` (current training set).
