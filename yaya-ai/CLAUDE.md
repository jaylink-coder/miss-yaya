# Yaya AI — Claude Code Project Instructions

## Project overview
Yaya is a multimodal AI model built from scratch in PyTorch. The working directory is `yaya-ai/`.

## Key commands
```
make sft-tiny-focused    # Resume focused SFT training (from sft-clean step 2000)
make eval-tiny           # Evaluate instruction-following (13 questions)
make eval-loop           # Eval + augment dataset for failures + retrain prompt
make chat-tiny           # Interactive CLI chat
make web-ui              # Gradio web chat
make dpo-tiny            # DPO alignment (auto-detects best SFT checkpoint)
make training-status     # Show log tail + checkpoint list
make self-eval           # Broader 16-question benchmark
make self-improve        # Self-eval + data augmentation
make continuous-learn    # Convert chat logs → SFT data
```

## Python environment
Always use `.venv/Scripts/python.exe` or activate `.venv` first.
```
.venv/Scripts/python.exe scripts/...
```

## Important rules
- **All entry-point scripts must have UTF-8 reconfigure at the top** (Windows charmap issue):
  ```python
  if hasattr(sys.stdout, "reconfigure"):
      sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  ```
- **Always use `repetition_penalty=1.5`** in GenerationConfig for inference — prevents degenerate "0000..." outputs
- **Training resumes from weights only** (`--pretrain_checkpoint`) not optimizer state (`--resume`) when switching configs
- **Never use `required=True`** for `--checkpoint` or `--model_config` — all scripts should auto-detect

## Token format
- `SYSTEM_TOKEN = "<|system|>"`
- `USER_TOKEN = "</|user|>"`  (closing-style, used as prefix)
- `ASSISTANT_TOKEN = "</|assistant|>"`
- Prompt construction: `tokenizer.format_chat(history) + ASSISTANT_TOKEN + "\n"`

## Current training state
- `checkpoints/yaya-tiny/checkpoint-00010000` — pretrain base
- `checkpoints/yaya-tiny-sft-clean/checkpoint-00002000` — SFT clean (starting point for focused)
- `checkpoints/yaya-tiny-sft-focused/` — in progress (focused SFT)
- `checkpoints/yaya-tiny-dpo/` — not yet created

## Data files
- `data/sft/yaya_instruct_clean.jsonl` — 12,101 clean training examples
- `data/sft/yaya_instruct_clean_eval.jsonl` — 366 eval examples
- `data/sft/yaya_dpo_combined.jsonl` — 1,052 DPO preference pairs
- `data/tokenizer/yaya_tokenizer.model` — SentencePiece tokenizer (vocab 32768)

## Self-improvement pipeline
```
Training → Checkpoint → eval_instruction.py → eval_loop.py → augment dataset → retrain
                     → self_eval.py → self_improve.py → augment dataset → retrain
                     → continuous_learn.py (from chat logs)
```
