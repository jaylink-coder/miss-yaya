"""HuggingFace SFT Fine-Tuning Script for Yaya.

Fine-tunes a pretrained HuggingFace model (e.g., Qwen 2.5 1.5B) on Yaya's
SFT data using either full fine-tuning or LoRA/QLoRA.

Usage:
    # LoRA (default, works on single GPU with 16GB+ VRAM)
    python scripts/finetune_hf.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --data_dir data/sft \
        --output_dir outputs/yaya-sft \
        --method lora

    # Full fine-tuning (needs more VRAM, better quality)
    python scripts/finetune_hf.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --data_dir data/sft \
        --output_dir outputs/yaya-sft \
        --method full

    # QLoRA (4-bit, works on 8GB VRAM)
    python scripts/finetune_hf.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --data_dir data/sft \
        --output_dir outputs/yaya-sft \
        --method qlora

Requirements:
    pip install transformers datasets peft trl bitsandbytes accelerate
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_sft_dataset(data_dir: str):
    """Load SFT data from JSONL files into HuggingFace Dataset format."""
    from datasets import Dataset

    train_path = os.path.join(data_dir, "yaya_sft_train.jsonl")
    eval_path = os.path.join(data_dir, "yaya_sft_eval.jsonl")

    def read_jsonl(path):
        records = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        return records

    train_data = read_jsonl(train_path)
    eval_data = read_jsonl(eval_path)

    if not train_data:
        print(f"ERROR: No training data found at {train_path}")
        print("Run: python scripts/generate_training_data.py")
        sys.exit(1)

    print(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None
    return train_ds, eval_ds


def format_messages_for_training(example, tokenizer):
    """Format a single example's messages into the model's chat template."""
    messages = example["messages"]
    # Use the tokenizer's built-in chat template if available
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: manual ChatML format
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return {"text": text}


def create_model_and_tokenizer(base_model: str, method: str, bf16: bool = True):
    """Load model and tokenizer with appropriate configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {base_model}")
    print(f"Method: {method}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "use_cache": False,  # Disable for training
    }

    if bf16:
        import torch
        model_kwargs["torch_dtype"] = torch.bfloat16

    if method == "qlora":
        from transformers import BitsAndBytesConfig
        import torch

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        if method == "qlora":
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HF model for Yaya")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, default="data/sft",
                        help="Directory containing yaya_sft_train.jsonl / eval.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/yaya-sft",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--method", type=str, default="lora",
                        choices=["full", "lora", "qlora"],
                        help="Fine-tuning method")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (auto-selected if not set)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 training")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bfloat16")
    parser.add_argument("--wandb_project", type=str, default="yaya-ai",
                        help="W&B project name (set to 'none' to disable)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_bf16:
        args.bf16 = False

    # Auto-select learning rate based on method
    if args.lr is None:
        args.lr = {
            "full": 2e-5,
            "lora": 2e-4,
            "qlora": 2e-4,
        }[args.method]

    # ── Load data ──────────────────────────────────────────────────────────────
    train_ds, eval_ds = load_sft_dataset(args.data_dir)

    # ── Load model ─────────────────────────────────────────────────────────────
    model, tokenizer = create_model_and_tokenizer(
        args.base_model, args.method, bf16=args.bf16
    )

    # ── Format data ────────────────────────────────────────────────────────────
    train_ds = train_ds.map(
        lambda ex: format_messages_for_training(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    if eval_ds:
        eval_ds = eval_ds.map(
            lambda ex: format_messages_for_training(ex, tokenizer),
            remove_columns=eval_ds.column_names,
        )

    # ── Training config ────────────────────────────────────────────────────────
    from transformers import TrainingArguments
    from trl import SFTTrainer

    # Effective batch size = batch_size * grad_accum * n_gpus
    print(f"\nTraining config:")
    print(f"  Method:         {args.method}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Output:         {args.output_dir}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=100 if eval_ds else None,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        seed=args.seed,
        report_to="wandb" if args.wandb_project != "none" else "none",
        run_name=f"yaya-sft-{args.method}",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # If LoRA, also save merged model
    if args.method in ("lora", "qlora"):
        merged_dir = args.output_dir + "-merged"
        print(f"Merging LoRA weights and saving to {merged_dir}...")
        try:
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Merged model saved to {merged_dir}")
        except Exception as e:
            print(f"Warning: Could not merge LoRA weights: {e}")
            print("You can merge later with: model.merge_and_unload()")

    print("\n" + "=" * 60)
    print("SFT TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Model saved to: {args.output_dir}")
    if args.method in ("lora", "qlora"):
        print(f"  Merged model:   {args.output_dir}-merged")
    print()
    print("Next steps:")
    print(f"  DPO:    python scripts/align_dpo_hf.py --model_path {args.output_dir}")
    print(f"  Eval:   python scripts/eval_model.py --model_path {args.output_dir}")
    print(f"  Serve:  python scripts/deploy_model.py --model_path {args.output_dir}")


if __name__ == "__main__":
    main()
