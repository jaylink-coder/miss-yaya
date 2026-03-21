"""DPO Alignment Script for Yaya (HuggingFace/TRL).

Takes an SFT-tuned model and aligns it using Direct Preference Optimization
on preference pairs (chosen vs rejected responses).

Usage:
    # After SFT fine-tuning
    python scripts/align_dpo_hf.py \
        --model_path outputs/yaya-sft \
        --data_dir data/sft \
        --output_dir outputs/yaya-dpo

    # With LoRA on the SFT model
    python scripts/align_dpo_hf.py \
        --model_path outputs/yaya-sft-merged \
        --data_dir data/sft \
        --output_dir outputs/yaya-dpo \
        --method lora

Requirements:
    pip install transformers datasets peft trl bitsandbytes accelerate
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dpo_dataset(data_dir: str):
    """Load DPO preference pairs from JSONL files."""
    from datasets import Dataset

    train_path = os.path.join(data_dir, "yaya_dpo_train.jsonl")
    eval_path = os.path.join(data_dir, "yaya_dpo_eval.jsonl")

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
        print(f"ERROR: No DPO data found at {train_path}")
        print("Run: python scripts/generate_training_data.py")
        sys.exit(1)

    print(f"Loaded {len(train_data)} train, {len(eval_data)} eval preference pairs")
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None
    return train_ds, eval_ds


def format_dpo_example(example, tokenizer):
    """Format a DPO example into the expected format for TRL's DPOTrainer."""
    system_msg = (
        "You are Yaya, a helpful, accurate, and safe AI assistant. "
        "Provide clear, well-structured answers."
    )
    prompt_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": example["prompt"]},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    return {
        "prompt": prompt_text,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def main():
    parser = argparse.ArgumentParser(description="DPO alignment for Yaya")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SFT model (local or HF hub)")
    parser.add_argument("--data_dir", type=str, default="data/sft",
                        help="Directory containing yaya_dpo_train/eval.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/yaya-dpo",
                        help="Output directory for aligned model")
    parser.add_argument("--method", type=str, default="lora",
                        choices=["full", "lora"],
                        help="Fine-tuning method for DPO")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter (lower = stronger preference)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of DPO epochs (usually 1-2)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate (DPO uses very low LR)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="yaya-ai")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    train_ds, eval_ds = load_dpo_dataset(args.data_dir)

    # ── Load model ─────────────────────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading SFT model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        use_cache=False,
    )

    # Reference model (frozen copy of SFT model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # Apply LoRA if requested
    if args.method == "lora":
        from peft import LoraConfig, TaskType

        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
    else:
        peft_config = None

    # ── Format data ────────────────────────────────────────────────────────────
    train_ds = train_ds.map(
        lambda ex: format_dpo_example(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    if eval_ds:
        eval_ds = eval_ds.map(
            lambda ex: format_dpo_example(ex, tokenizer),
            remove_columns=eval_ds.column_names,
        )

    # ── DPO Training ───────────────────────────────────────────────────────────
    from transformers import TrainingArguments
    from trl import DPOTrainer, DPOConfig

    print(f"\nDPO Training config:")
    print(f"  Beta:           {args.beta}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"  Method:         {args.method}")
    print(f"  Output:         {args.output_dir}")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=50 if eval_ds else None,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=args.seed,
        report_to="wandb" if args.wandb_project != "none" else "none",
        run_name="yaya-dpo",
        gradient_checkpointing=True,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model if peft_config is None else None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\nStarting DPO training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"\nSaving aligned model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Merge LoRA if applicable
    if args.method == "lora":
        merged_dir = args.output_dir + "-merged"
        print(f"Merging LoRA weights -> {merged_dir}...")
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Merged model saved to {merged_dir}")
        except Exception as e:
            print(f"Warning: Could not merge: {e}")

    print("\n" + "=" * 60)
    print("DPO ALIGNMENT COMPLETE!")
    print("=" * 60)
    print(f"  Model saved to: {args.output_dir}")
    print()
    print("Next steps:")
    print(f"  Eval:   python scripts/eval_model.py --model_path {args.output_dir}")
    print(f"  Serve:  python scripts/deploy_model.py --model_path {args.output_dir}")


if __name__ == "__main__":
    main()
