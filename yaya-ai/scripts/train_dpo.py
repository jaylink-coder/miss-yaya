"""Train Yaya model with Direct Preference Optimization (DPO).

Phase 3 of training: align the SFT model with human preferences.

Usage:
    python scripts/train_dpo.py \
        --model_config configs/model/yaya_1_5b.yaml \
        --train_config configs/training/dpo_1_5b.yaml \
        --checkpoint checkpoints/yaya-1.5b-sft/latest \
        --preference_data data/preferences/train.jsonl
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_model_config, load_training_config, ModelConfig
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer
from src.training.alignment import DPOTrainer, DPOConfig, PreferenceDataset
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.checkpointing import CheckpointManager
from src.training.logging_utils import TrainingLogger


def collate_preference_batch(batch):
    """Collate preference pairs with padding."""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        tensors = [sample[key] for sample in batch]
        max_len = max(t.shape[0] for t in tensors)
        pad_value = -100 if "labels" in key else 0

        padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        masks = torch.zeros(len(tensors), max_len, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
            masks[i, :t.shape[0]] = 1

        collated[key] = padded
        if "attention_mask" in key:
            collated[key] = masks

    return collated


def main():
    parser = argparse.ArgumentParser(description="DPO alignment training")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="SFT checkpoint path")
    parser.add_argument("--preference_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/yaya-dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--loss_type", type=str, default="sigmoid", choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model config
    model_config = load_model_config(args.model_config)

    # Load tokenizer
    tokenizer_path = "data/tokenizer/yaya_tokenizer.model"
    if args.train_config:
        train_config = load_training_config(args.train_config)
        tokenizer_path = train_config.tokenizer_path
    tokenizer = YayaTokenizer(tokenizer_path)

    # Load policy model from SFT checkpoint
    print(f"Loading SFT checkpoint: {args.checkpoint}")
    policy_model = YayaForCausalLM(model_config).to(device)
    ckpt_manager = CheckpointManager(save_dir=args.output_dir)
    ckpt_manager.load(policy_model, checkpoint_path=args.checkpoint)

    # DPO setup
    dpo_config = DPOConfig(
        beta=args.beta,
        loss_type=args.loss_type,
    )
    dpo_trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=None,  # Will deep-copy policy
        config=dpo_config,
    )

    # Dataset
    print(f"Loading preference data: {args.preference_data}")
    dataset = PreferenceDataset(
        data_path=args.preference_data,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_preference_batch,
        num_workers=2,
    )

    # Optimizer
    optimizer = create_optimizer(
        policy_model,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        schedule_type="cosine",
    )

    # Logger
    logger = TrainingLogger(log_dir=os.path.join(args.output_dir, "logs"))

    # Training loop
    print(f"\nStarting DPO training: {args.max_steps} steps")
    print(f"  Beta: {args.beta}, Loss: {args.loss_type}")
    print(f"  LR: {args.learning_rate}, Batch: {args.batch_size} x {args.gradient_accumulation_steps}")
    print()

    global_step = 0
    policy_model.train()
    data_iter = iter(dataloader)

    while global_step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_metrics = {"chosen_reward": 0, "rejected_reward": 0, "accuracy": 0}

        for accum_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            results = dpo_trainer.compute_dpo_loss(
                chosen_input_ids=batch["chosen_input_ids"],
                chosen_labels=batch["chosen_labels"],
                rejected_input_ids=batch["rejected_input_ids"],
                rejected_labels=batch["rejected_labels"],
                chosen_attention_mask=batch["chosen_attention_mask"],
                rejected_attention_mask=batch["rejected_attention_mask"],
            )

            loss = results["loss"] / args.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            for k in accum_metrics:
                accum_metrics[k] += results[k].item() / args.gradient_accumulation_steps

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % args.log_steps == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"Step {global_step}/{args.max_steps} | "
                f"Loss: {accum_loss:.4f} | "
                f"Chosen: {accum_metrics['chosen_reward']:.3f} | "
                f"Rejected: {accum_metrics['rejected_reward']:.3f} | "
                f"Acc: {accum_metrics['accuracy']:.3f} | "
                f"LR: {lr:.2e}"
            )

        if global_step % args.save_steps == 0:
            ckpt_manager.save(policy_model, optimizer, step=global_step, loss=accum_loss)
            print(f"  Saved checkpoint at step {global_step}")

    # Final save
    ckpt_manager.save(policy_model, optimizer, step=global_step, loss=accum_loss)
    print(f"\nDPO training complete. Final checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
