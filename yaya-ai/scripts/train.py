"""Main training entry point for Yaya model.

Usage:
    # Single GPU
    python scripts/train.py --model_config configs/model/yaya_1_5b.yaml \
                            --train_config configs/training/train_1_5b.yaml

    # Distributed (8 GPUs)
    torchrun --nproc_per_node=8 scripts/train.py \
        --model_config configs/model/yaya_1_5b.yaml \
        --train_config configs/training/train_1_5b.yaml

    # Resume from checkpoint
    torchrun --nproc_per_node=8 scripts/train.py \
        --model_config configs/model/yaya_1_5b.yaml \
        --train_config configs/training/train_1_5b.yaml \
        --resume checkpoints/yaya-1.5b/checkpoint-00010000
"""

import argparse
import sys
import os

# UTF-8 output on Windows (model summary uses box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config, load_training_config
from src.model.yaya_model import YayaForCausalLM
from src.data.dataset import TextDataset, StreamingTextDataset
from src.data.dataloader import create_dataloader
from src.training.trainer import Trainer
from src.training.distributed import setup_distributed, is_main_process


def main():
    parser = argparse.ArgumentParser(description="Train Yaya model")
    parser.add_argument("--model_config", type=str, required=True, help="Model config YAML path")
    parser.add_argument("--train_config", type=str, required=True, help="Training config YAML path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    # Load configs
    model_config = load_model_config(args.model_config)
    train_config = load_training_config(args.train_config)

    if args.seed is not None:
        train_config.seed = args.seed

    # Set seeds
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    # Setup distributed
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]

    if is_main_process():
        print(f"Model config: {args.model_config}")
        print(f"Training config: {args.train_config}")
        print(f"Model: {model_config.model_name}")

    # Create model
    if is_main_process():
        print("Initializing model...")
    model = YayaForCausalLM(model_config)

    if is_main_process():
        print(model.generate_summary())

    # Create datasets
    if is_main_process():
        print("Loading datasets...")

    if os.path.isdir(train_config.train_data):
        train_dataset = StreamingTextDataset(
            data_dir=train_config.train_data,
            max_seq_length=train_config.max_seq_length,
            seed=train_config.seed,
        )
    else:
        train_dataset = TextDataset(
            data_path=train_config.train_data,
            max_seq_length=train_config.max_seq_length,
            split="train",
        )

    eval_dataset = None
    if train_config.eval_data and os.path.exists(train_config.eval_data):
        eval_dataset = TextDataset(
            data_path=train_config.eval_data,
            max_seq_length=train_config.max_seq_length,
            split="eval",
        )

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=train_config.per_device_batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        seed=train_config.seed,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=train_config.per_device_batch_size,
            num_workers=train_config.num_workers,
            shuffle=False,
            distributed=False,
        )

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
