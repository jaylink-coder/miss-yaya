"""DPO training for Yaya alignment (simplified RLHF).

Usage:
    python scripts/train_dpo.py                              # auto-finds best SFT ckpt
    python scripts/train_dpo.py --sft_checkpoint path/to/checkpoint-XXXXX
    python scripts/train_dpo.py --sft_checkpoint path/to/model.pt
"""
import argparse, sys, os, json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN
from src.training.checkpointing import CheckpointManager

_DPO_SYSTEM = "You are Yaya, a helpful and honest AI assistant. You answer questions clearly and concisely."

DEFAULT_SFT_DIRS = [
    "checkpoints/yaya-125m-sft",
    "checkpoints/yaya-125m-reasoning",
    "checkpoints/yaya-125m",
]


def _find_sft_checkpoint():
    for d in DEFAULT_SFT_DIRS:
        latest = os.path.join(d, "latest")
        if os.path.exists(latest):
            with open(latest) as f:
                name = f.read().strip()
            path = os.path.join(d, name)
            if os.path.isdir(path):
                return path
    return None


def _load_weights(model, ckpt_path):
    """Load model weights from either a model.pt file or a checkpoint directory."""
    if os.path.isfile(ckpt_path):
        # Direct model.pt file
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Handle both raw state_dict and {"model_state_dict": ...} wrapping
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Weights loaded from {os.path.basename(ckpt_path)} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif os.path.isdir(ckpt_path):
        model_pt = os.path.join(ckpt_path, "model.pt")
        if not os.path.exists(model_pt):
            print(f"  ERROR: model.pt not found in {ckpt_path}")
            return
        state = torch.load(model_pt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Weights loaded from {os.path.basename(ckpt_path)}/model.pt "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"  ERROR: checkpoint not found: {ckpt_path!r}")


BETA = 0.1


class DPODataset(Dataset):
    """DPO preference dataset.

    Supports two formats per line:
      1. {"prompt": "...", "chosen": "...", "rejected": "..."}
      2. {"messages": [{"role":"user","content":"..."}], "chosen": "...", "rejected": "..."}
         (format produced by generate_phase_data.py for phase 16)
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    # Normalise to {"prompt": str, "chosen": str, "rejected": str}
                    if "prompt" not in ex and "messages" in ex:
                        # Extract the last user turn as the prompt
                        user_turns = [m["content"] for m in ex["messages"]
                                      if m.get("role") == "user"]
                        ex["prompt"] = user_turns[-1] if user_turns else ""
                    self.pairs.append(ex)
                except Exception:
                    pass

    def _encode(self, prompt, response):
        text = (SYSTEM_TOKEN + "\n" + _DPO_SYSTEM + "\n" +
                USER_TOKEN + "\n" + prompt + "\n" +
                ASSISTANT_TOKEN + "\n" + response + "</s>")
        ids = self.tokenizer.encode(text)[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return {
            "chosen":   self._encode(p["prompt"], p["chosen"]),
            "rejected": self._encode(p["prompt"], p["rejected"]),
        }


def pad_collate(batch):
    def pad(seqs):
        max_len = max(s.shape[0] for s in seqs)
        return torch.stack([F.pad(s, (0, max_len - s.shape[0]), value=0) for s in seqs])
    return {
        "chosen":   pad([b["chosen"]   for b in batch]),
        "rejected": pad([b["rejected"] for b in batch]),
    }


def get_log_probs(model, input_ids):
    out = model(input_ids=input_ids[:, :-1])
    logits = out["logits"] if isinstance(out, dict) else out[0]
    log_p = F.log_softmax(logits, dim=-1)
    return log_p.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)


def dpo_loss(policy, ref, chosen, rejected):
    pc = get_log_probs(policy, chosen)
    pr = get_log_probs(policy, rejected)
    with torch.no_grad():
        rc = get_log_probs(ref, chosen)
        rr = get_log_probs(ref, rejected)
    chosen_rewards  = BETA * (pc - rc)
    rejected_rewards = BETA * (pr - rr)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    acc  = (chosen_rewards > rejected_rewards).float().mean()
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description="DPO alignment training for Yaya")
    parser.add_argument("--model_config",   type=str, default="configs/model/yaya_125m.yaml")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="SFT checkpoint to start from. Accepts a model.pt file "
                             "OR a checkpoint directory. Auto-detected if omitted.")
    parser.add_argument("--dpo_data",       type=str, default="data/sft/yaya_dpo_combined.jsonl")
    parser.add_argument("--tokenizer",      type=str, default="data/tokenizer/yaya_tokenizer.model")
    parser.add_argument("--save_dir",       type=str, default="checkpoints/yaya-125m-dpo")
    parser.add_argument("--lr",             type=float, default=5e-7)
    parser.add_argument("--max_steps",      type=int,   default=1000)
    parser.add_argument("--batch_size",     type=int,   default=2)
    parser.add_argument("--log_steps",      type=int,   default=50)
    parser.add_argument("--save_steps",     type=int,   default=500)
    parser.add_argument("--hub_repo",       type=str,   default=None)
    parser.add_argument("--hub_prefix",     type=str,   default="dpo-checkpoint")
    args = parser.parse_args()

    sft_ckpt = args.sft_checkpoint or _find_sft_checkpoint()
    if sft_ckpt is None:
        print("ERROR: No SFT checkpoint found. Pass --sft_checkpoint.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DPO training on {device}")
    print(f"SFT checkpoint : {sft_ckpt}")
    print(f"DPO data       : {args.dpo_data}")
    print(f"Steps          : {args.max_steps}   LR: {args.lr}")

    mc = load_model_config(args.model_config)

    # Policy model (trained)
    policy = YayaForCausalLM(mc)
    _load_weights(policy, sft_ckpt)
    policy.to(device)

    # Reference model (frozen, same initial weights)
    ref = YayaForCausalLM(mc)
    _load_weights(ref, sft_ckpt)
    ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    tokenizer = YayaTokenizer(args.tokenizer)
    dataset = DPODataset(args.dpo_data, tokenizer)
    if len(dataset) == 0:
        print(f"ERROR: DPO dataset is empty: {args.dpo_data}")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=pad_collate, num_workers=0)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    print(f"DPO pairs: {len(dataset):,}   batch: {args.batch_size}   "
          f"max_steps: {args.max_steps}   lr: {args.lr}")

    ckpt_mgr = CheckpointManager(save_dir=args.save_dir, keep_last_n=3)

    step = 0
    policy.train()
    running_loss = 0.0
    running_acc  = 0.0
    last_loss    = 0.0

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            chosen   = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)

            loss, acc = dpo_loss(policy, ref, chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            step        += 1
            last_loss    = loss.item()
            running_loss += last_loss
            running_acc  += acc.item()

            if step % args.log_steps == 0:
                avg_loss = running_loss / args.log_steps
                avg_acc  = running_acc  / args.log_steps
                print(f"Step {step:5d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}", flush=True)
                running_loss = 0.0
                running_acc  = 0.0

            if step % args.save_steps == 0 and step < args.max_steps:
                ckpt_mgr.save(policy, optimizer=None, step=step, loss=last_loss)

    # Final checkpoint
    ckpt_mgr.save(policy, optimizer=None, step=step, loss=last_loss)
    print(f"\nDPO training complete. Checkpoint saved to {args.save_dir}")


if __name__ == "__main__":
    main()
