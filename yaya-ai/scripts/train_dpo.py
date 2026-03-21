"""DPO training for Yaya alignment (simplified RLHF)."""
import argparse, sys, os, json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, USER_TOKEN, ASSISTANT_TOKEN
from src.training.checkpointing import CheckpointManager

BETA = 0.1

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip(): self.pairs.append(json.loads(line))
    def _encode(self, prompt, response):
        text = USER_TOKEN + "\n" + prompt + ASSISTANT_TOKEN + "\n" + response + "</s>"
        ids = self.tokenizer.encode(text)[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]
        return {"chosen": self._encode(p["prompt"], p["chosen"]),
                "rejected": self._encode(p["prompt"], p["rejected"])}

def pad_collate(batch):
    def pad(seqs):
        max_len = max(s.shape[0] for s in seqs)
        return torch.stack([F.pad(s, (0, max_len - s.shape[0]), value=0) for s in seqs])
    return {"chosen": pad([b["chosen"] for b in batch]),
            "rejected": pad([b["rejected"] for b in batch])}

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
    cr = BETA * (pc - rc)
    rr2 = BETA * (pr - rr)
    loss = -F.logsigmoid(cr - rr2).mean()
    return loss, (cr > rr2).float().mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--dpo_data", default="data/sft/dpo_pairs.jsonl")
    parser.add_argument("--save_dir", default="checkpoints/yaya-dpo")
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DPO on {device}")
    mc = load_model_config(args.model_config)
    policy = YayaForCausalLM(mc)
    ckpt = CheckpointManager(save_dir=os.path.dirname(args.sft_checkpoint))
    ckpt.load(policy, checkpoint_path=args.sft_checkpoint)
    policy.to(device)
    ref = YayaForCausalLM(mc)
    ckpt.load(ref, checkpoint_path=args.sft_checkpoint)
    ref.to(device).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    dataset = DPODataset(args.dpo_data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    print(f"DPO pairs: {len(dataset)}")
    os.makedirs(args.save_dir, exist_ok=True)
    step = 0
    policy.train()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break
            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)
            loss, acc = dpo_loss(policy, ref, chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc.item():.2%}")
    sp = os.path.join(args.save_dir, "checkpoint-final")
    os.makedirs(sp, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(sp, "model.pt"))
    print(f"Saved to {sp}")

if __name__ == "__main__":
    main()
