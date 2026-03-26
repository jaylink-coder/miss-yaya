"""Interactive chat with Yaya-tiny — uses persistent memory.

Usage:
    python scripts/chat_tiny.py
    python scripts/chat_tiny.py --checkpoint checkpoints/yaya-tiny-sft-clean/checkpoint-XXXXXXXX

Commands during chat:
    /memory     — show stored memories
    /clear      — clear session memory
    quit / exit — end the chat
"""

import argparse
import sys
import os

# UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN
from src.inference.generator import TextGenerator, GenerationConfig
from src.training.checkpointing import CheckpointManager
from src.agent.persistent_memory import SessionMemory

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly and honestly."
)

DEFAULT_CHECKPOINT_DIRS = [
    "checkpoints/yaya-tiny-sft-clean",
    "checkpoints/yaya-tiny-sft-v2",
    "checkpoints/yaya-tiny-sft",
    "checkpoints/yaya-tiny",
]


def _find_latest_checkpoint(dirs):
    for d in dirs:
        if os.path.isdir(d):
            latest = os.path.join(d, "latest")
            if os.path.exists(latest):
                with open(latest) as f:
                    name = f.read().strip()
                path = os.path.join(d, name)
                if os.path.isdir(path):
                    return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Chat with Yaya-tiny")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--top_k",       type=int,   default=50)
    args = parser.parse_args()

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = _find_latest_checkpoint(DEFAULT_CHECKPOINT_DIRS)
    if checkpoint is None:
        print("ERROR: No checkpoint found. Train a model first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Yaya-tiny from {checkpoint} on {device}...")

    model_config = load_model_config("configs/model/yaya_tiny.yaml")
    model = YayaForCausalLM(model_config)

    ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(checkpoint))
    ckpt_mgr.load(model, checkpoint_path=checkpoint)
    model.eval().to(device)

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    memory = SessionMemory(store_dir="data/memory", session_id="chat_tiny")
    generator = TextGenerator(model, tokenizer, device=device, memory=memory)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.5,   # Prevent degenerate token repetition
        do_sample=args.temperature > 0,
    )

    print("\n" + "=" * 55)
    print("  Chat with Yaya-tiny  (type 'quit' to exit)")
    print("  Commands: /memory, /clear")
    print("=" * 55 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Yaya: Goodbye!")
            break

        if user_input.lower() == "/memory":
            facts = memory.top_facts(20)
            if not facts:
                print("[No memories yet]\n")
            else:
                print(f"[{len(facts)} memories:]")
                for f in facts:
                    print(f"  - {f}")
                print()
            continue

        if user_input.lower() == "/clear":
            memory._session_facts.clear()
            memory._session_entities.clear()
            conversation.clear()
            print("[Session cleared]\n")
            continue

        # Build prompt
        history = ([{"role": "system", "content": SYSTEM_PROMPT}]
                   + conversation
                   + [{"role": "user", "content": user_input}])
        prompt = tokenizer.format_chat(history) + ASSISTANT_TOKEN + "\n"

        # Generate (memory injected automatically by generator)
        full_output = generator.generate(prompt, gen_config)

        # Extract response
        response = full_output[len(prompt):]
        for stop in [USER_TOKEN, SYSTEM_TOKEN, "</s>", "<|endoftext|>"]:
            if stop in response:
                response = response.split(stop)[0]
        response = response.strip()

        print(f"Yaya: {response}\n")

        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": response})

        # Keep context to last 8 exchanges
        if len(conversation) > 16:
            conversation = conversation[-16:]


if __name__ == "__main__":
    main()
