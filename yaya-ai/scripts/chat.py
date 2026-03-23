"""Interactive chat with Yaya — with long-term memory.

Usage:
    python scripts/chat.py \
        --model_config configs/model/yaya_125m.yaml \
        --checkpoint checkpoints/yaya-125m-sft/checkpoint-XXXXXXXX

Commands during chat:
    /memory        — show all stored memories
    /forget <id>   — delete a memory by ID
    quit / exit    — end the chat
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN
from src.inference.generator import TextGenerator
from src.training.checkpointing import CheckpointManager
from src.memory.memory_store import MemoryStore
from scripts.continuous_learn import log_conversation

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly, tell jokes when asked, and are always honest. "
    "When solving problems, you think step by step before giving your final answer."
)


def build_system_prompt(memory: MemoryStore, query: str) -> str:
    """Extend the system prompt with relevant memories."""
    mem_context = memory.format_for_prompt(query)
    if mem_context:
        return SYSTEM_PROMPT + '\n\n' + mem_context
    return SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Chat with Yaya")
    parser.add_argument("--model_config",  type=str, required=True)
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--max_tokens",    type=int,   default=200)
    parser.add_argument("--temperature",   type=float, default=0.8)
    parser.add_argument("--top_p",         type=float, default=0.9)
    parser.add_argument("--memory_path",   type=str,   default='data/memory/yaya_memory.json')
    parser.add_argument("--conv_log",      type=str,   default='data/memory/conversations.jsonl')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Yaya on {device}...")

    model_config = load_model_config(args.model_config)
    model = YayaForCausalLM(model_config)

    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_mgr = CheckpointManager(save_dir=ckpt_dir)
    ckpt_mgr.load(model, checkpoint_path=args.checkpoint)
    model.eval().to(device)

    tokenizer = YayaTokenizer(
        os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)),
                     "data/tokenizer/yaya_tokenizer.model")
        if not os.path.exists("data/tokenizer/yaya_tokenizer.model")
        else "data/tokenizer/yaya_tokenizer.model"
    )

    generator = TextGenerator(model, tokenizer, device=device)
    memory    = MemoryStore(args.memory_path)

    print("\n" + "=" * 55)
    print("  Chat with Yaya  (type 'quit' to exit)")
    print(f"  Memory: {len(memory)} stored memories")
    print("=" * 55 + "\n")

    conversation = []  # current session history (no system prompt here)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Built-in commands ────────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Yaya: Goodbye! It was great chatting with you.")
            break

        if user_input.lower() == '/memory':
            mems = memory.list_all()
            if not mems:
                print("[No memories stored yet]\n")
            else:
                print(f"[{len(mems)} memories stored:]")
                for m in mems:
                    print(f"  [{m['id']}] {m['content']}  ({m['timestamp'][:10]})")
                print()
            continue

        if user_input.lower().startswith('/forget '):
            try:
                mem_id = int(user_input.split()[1])
                memory.forget(mem_id)
                print(f"[Memory {mem_id} deleted]\n")
            except (ValueError, IndexError):
                print("[Usage: /forget <id>]\n")
            continue

        # ── Auto-detect memorable information ───────────────────────────────
        memorable = memory.extract_from_message(user_input)
        if memorable:
            memory.remember(memorable, category='user_info', source='conversation')

        # ── Build prompt with memory context ────────────────────────────────
        system = build_system_prompt(memory, user_input)
        history = [{"role": "system", "content": system}] + conversation + [{"role": "user", "content": user_input}]

        prompt = tokenizer.format_chat(history) + ASSISTANT_TOKEN + "\n"

        response = generator.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Clean up response
        if ASSISTANT_TOKEN in response:
            response = response.split(ASSISTANT_TOKEN)[-1]
        elif prompt in response:
            response = response[len(prompt):]
        for end_tag in [USER_TOKEN, SYSTEM_TOKEN, "</s>"]:
            response = response.split(end_tag)[0]
        response = response.strip()

        print(f"Yaya: {response}\n")

        # Log exchange for continuous learning
        log_conversation(user_input, response, log_path=args.conv_log)

        # Update conversation history
        conversation.append({"role": "user",      "content": user_input})
        conversation.append({"role": "assistant", "content": response})

        # Keep context window manageable (last 10 exchanges)
        if len(conversation) > 20:
            conversation = conversation[-20:]


if __name__ == "__main__":
    main()
