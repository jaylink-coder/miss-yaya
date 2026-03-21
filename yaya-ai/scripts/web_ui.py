"""Simple web chat UI for Yaya.

Runs a local web server with a chat interface in the browser.

Usage:
    python scripts/web_ui.py \
        --model_config configs/model/yaya_125m.yaml \
        --checkpoint checkpoints/yaya-125m-sft/checkpoint-XXXXXXXX

Then open http://localhost:7860 in your browser.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly, tell jokes when asked, and are always honest."
)


def load_model(model_config_path, checkpoint_path, device):
    from src.utils.config import load_model_config
    from src.model.yaya_model import YayaForCausalLM
    from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN
    from src.inference.generator import TextGenerator
    from src.training.checkpointing import CheckpointManager

    print(f"Loading Yaya on {device}...")
    model_config = load_model_config(model_config_path)
    model = YayaForCausalLM(model_config)

    ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(checkpoint_path))
    ckpt_mgr.load(model, checkpoint_path=checkpoint_path)
    model.eval().to(device)

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    generator = TextGenerator(model, tokenizer, device=device)
    print("Yaya ready.")
    return generator, tokenizer


def run_gradio(generator, tokenizer, args):
    try:
        import gradio as gr
    except ImportError:
        print("Installing gradio...")
        os.system(f"{sys.executable} -m pip install gradio -q")
        import gradio as gr

    def chat(message, history):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, bot_msg in history:
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        prompt = tokenizer.format_chat(messages) + "<|assistant|>\n"
        response = generator.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        elif prompt in response:
            response = response[len(prompt):]
        for end_tag in ["<|user|>", "<|system|>", "</s>"]:
            response = response.split(end_tag)[0]

        return response.strip()

    with gr.Blocks(title="Chat with Yaya", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 Chat with Yaya\nYaya is a custom AI assistant built from scratch.")
        chatbot = gr.ChatInterface(
            fn=chat,
            examples=[
                "Tell me a joke.",
                "What is artificial intelligence?",
                "Write a short poem about the stars.",
                "What can you do?",
            ],
            title="",
        )

    print(f"\nYaya web UI running at http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--port",         type=int, default=7860)
    parser.add_argument("--share",        action="store_true",
                        help="Create public Gradio link (useful on Colab)")
    parser.add_argument("--max_tokens",   type=int,   default=200)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_p",        type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator, tokenizer = load_model(args.model_config, args.checkpoint, device)
    run_gradio(generator, tokenizer, args)


if __name__ == "__main__":
    main()
