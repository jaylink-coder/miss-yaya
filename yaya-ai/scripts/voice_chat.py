"""
Phase 13: Voice — Yaya can listen and speak.

Uses:
- OpenAI Whisper (open source) for speech-to-text
- pyttsx3 or gTTS for text-to-speech

Install:
    pip install openai-whisper pyttsx3 sounddevice soundfile numpy

Usage:
    python scripts/voice_chat.py \
        --model_config configs/model/yaya_125m.yaml \
        --checkpoint checkpoints/yaya-125m-sft/checkpoint-XXXXXXXX
"""

import argparse
import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN
from src.inference.generator import TextGenerator
from src.training.checkpointing import CheckpointManager
from src.memory.memory_store import MemoryStore

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly and are always honest. "
    "Keep your spoken responses concise — 2 to 4 sentences maximum, "
    "since this is a voice conversation."
)


def load_whisper():
    """Load Whisper speech-to-text model."""
    try:
        import whisper
        print("Loading Whisper (speech recognition)...")
        model = whisper.load_model("base")  # base model is fast and accurate enough
        print("Whisper ready.")
        return model
    except ImportError:
        print("Whisper not installed. Run: pip install openai-whisper")
        return None


def load_tts():
    """Load text-to-speech engine."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Set a natural speaking rate and volume
        engine.setProperty('rate', 165)
        engine.setProperty('volume', 0.9)
        # Try to find a good voice
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        print("Text-to-speech ready.")
        return engine
    except ImportError:
        print("pyttsx3 not installed. Run: pip install pyttsx3")
        return None


def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    """Record audio from microphone and save to temp file."""
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np

        print(f"\nListening for {duration} seconds... (speak now)")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=1, dtype='float32')
        sd.wait()
        print("Processing...")

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, audio, sample_rate)
        return tmp.name

    except ImportError:
        print("sounddevice/soundfile not installed. Run: pip install sounddevice soundfile")
        return None


def speech_to_text(whisper_model, audio_path: str) -> str:
    """Convert audio file to text using Whisper."""
    if not whisper_model or not audio_path:
        return None
    result = whisper_model.transcribe(audio_path, language='en')
    text = result.get('text', '').strip()
    os.unlink(audio_path)  # clean up temp file
    return text


def text_to_speech(tts_engine, text: str):
    """Speak text aloud."""
    if not tts_engine:
        return
    # Clean text for speech (remove markdown)
    clean = text.replace('**', '').replace('*', '').replace('`', '')
    clean = clean.replace('\n\n', '. ').replace('\n', ' ')
    tts_engine.say(clean)
    tts_engine.runAndWait()


def generate_response(generator, tokenizer, history: list,
                      max_tokens: int = 120, temperature: float = 0.7) -> str:
    prompt = tokenizer.format_chat(history) + "<|assistant|>\n"
    response = generator.generate(prompt, max_new_tokens=max_tokens,
                                  temperature=temperature, top_p=0.9)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    elif prompt in response:
        response = response[len(prompt):]
    for stop in ["<|user|>", "<|system|>", "</s>"]:
        response = response.split(stop)[0]
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Voice chat with Yaya")
    parser.add_argument("--model_config",  type=str, required=True)
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--listen_secs",   type=int,   default=6,
                        help="Seconds to listen per turn")
    parser.add_argument("--text_mode",     action="store_true",
                        help="Type instead of speaking (no microphone needed)")
    parser.add_argument("--no_speech_out", action="store_true",
                        help="Print response only, do not speak it")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Yaya on {device}...")

    model_config = load_model_config(args.model_config)
    model = YayaForCausalLM(model_config)
    ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(args.checkpoint))
    ckpt_mgr.load(model, checkpoint_path=args.checkpoint)
    model.eval().to(device)

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    generator = TextGenerator(model, tokenizer, device=device)
    memory    = MemoryStore()

    # Load speech components
    whisper_model = None if args.text_mode else load_whisper()
    tts_engine    = None if args.no_speech_out else load_tts()

    print("\n" + "=" * 50)
    print("  Yaya Voice Chat")
    if args.text_mode:
        print("  Mode: Text input")
    else:
        print(f"  Mode: Voice input ({args.listen_secs}s per turn)")
    print("  Say 'goodbye' or type 'quit' to exit")
    print("=" * 50 + "\n")

    # Greeting
    greeting = "Hello! I am Yaya. How can I help you today?"
    print(f"Yaya: {greeting}")
    if tts_engine:
        text_to_speech(tts_engine, greeting)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        # Get input
        if args.text_mode or not whisper_model:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        else:
            audio_path = record_audio(duration=args.listen_secs)
            user_input = speech_to_text(whisper_model, audio_path)
            if not user_input:
                print("(Could not hear anything, please try again)")
                continue
            print(f"\nYou said: {user_input}")

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "goodbye", "bye"):
            farewell = "Goodbye! It was great talking with you."
            print(f"Yaya: {farewell}")
            if tts_engine:
                text_to_speech(tts_engine, farewell)
            break

        # Auto-save memorable info
        if memory.extract_from_message(user_input):
            memory.remember(user_input, category='user_info')

        # Build history with memory context
        system = SYSTEM_PROMPT
        mem_ctx = memory.format_for_prompt(user_input)
        if mem_ctx:
            system += '\n\n' + mem_ctx

        full_history = [{"role": "system", "content": system}] + history[1:] + \
                       [{"role": "user", "content": user_input}]

        response = generate_response(generator, tokenizer, full_history)

        print(f"Yaya: {response}")
        if tts_engine:
            text_to_speech(tts_engine, response)

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})
        if len(history) > 22:
            history = [history[0]] + history[-20:]


if __name__ == "__main__":
    main()
