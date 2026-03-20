"""
Phase 14: Continuous Learning — Yaya learns from conversations.

Collects conversation logs, filters high-quality exchanges,
and prepares them as new SFT training data for the next fine-tune cycle.

Usage:
    # After chatting, run:
    python scripts/continuous_learn.py \
        --conversations data/memory/conversations.jsonl \
        --output data/sft/yaya_instruct.jsonl \
        --min_quality 0.6
"""

import argparse
import json
import os
from datetime import datetime


def score_exchange(user_msg: str, assistant_msg: str) -> float:
    """
    Score a conversation exchange for training quality.
    Returns 0.0 to 1.0.
    """
    score = 0.0

    # Assistant response length (too short = bad, too long = ok)
    words = len(assistant_msg.split())
    if words >= 10:
        score += 0.3
    if words >= 30:
        score += 0.2

    # User message is a real question or request
    question_words = ['what', 'how', 'why', 'who', 'when', 'where', 'can', 'could',
                      'explain', 'tell', 'describe', 'help', 'write', 'give']
    if any(w in user_msg.lower() for w in question_words):
        score += 0.2

    # Response contains substantive content (not just filler)
    filler = ['i cannot', 'i do not know', 'sorry', 'unfortunately']
    if not any(f in assistant_msg.lower() for f in filler):
        score += 0.15

    # Response has structure (numbers, bullet points, paragraphs)
    if any(c in assistant_msg for c in ['1.', '2.', '-', '\n']):
        score += 0.15

    return min(score, 1.0)


def extract_conversations(log_path: str) -> list:
    """Load conversation logs."""
    if not os.path.exists(log_path):
        return []
    conversations = []
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return conversations


def log_conversation(user_msg: str, assistant_msg: str,
                     log_path: str = 'data/memory/conversations.jsonl'):
    """Log a conversation exchange for future learning."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        'timestamp': datetime.now().isoformat(),
        'user':      user_msg,
        'assistant': assistant_msg,
    }
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conversations', type=str,
                        default='data/memory/conversations.jsonl')
    parser.add_argument('--output',        type=str,
                        default='data/sft/yaya_instruct.jsonl')
    parser.add_argument('--min_quality',   type=float, default=0.6,
                        help='Minimum quality score to include (0.0-1.0)')
    args = parser.parse_args()

    SYS = ('You are Yaya, a helpful and friendly AI assistant. '
           'You answer questions clearly, tell jokes when asked, and are always honest.')

    conversations = extract_conversations(args.conversations)
    if not conversations:
        print(f'No conversations found at {args.conversations}')
        print('Conversations are logged automatically during chat sessions.')
        return

    print(f'Found {len(conversations)} conversation exchanges')

    added = 0
    skipped = 0
    for conv in conversations:
        user = conv.get('user', '').strip()
        assistant = conv.get('assistant', '').strip()

        if not user or not assistant:
            skipped += 1
            continue

        quality = score_exchange(user, assistant)
        if quality < args.min_quality:
            skipped += 1
            continue

        ex = {'messages': [
            {'role': 'system',    'content': SYS},
            {'role': 'user',      'content': user},
            {'role': 'assistant', 'content': assistant},
        ]}
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        added += 1

    with open(args.output, encoding='utf-8') as f:
        total = sum(1 for line in f if line.strip())

    print(f'Added {added} high-quality exchanges (skipped {skipped} low-quality)')
    print(f'Total SFT dataset: {total} examples')
    print(f'\nNext step: retrain SFT on the updated dataset to incorporate new knowledge.')


if __name__ == '__main__':
    main()
