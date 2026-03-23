"""
Yaya Long-Term Memory Store
Saves and retrieves memories across conversations using simple vector similarity.
"""

import json
import os
import math
from datetime import datetime
from typing import Optional


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _simple_embed(text: str) -> tuple:
    """
    Word + character-trigram embedding for better semantic matching.
    Words handle exact matches; trigrams handle morphological variants
    (e.g. 'coding' and 'code' share trigrams like 'cod', 'odi').
    No external dependencies.
    """
    text = text.lower()
    words = text.split()

    # Word-level features (weighted 2x)
    word_features = [f"w:{w}" for w in words]

    # Character trigram features from each word
    trigram_features = []
    for w in words:
        if len(w) >= 3:
            trigram_features += [f"t:{w[i:i+3]}" for i in range(len(w) - 2)]

    all_features = word_features * 2 + trigram_features  # words weighted higher
    vocab = sorted(set(all_features))
    vec = [all_features.count(f) for f in vocab]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec], vocab


class MemoryStore:
    """
    Persistent memory for Yaya across conversations.

    Stores facts, user preferences, and important context.
    Retrieves relevant memories based on semantic similarity.
    """

    def __init__(self, memory_path: str = 'data/memory/yaya_memory.json'):
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        self.memories = self._load()

    def _load(self) -> list:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2, ensure_ascii=False)

    def remember(self, content: str, category: str = 'general', source: str = 'conversation'):
        """Save a new memory."""
        next_id = (max(m['id'] for m in self.memories) + 1) if self.memories else 0
        memory = {
            'id':        next_id,
            'content':   content,
            'category':  category,
            'source':    source,
            'timestamp': datetime.now().isoformat(),
        }
        self.memories.append(memory)
        self._save()
        return memory

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list:
        """Retrieve the most relevant memories for a given query."""
        if not self.memories:
            return []

        query_vec, query_vocab = _simple_embed(query)

        scored = []
        for mem in self.memories:
            mem_vec, mem_vocab = _simple_embed(mem['content'])

            # align vectors to same vocabulary
            all_vocab = sorted(set(query_vocab) | set(mem_vocab))
            q_map = {w: i for i, w in enumerate(query_vocab)}
            m_map = {w: i for i, w in enumerate(mem_vocab)}
            q_full = [query_vec[q_map[w]] if w in q_map else 0.0 for w in all_vocab]
            m_full = [mem_vec[m_map[w]]   if w in m_map else 0.0 for w in all_vocab]

            sim = _cosine_similarity(q_full, m_full)
            if sim >= threshold:
                scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def forget(self, memory_id: int):
        """Remove a memory by ID."""
        self.memories = [m for m in self.memories if m['id'] != memory_id]
        self._save()

    def list_all(self, category: Optional[str] = None) -> list:
        """List all memories, optionally filtered by category."""
        if category:
            return [m for m in self.memories if m['category'] == category]
        return self.memories

    def format_for_prompt(self, query: str) -> str:
        """Return relevant memories formatted as a system prompt addition."""
        relevant = self.recall(query)
        if not relevant:
            return ''
        lines = ['[Yaya\'s memory — things she knows about you:]']
        for mem in relevant:
            lines.append(f'- {mem["content"]}')
        return '\n'.join(lines)

    def extract_from_message(self, message: str) -> Optional[str]:
        """
        Detect if a user message contains memorable information
        like name, preferences, or important facts.
        """
        msg = message.lower().strip()
        triggers = [
            'my name is', 'i am called', 'call me',
            'i like', 'i love', 'i hate', 'i prefer',
            'i work', 'i live', 'i am from', 'i study',
            'remember that', 'remember this', 'please remember',
            'do not forget', 'keep in mind',
        ]
        for trigger in triggers:
            if trigger in msg:
                return message  # worth remembering
        return None

    def __len__(self):
        return len(self.memories)

    def __repr__(self):
        return f'MemoryStore({len(self.memories)} memories at {self.memory_path})'
