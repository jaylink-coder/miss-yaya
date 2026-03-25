"""Persistent cross-session WorkingMemory for Yaya.

Extends the per-conversation WorkingMemory from reasoning.py with JSON disk
persistence so facts, entities, and goals survive across sessions.

Design choices:
- One JSON file per session_id (or 'global' for the shared long-term store).
- Long-term memory is *consolidated* from session memories — only important
  facts are promoted, avoiding unbounded growth.
- Thread-safe: all mutations are protected by a threading.Lock.

Usage:
    # Long-term shared memory
    mem = PersistentMemory(store_dir="checkpoints/memory")
    mem.load()                                # restore from disk

    mem.add_fact("Kenya has 47 counties")
    mem.add_entity("Nairobi", "capital and largest city of Kenya")
    mem.set_goal("answer geography questions about East Africa")

    mem.save()                                # flush to disk

    # Session-scoped memory that auto-consolidates to long-term
    session = SessionMemory(store_dir="checkpoints/memory", session_id="abc123")
    session.load_long_term()                  # pull shared knowledge in
    session.add_fact("User's name is Alex")
    session.consolidate_to_long_term(min_importance=2)   # promote key facts
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Long-term memory entry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory item with metadata."""
    content: str
    kind: str                      # "fact" | "entity" | "goal"
    entity_key: Optional[str] = None   # set for entity entries
    importance: int = 1            # incremented each time it's re-confirmed
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    session_id: Optional[str] = None

    def touch(self):
        self.last_seen = time.time()
        self.importance += 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**d)


# ---------------------------------------------------------------------------
# PersistentMemory — long-term, disk-backed
# ---------------------------------------------------------------------------

class PersistentMemory:
    """Persistent long-term memory store.

    Backs the in-memory facts/entities/goals dicts to a JSON file on disk.
    Safe to call from multiple threads; NOT safe to share across processes.
    """

    MAX_FACTS = 500
    MAX_ENTITIES = 200
    MAX_GOALS = 50

    def __init__(self, store_dir: str = "checkpoints/memory", name: str = "long_term"):
        self.store_dir = store_dir
        self.name = name
        self._path = os.path.join(store_dir, f"{name}.json")
        self._lock = threading.Lock()

        # In-memory stores
        self._facts: List[MemoryEntry] = []
        self._entities: Dict[str, MemoryEntry] = {}   # entity_key → entry
        self._goals: List[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def add_fact(self, fact: str, importance: int = 1, session_id: Optional[str] = None) -> bool:
        """Add a fact. Returns True if it was new, False if it was a duplicate (importance bumped)."""
        with self._lock:
            for entry in self._facts:
                if entry.content.strip().lower() == fact.strip().lower():
                    entry.touch()
                    return False
            self._facts.append(
                MemoryEntry(content=fact, kind="fact", importance=importance, session_id=session_id)
            )
            self._trim(self._facts, self.MAX_FACTS)
            return True

    def add_entity(
        self,
        name: str,
        description: str,
        importance: int = 1,
        session_id: Optional[str] = None,
    ) -> bool:
        """Add or update an entity. Returns True if new."""
        key = name.strip().lower()
        with self._lock:
            if key in self._entities:
                e = self._entities[key]
                e.content = description  # update description
                e.touch()
                return False
            self._entities[key] = MemoryEntry(
                content=description,
                kind="entity",
                entity_key=key,
                importance=importance,
                session_id=session_id,
            )
            return True

    def set_goal(self, goal: str, importance: int = 1, session_id: Optional[str] = None) -> bool:
        """Add a goal if not already present."""
        with self._lock:
            for entry in self._goals:
                if entry.content.strip().lower() == goal.strip().lower():
                    entry.touch()
                    return False
            self._goals.append(
                MemoryEntry(content=goal, kind="goal", importance=importance, session_id=session_id)
            )
            self._trim(self._goals, self.MAX_GOALS)
            return True

    def forget_fact(self, fact: str) -> bool:
        """Remove a fact by content match. Returns True if found and removed."""
        key = fact.strip().lower()
        with self._lock:
            before = len(self._facts)
            self._facts = [e for e in self._facts if e.content.strip().lower() != key]
            return len(self._facts) < before

    def forget_entity(self, name: str) -> bool:
        """Remove an entity by name."""
        key = name.strip().lower()
        with self._lock:
            if key in self._entities:
                del self._entities[key]
                return True
            return False

    def clear_goals(self):
        with self._lock:
            self._goals.clear()

    def clear(self):
        with self._lock:
            self._facts.clear()
            self._entities.clear()
            self._goals.clear()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def facts(self) -> List[str]:
        with self._lock:
            return [e.content for e in sorted(self._facts, key=lambda e: -e.importance)]

    @property
    def entities(self) -> Dict[str, str]:
        with self._lock:
            return {e.entity_key or k: e.content for k, e in self._entities.items()}

    @property
    def goals(self) -> List[str]:
        with self._lock:
            return [e.content for e in sorted(self._goals, key=lambda e: -e.importance)]

    def top_facts(self, n: int = 10) -> List[str]:
        with self._lock:
            sorted_facts = sorted(self._facts, key=lambda e: (-e.importance, -e.last_seen))
            return [e.content for e in sorted_facts[:n]]

    def top_entities(self, n: int = 5) -> Dict[str, str]:
        with self._lock:
            sorted_ents = sorted(
                self._entities.items(), key=lambda kv: (-kv[1].importance, -kv[1].last_seen)
            )
            return {k: e.content for k, e in sorted_ents[:n]}

    def format_for_prompt(self, max_facts: int = 10, max_entities: int = 5) -> str:
        """Format memory for injection into a model prompt."""
        parts = []
        goals = self.goals
        if goals:
            parts.append("Goals: " + "; ".join(goals[:3]))
        facts = self.top_facts(max_facts)
        if facts:
            parts.append("Background knowledge:\n" + "\n".join(f"  - {f}" for f in facts))
        ents = self.top_entities(max_entities)
        if ents:
            parts.append("Key entities: " + "; ".join(f"{k}: {v}" for k, v in ents.items()))
        return "\n".join(parts)

    def __len__(self) -> int:
        with self._lock:
            return len(self._facts) + len(self._entities) + len(self._goals)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush in-memory store to JSON on disk."""
        os.makedirs(self.store_dir, exist_ok=True)
        with self._lock:
            data = {
                "facts":    [e.to_dict() for e in self._facts],
                "entities": {k: e.to_dict() for k, e in self._entities.items()},
                "goals":    [e.to_dict() for e in self._goals],
                "saved_at": time.time(),
            }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load from disk. No-op if file doesn't exist."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._facts = [MemoryEntry.from_dict(d) for d in data.get("facts", [])]
                self._entities = {
                    k: MemoryEntry.from_dict(v)
                    for k, v in data.get("entities", {}).items()
                }
                self._goals = [MemoryEntry.from_dict(d) for d in data.get("goals", [])]
        except Exception as e:
            print(f"[PersistentMemory] WARNING: could not load {self._path}: {e}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _trim(lst: list, max_size: int) -> None:
        """Keep only the highest-importance entries when over capacity."""
        if len(lst) > max_size:
            lst.sort(key=lambda e: (-e.importance, -e.last_seen))
            del lst[max_size:]


# ---------------------------------------------------------------------------
# SessionMemory — per-session, with optional consolidation to long-term
# ---------------------------------------------------------------------------

class SessionMemory:
    """Per-session working memory that can be consolidated into long-term memory.

    Typical lifecycle:
        1. Create with session_id at conversation start.
        2. Call load_long_term() to inject persistent context.
        3. Use add_fact / add_entity / set_goal throughout the session.
        4. Call consolidate_to_long_term() at session end to promote key facts.
        5. PersistentMemory.save() to flush to disk.
    """

    def __init__(
        self,
        store_dir: str = "checkpoints/memory",
        session_id: Optional[str] = None,
        long_term: Optional[PersistentMemory] = None,
    ):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.long_term = long_term or PersistentMemory(store_dir=store_dir)

        # Session-local stores (transient — not saved to disk independently)
        self._session_facts: List[str] = []
        self._session_entities: Dict[str, str] = {}
        self._session_goals: List[str] = []

    # ------------------------------------------------------------------
    # Session-local API (mirrors WorkingMemory interface)
    # ------------------------------------------------------------------

    def add_fact(self, fact: str) -> None:
        if fact and fact not in self._session_facts:
            self._session_facts.append(fact)

    def add_entity(self, name: str, description: str) -> None:
        self._session_entities[name] = description

    def set_goal(self, goal: str) -> None:
        if goal not in self._session_goals:
            self._session_goals.append(goal)

    def extract_from_text(self, text: str) -> None:
        """Heuristically extract facts/entities from text (mirrors WorkingMemory)."""
        import re
        numbers = re.findall(
            r'\b\d+(?:[.,]\d+)?(?:\s*(?:km|kg|mph|GB|TB|%|dollars?|years?))?\b', text
        )
        for n in numbers[:3]:
            self.add_fact(n)
        for m in re.finditer(
            r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+is\s+([^.]{5,60})', text
        ):
            self.add_entity(m.group(1), m.group(2).strip(".").strip())

    # ------------------------------------------------------------------
    # Long-term integration
    # ------------------------------------------------------------------

    def load_long_term(self) -> None:
        """Pull long-term memory into session context (read-only pull)."""
        self.long_term.load()

    def consolidate_to_long_term(self, min_importance: int = 1) -> int:
        """Promote session facts/entities/goals to long-term memory.

        Args:
            min_importance: Only promote entries that have been seen at least
                            this many times (currently all session entries = 1).

        Returns:
            Number of entries promoted.
        """
        promoted = 0
        for fact in self._session_facts:
            if self.long_term.add_fact(fact, session_id=self.session_id):
                promoted += 1
        for name, desc in self._session_entities.items():
            if self.long_term.add_entity(name, desc, session_id=self.session_id):
                promoted += 1
        for goal in self._session_goals:
            if self.long_term.set_goal(goal, session_id=self.session_id):
                promoted += 1
        return promoted

    # ------------------------------------------------------------------
    # Combined view for prompt injection
    # ------------------------------------------------------------------

    def format_for_prompt(self, max_facts: int = 10, max_entities: int = 5) -> str:
        """Merge long-term context + session context for prompt injection."""
        parts = []

        # Long-term background
        lt_str = self.long_term.format_for_prompt(max_facts=max_facts, max_entities=max_entities)
        if lt_str:
            parts.append(lt_str)

        # Session-specific additions (highest priority, shown last)
        if self._session_goals:
            parts.append("Session goals: " + "; ".join(self._session_goals[:3]))
        if self._session_facts:
            parts.append(
                "Session facts:\n" + "\n".join(f"  - {f}" for f in self._session_facts[-5:])
            )
        if self._session_entities:
            ents = "; ".join(
                f"{k}: {v}" for k, v in list(self._session_entities.items())[-3:]
            )
            parts.append("Session entities: " + ents)

        return "\n".join(parts)
