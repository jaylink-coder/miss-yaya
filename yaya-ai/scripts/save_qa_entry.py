"""
Save a full Q&A entry to both Yaya knowledge base documents.
Used after each question to persist the complete answer text.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.memory.memory_store import MemoryStore


FULL_DOC   = "data/memory/yaya_qa_full.md"
SUMMARY_DOC = "data/memory/yaya_qa_knowledge_base.md"


def save_qa(q_num: int, domain: str, question: str, answer: str,
            rating: float, summary: str):
    """
    Append a full Q&A entry to both documents and save a memory summary.

    Args:
        q_num:    Question number (e.g. 145)
        domain:   Domain label (e.g. "Physics")
        question: The question text
        answer:   Yaya's full answer text
        rating:   Numeric rating out of 10
        summary:  One-line memory summary (stored in yaya_memory.json)
    """
    memory = MemoryStore('data/memory/yaya_memory.json')

    # 1. Save compressed memory
    memory.remember(summary, category='knowledge', source='qa_session_2026_03_23')

    # 2. Append to full doc
    entry = (
        f"\n### Q{q_num} — {domain}\n"
        f"**{question}**\n\n"
        f"{answer}\n\n"
        f"**Rating: {rating}/10**\n\n"
        f"---\n"
    )
    with open(FULL_DOC, 'a', encoding='utf-8') as f:
        f.write(entry)

    # 3. Rebuild summary doc from all memories
    all_mems = [m for m in memory.list_all() if m['source'] == 'qa_session_2026_03_23']
    lines = [
        "# Yaya Q&A Knowledge Base",
        "Generated from Q&A session — 2026-03-23",
        f"Total entries: {len(all_mems)}",
        "",
        "---",
        "",
    ]
    for m in all_mems:
        lines.append(f"**[{m['id']}]** {m['content']}")
        lines.append("")
    with open(SUMMARY_DOC, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return len(memory)


if __name__ == "__main__":
    # Quick test
    total = save_qa(
        q_num=0, domain="Test", question="Test question?",
        answer="Test answer.", rating=9.0,
        summary="Q0 Test: this is a test entry"
    )
    print(f"Test entry saved. Total memories: {total}")
