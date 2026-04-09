"""
Rebuild data/sft/yaya_qa_focused.jsonl from data/memory/yaya_qa_full.md.

Parses all full-text Q&A entries (### Q145 — Domain format) and writes
them as SFT JSONL samples. Replaces the existing file entirely.
"""
import sys, os, re, json, io
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SYS = (
    "You are Yaya, a knowledgeable AI assistant. You give thorough, "
    "accurate, and thoughtful answers across all domains of knowledge."
)

FULL_DOC  = "data/memory/yaya_qa_full.md"
OUT_PATH  = "data/sft/yaya_qa_focused.jsonl"

with open(FULL_DOC, encoding="utf-8") as f:
    content = f.read()

# Split on ### Q<number> headers
# Each entry: ### Q{n} — {domain}\n**{question}**\n\n{answer}\n\n**Rating: ...
pattern = re.compile(
    r"### Q(\d+)\s*[—–-]+\s*(.+?)\n"   # header: Q-number and domain
    r"\*\*(.+?)\*\*\n\n"                 # **question**
    r"(.*?)"                             # answer (non-greedy)
    r"(?:\n\n\*\*Rating:.*?\*\*\n|\n\n---|\Z)",  # ends at Rating or --- or EOF
    re.DOTALL,
)

entries = []
for m in pattern.finditer(content):
    q_num   = int(m.group(1))
    domain  = m.group(2).strip()
    question = m.group(3).strip()
    answer   = m.group(4).strip()
    if not question or not answer or len(answer) < 30:
        continue
    entries.append((q_num, domain, question, answer))

entries.sort(key=lambda x: x[0])

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for q_num, domain, question, answer in entries:
        sample = {
            "messages": [
                {"role": "system",    "content": SYS},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Rebuilt {OUT_PATH}")
print(f"Total entries: {len(entries)}")
print(f"Range: Q{entries[0][0]} — Q{entries[-1][0]}")
print(f"\nSample domains: {', '.join(set(d for _,d,_,_ in entries[:20]))}")

# Verify a few
print("\nSpot-check last 5:")
for q_num, domain, question, answer in entries[-5:]:
    print(f"  Q{q_num} ({domain}): {question[:60]}")
