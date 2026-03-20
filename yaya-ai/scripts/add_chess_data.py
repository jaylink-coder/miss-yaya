"""
add_chess_data.py
-----------------
Reads data/chess/chess_sft.jsonl and appends all examples to
data/sft/yaya_instruct.jsonl.

Usage:
    python scripts/add_chess_data.py
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.join(SCRIPT_DIR, "..")
CHESS_SFT    = os.path.join(REPO_ROOT, "data", "chess", "chess_sft.jsonl")
INSTRUCT_SFT = os.path.join(REPO_ROOT, "data", "sft", "yaya_instruct.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_lines(path: str) -> int:
    """Count non-empty lines in a JSONL file."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate_example(obj: dict) -> bool:
    """
    Return True if the object looks like a valid SFT example
    (has a 'messages' key that is a non-empty list).
    """
    return (
        isinstance(obj, dict)
        and "messages" in obj
        and isinstance(obj["messages"], list)
        and len(obj["messages"]) > 0
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Verify source file exists ---
    if not os.path.exists(CHESS_SFT):
        print(f"ERROR: Source file not found: {os.path.abspath(CHESS_SFT)}")
        print("Run generate_chess_data.py first to create it.")
        sys.exit(1)

    # --- Verify destination directory exists ---
    dest_dir = os.path.dirname(INSTRUCT_SFT)
    if not os.path.exists(dest_dir):
        print(f"ERROR: Destination directory not found: {dest_dir}")
        print("The data/sft/ directory must exist before running this script.")
        sys.exit(1)

    # --- Load source examples ---
    examples: list[dict] = []
    skipped = 0
    with open(CHESS_SFT, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARNING: Skipping malformed JSON on line {lineno}: {exc}")
                skipped += 1
                continue
            if validate_example(obj):
                examples.append(obj)
            else:
                print(f"  WARNING: Skipping invalid example on line {lineno} (missing 'messages').")
                skipped += 1

    if not examples:
        print("No valid examples found in the chess SFT file. Nothing to add.")
        sys.exit(0)

    # --- Count existing examples before appending ---
    before = count_lines(INSTRUCT_SFT)

    # --- Append to destination ---
    with open(INSTRUCT_SFT, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    after = count_lines(INSTRUCT_SFT)
    added = after - before

    # --- Report ---
    print(f"Source file   : {os.path.abspath(CHESS_SFT)}")
    print(f"Destination   : {os.path.abspath(INSTRUCT_SFT)}")
    if skipped:
        print(f"Skipped       : {skipped} malformed/invalid line(s)")
    print(f"Examples added: {added}")
    print(f"Total examples: {after}")


if __name__ == "__main__":
    main()
