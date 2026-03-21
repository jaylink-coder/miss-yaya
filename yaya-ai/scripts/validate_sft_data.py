"""Validate SFT JSONL files before training.

Checks every example in data/sft/ for:
  - Valid JSON
  - Required fields and types
  - At least one assistant turn
  - Non-empty content in every role
  - No extremely long sequences (would OOM during training)

Run before starting SFT to catch issues early:
    python scripts/validate_sft_data.py
    python scripts/validate_sft_data.py --file data/sft/yaya_instruct.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_CHARS = 20_000   # warn if any single example exceeds this many characters
VALID_ROLES = {"system", "user", "assistant"}


def validate_file(path: str, verbose: bool = False) -> tuple[int, int]:
    """Validate a JSONL file. Returns (ok_count, error_count)."""
    ok = 0
    errors = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # JSON validity
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ERROR line {lineno}: invalid JSON — {e}")
                errors += 1
                continue

            # Must have "messages" key
            if "messages" not in obj:
                print(f"  ERROR line {lineno}: missing 'messages' key")
                errors += 1
                continue

            messages = obj["messages"]
            if not isinstance(messages, list) or len(messages) == 0:
                print(f"  ERROR line {lineno}: 'messages' must be a non-empty list")
                errors += 1
                continue

            has_assistant = False
            example_ok = True
            total_chars = 0

            for i, msg in enumerate(messages):
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role not in VALID_ROLES:
                    print(f"  ERROR line {lineno} msg[{i}]: unknown role {role!r}")
                    example_ok = False

                if not isinstance(content, str) or not content.strip():
                    print(f"  ERROR line {lineno} msg[{i}] role={role!r}: empty content")
                    example_ok = False

                if role == "assistant":
                    has_assistant = True

                total_chars += len(content)

            if not has_assistant:
                print(f"  ERROR line {lineno}: no assistant turn found")
                example_ok = False

            if total_chars > MAX_CHARS:
                if verbose:
                    print(f"  WARN  line {lineno}: very long example ({total_chars:,} chars) — may truncate")

            if example_ok:
                ok += 1
            else:
                errors += 1

    return ok, errors


def main():
    parser = argparse.ArgumentParser(description="Validate SFT JSONL training data")
    parser.add_argument("--file",    type=str, default=None,
                        help="Specific file to validate (default: all in data/sft/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also warn about long examples")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        sft_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sft")
        files = sorted(f for f in (
            os.path.join(sft_dir, fn) for fn in os.listdir(sft_dir)
            if fn.endswith(".jsonl")
        ) if os.path.isfile(f))

    if not files:
        print("No JSONL files found.")
        sys.exit(1)

    total_ok = 0
    total_errors = 0
    any_errors = False

    for path in files:
        fname = os.path.basename(path)
        ok, errors = validate_file(path, verbose=args.verbose)
        total_ok += ok
        total_errors += errors
        status = "OK" if errors == 0 else "ERRORS"
        print(f"  [{status}] {fname}: {ok:,} valid, {errors} errors")
        if errors:
            any_errors = True

    print(f"\nTotal: {total_ok:,} valid examples, {total_errors} errors across {len(files)} files")

    if any_errors:
        print("\nFix errors before training to avoid silent data corruption.")
        sys.exit(1)
    else:
        print("All files valid. Ready for SFT training.")


if __name__ == "__main__":
    main()
