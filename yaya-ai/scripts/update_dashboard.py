"""Regenerate docs/dashboard.html with latest benchmark and phase-test data.

Run after any benchmark or phase test to keep the dashboard current:
    python scripts/update_dashboard.py

Also called automatically by benchmark.py and phase_tester.py.
"""

import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_FILE   = os.path.join(REPO_ROOT, "docs", "benchmark_results.jsonl")
PHASE_FILE   = os.path.join(REPO_ROOT, "docs", "phase_test_results.jsonl")
DASHBOARD    = os.path.join(REPO_ROOT, "docs", "dashboard.html")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def jsonl_to_js_string(rows):
    """Return rows as a multi-line JS string literal (no backticks — use concat)."""
    lines = [json.dumps(r, separators=(",", ":")) for r in rows]
    return "\n".join(lines)


def update_dashboard(bench_rows, phase_rows):
    if not os.path.exists(DASHBOARD):
        print(f"Dashboard not found: {DASHBOARD}")
        return False

    with open(DASHBOARD, encoding="utf-8") as f:
        html = f.read()

    # Replace the embedded BENCH_JSONL constant
    bench_js = jsonl_to_js_string(bench_rows)
    html = re.sub(
        r'(const BENCH_JSONL\s*=\s*`)[^`]*(`;)',
        lambda m: m.group(1) + bench_js + m.group(2),
        html,
        flags=re.DOTALL,
    )

    # Replace or inject PHASE_JSONL constant
    phase_js = jsonl_to_js_string(phase_rows)
    if "const PHASE_JSONL" in html:
        html = re.sub(
            r'(const PHASE_JSONL\s*=\s*`)[^`]*(`;)',
            lambda m: m.group(1) + phase_js + m.group(2),
            html,
            flags=re.DOTALL,
        )
    else:
        # Inject after BENCH_JSONL line
        html = html.replace(
            "const BENCH_JSONL",
            f"const PHASE_JSONL = `{phase_js}`;\nconst BENCH_JSONL",
        )

    with open(DASHBOARD, "w", encoding="utf-8") as f:
        f.write(html)

    return True


def main():
    bench_rows = load_jsonl(BENCH_FILE)
    phase_rows = load_jsonl(PHASE_FILE)

    print(f"Benchmark results: {len(bench_rows)} entries")
    print(f"Phase test results: {len(phase_rows)} entries")

    if not bench_rows and not phase_rows:
        print("Nothing to update.")
        return

    ok = update_dashboard(bench_rows, phase_rows)
    if ok:
        print(f"Dashboard updated: {DASHBOARD}")
    else:
        print("Dashboard update failed.")


if __name__ == "__main__":
    main()
