"""The README's knowledge comparison (what the lab loses without its limits, and what the agent recovers),
printed as a markdown table from the v3 runs.   PYTHONPATH=src python scripts/headline_table.py"""

from __future__ import annotations

import json
from pathlib import Path

from labloop.metrics import summarize

RES = Path("results/m9/level4")
ROWS = [("blind_v3", "No limits"), ("agent_open_v3", "Agent, before launch"),
        ("agent_review_v3", "Agent + in-run reviews"), ("gold_v3", "All real limits (gold)")]


def main() -> None:
    print("| Run | Synergy missed (regret) | True best pairs found | Wrong values kept | Invalid plates | Hours to finish |")
    print("|---|---|---|---|---|---|")
    for variant, label in ROWS:
        f = RES / f"{variant}.json"
        if not f.exists():
            print(f"| {label} | (missing {variant}) | | | | |")
            continue
        runs = json.loads(f.read_text())
        s = summarize(runs)
        n = 2 * len(runs)
        print(f"| {label} | {s['regret']:.2f} | {round(s['hit'] * n)} / {n} | {s['corrupt_accepted']:.0f} | "
              f"{s['invalid_plates']:.1f} | {s['makespan_h']:.0f} |")


if __name__ == "__main__":
    main()
