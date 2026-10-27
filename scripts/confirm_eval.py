"""Pre-registered confirmatory test: ours vs. the two strongest baselines (plain BO, FIFO) on fresh seeds.

    PYTHONPATH=src python scripts/confirm_eval.py [results/m9/confirm/level4]

Fixed before the runs (seeds 30-45, scripts/run_baselines.sh with OUT=results/m9/confirm):
- unit: one campaign of one seed (2 campaigns x 16 seeds = 32 per method), paired by (seed, campaign):
  the same seed draws the same lab noise, failures and landscape for every method.
- metric: hours until the reported best pair's true synergy reaches >= 90% of the best possible and never
  drops below it again; a campaign that never holds is censored at its run's end + 1 h.
- claim: ours beats a baseline only if its median is lower and a one-sided paired Wilcoxon test gives
  p < 0.05. Final synergy (1 - regret) and best-pair hits are reported alongside, whatever they show.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from labloop.metrics import time_to_target

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/m9/confirm/level4")
METHODS = [("ours", "base_ours_v3", "Ours"), ("bo", "base_bo_v3", "Plain BO"), ("fifo", "base_fifo_v3", "FIFO dispatch")]


def hold_hours(progress: list[dict], optimum: float, end_h: float) -> float:
    """First time from which the reported best stays >= 90% of the optimum to the end (censored: end + 1)."""
    ok = [p["best_true"] is not None and p["best_true"] >= 0.9 * optimum for p in progress]
    if not ok or not ok[-1]:
        return end_h + 1
    i = len(ok)
    while i > 0 and ok[i - 1]:
        i -= 1
    return progress[i]["t"] / 60.0


def per_campaign(variant: str) -> dict[tuple, dict]:
    out = {}
    for r in json.loads((RES / f"{variant}.json").read_text()):
        for name, c in r["campaigns"].items():
            opt, fb = c["optimum"]["excess"], c["final_best"]
            out[(r["seed"], name)] = {
                "hold_h": hold_hours(c["progress"], opt, r["makespan"] / 60.0),
                "found": 1 - (fb["regret"] if fb and fb["regret"] is not None else opt) / opt if opt else 0.0,
                "hit": time_to_target(c["progress"], opt)[0] is not None,
            }
    return out


def main() -> None:
    data = {k: per_campaign(v) for k, v, _ in METHODS if (RES / f"{v}.json").exists()}
    methods = [m for m in METHODS if m[0] in data]
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"{len(keys)} paired campaigns (seeds {min(k[0] for k in keys)}-{max(k[0] for k in keys)})")
    for k, _, label in methods:
        h = np.array([data[k][x]["hold_h"] for x in keys])
        f = np.array([data[k][x]["found"] for x in keys])
        print(f"{label:14s} hold-90% median {np.median(h):6.1f} h (IQR {np.percentile(h, 25):.0f}-{np.percentile(h, 75):.0f})"
              f" | final synergy mean {f.mean():.3f} | best pair {sum(data[k][x]['hit'] for x in keys)}/{len(keys)}")
    ours = np.array([data["ours"][x]["hold_h"] for x in keys])
    for k, _, label in methods[1:]:
        other = np.array([data[k][x]["hold_h"] for x in keys])
        stat = wilcoxon(ours, other, alternative="less", zero_method="zsplit")
        verdict = "BEATS" if np.median(ours) < np.median(other) and stat.pvalue < 0.05 else "does not beat"
        print(f"ours vs {label}: median diff {np.median(ours - other):+.1f} h, ours faster in "
              f"{int((ours < other).sum())}/{len(keys)}, Wilcoxon p = {stat.pvalue:.4f} -> ours {verdict} {label}")


if __name__ == "__main__":
    main()
