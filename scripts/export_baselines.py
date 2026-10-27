"""Method baselines for the Results tab (web/data/baselines.json).

    PYTHONPATH=src python scripts/export_baselines.py

Ours (gold_v3: throughput-aware BO + CP-SAT repair) against discovery and scheduling baselines on the same
lab (scripts/run_baselines.sh): every run knows all real limits and uses the gold labware and prep, so only
the method differs. Fresh seeds (10-17), none used while choosing our settings. Discovery curves: the reported best pair's true synergy as a share of the
best possible, mean over both campaigns and all seeds, against lab hours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from labloop.metrics import summarize

RES = Path("results/m9/level4")
OUT = Path("web/data/baselines.json")
# key, run variant, label, which comparison it belongs to, one line on what it does
METHODS = [
    ("ours", "base_ours_v3", "Ours", "both", "Throughput-aware BO picks each plate's drugs, cell line and read time by expected gain per lab hour; CP-SAT repairs the schedule around each disruption"),
    ("bo", "base_bo_v3", "Plain BO", "discovery", "Same model and Thompson sampling, but cell lines and read times taken in turn"),
    ("lhs", "base_lhs_v3", "Latin hypercube", "discovery", "Space-filling doses, no model"),
    ("random", "base_random_v3", "Random", "discovery", "Uniformly random conditions"),
    ("grid", "base_grid_v3", "Dose grid", "discovery", "A fixed dose-response grid, the way it is planned by hand"),
    ("fifo", "base_fifo_v3", "FIFO dispatch", "schedule", "Each machine runs the next waiting step, first come first served"),
    ("priority", "base_priority_v3", "Priority rules", "schedule", "Dispatching by least work remaining"),
    ("cpsat", "base_cpsat_v3", "Full re-solve", "schedule", "CP-SAT re-solves the whole schedule at every event, same time budget"),
]
DEFAULT_ON = {"bo", "fifo"}  # one baseline per chart is shown on load
STEP_MIN = 60


def synergy_curve(runs: list[dict], tmax: int) -> list[float]:
    grid = np.arange(0, tmax + STEP_MIN, STEP_MIN)
    curves = []
    for r in runs:
        for c in r["campaigns"].values():
            opt, v, out, pts = c["optimum"]["excess"], 0.0, [], c["progress"]
            i = 0
            for t in grid:
                while i < len(pts) and pts[i]["t"] <= t:
                    if pts[i]["best_true"] is not None:
                        v = min(1.0, max(0.0, pts[i]["best_true"] / opt))
                    i += 1
                out.append(v)
            curves.append(out)
    return [round(float(x), 4) for x in np.mean(curves, axis=0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(RES), help="folder with the runs (default results/m9/level4)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    res, out_path = Path(args.res), Path(args.out)
    runs = {k: json.loads((res / f"{v}.json").read_text()) for k, v, *_ in METHODS if (res / f"{v}.json").exists()}
    tmax = max(r["makespan"] for rs in runs.values() for r in rs)
    out = {"step_min": STEP_MIN, "seeds": len(next(iter(runs.values()))), "methods": []}
    for key, variant, label, group, desc in METHODS:
        if key not in runs:
            print("missing", variant)
            continue
        rs, s = runs[key], summarize(runs[key])
        reads = [sum(1 for e in r["events"] if e["kind"] == "read") for r in rs]
        out["methods"].append({
            "key": key, "label": label, "group": group, "desc": desc, "default": key in DEFAULT_ON,
            "synergy": synergy_curve(rs, tmax),
            "final": round(1 - s["regret"], 3), "hits": round(s["hit"] * 2 * len(rs)), "of": 2 * len(rs), "ttt_h": s["ttt_h"],
            "makespan_h": s["makespan_h"], "makespan_sd": s.get("makespan_h_sd"),
            "plates_per_day": round(float(np.mean([n / (r["makespan"] / 1440) for n, r in zip(reads, rs)])), 2),
            "idle_h": s["idle_h"], "moved_ops": s["moved_ops"], "replan_ms": s["resolve_ms"],
            "invalid_plates": s["invalid_plates"], "replans": s["n_resched"],
        })
        m = out["methods"][-1]
        below = [i for i, v in enumerate(m["synergy"]) if v < 0.9]  # the mean curve reaching 90% and staying there (the answer
        cross = (below[-1] + 1 if below else 0) if m["synergy"][-1] >= 0.9 else None  # the lab hands over no longer gets worse)
        m["ttt_h"] = round(cross * STEP_MIN / 60, 1) if cross is not None else None
        print(f"{label:16s} found {m['final']:.2f} hits {m['hits']}/{m['of']} ttt {m['ttt_h']} makespan {m['makespan_h']} "
              f"plates/day {m['plates_per_day']} moved {m['moved_ops']} replan {m['replan_ms']} ms invalid {m['invalid_plates']}")
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    print(out_path, f"{out_path.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
