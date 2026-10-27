"""Method baselines for the Results tab (web/data/baselines.json).

    PYTHONPATH=src python scripts/export_baselines.py

Ours (gold_v3: throughput-aware BO + CP-SAT repair) against discovery and scheduling baselines on the same
lab (scripts/run_baselines.sh): every run knows all real limits and uses the gold labware and prep, so only
the method differs. Fresh seeds (10-17), none used while choosing our settings. Discovery curves: the reported best pair's true synergy as a share of the
best possible, mean over both campaigns and all seeds, against lab hours.

    --suffix agentlab --out web/data/baselines_lab.json   # the web replay's lab (LAB=agent run_baselines.sh),
                                                          # overlaid on the Results tab's running synergy chart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

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


def hold_hours(progress: list[dict], optimum: float, end_h: float) -> float:
    """First time from which the reported best stays >= 90% of the optimum to the end (censored: end + 1)."""
    ok = [p["best_true"] is not None and p["best_true"] >= 0.9 * optimum for p in progress]
    if not ok or not ok[-1]:
        return end_h + 1
    i = len(ok)
    while i > 0 and ok[i - 1]:
        i -= 1
    return progress[i]["t"] / 60.0


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
    ap.add_argument("--suffix", default="v3", help="variant suffix: v3 (gold lab) or agentlab (the replay's lab)")
    args = ap.parse_args()
    res, out_path = Path(args.res), Path(args.out)
    name = lambda v: v[: -len("_v3")] + "_" + args.suffix  # base_ours_v3 -> base_ours_<suffix>
    runs = {k: json.loads((res / f"{name(v)}.json").read_text()) for k, v, *_ in METHODS if (res / f"{name(v)}.json").exists()}
    tmax = max(r["makespan"] for rs in runs.values() for r in rs)
    out = {"step_min": STEP_MIN, "seeds": len(next(iter(runs.values()))), "methods": []}
    hold_by: dict[str, dict] = {}
    for key, variant, label, group, desc in METHODS:
        if key not in runs:
            print("missing", name(variant))
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
        by_run = {(r["seed"], n): hold_hours(c["progress"], c["optimum"]["excess"], r["makespan"] / 60.0)
                  for r in rs for n, c in r["campaigns"].items()}
        holds = list(by_run.values())
        hold_by[key] = by_run
        m["ttt_h"] = round(float(np.median(holds)), 1)  # per campaign, as scripts/confirm_eval.py: median hours to hold 90%
        m["ttt_iqr"] = [round(float(np.percentile(holds, 25)), 1), round(float(np.percentile(holds, 75)), 1)]
        print(f"{label:16s} found {m['final']:.2f} hits {m['hits']}/{m['of']} ttt {m['ttt_h']} makespan {m['makespan_h']} "
              f"plates/day {m['plates_per_day']} moved {m['moved_ops']} replan {m['replan_ms']} ms invalid {m['invalid_plates']}")
    # paired by (seed, campaign), as scripts/confirm_eval.py: does ours hold 90% sooner? (one-sided Wilcoxon)
    seeds = sorted({s for k in hold_by.values() for s, _ in k})
    out["seed_range"] = [seeds[0], seeds[-1]]
    for m in out["methods"]:  # a seed a method could not finish (e.g. a dispatch deadlock) is reported, not hidden
        mine = {s for s, _ in hold_by[m["key"]]}
        if set(seeds) - mine:
            m["missing_seeds"] = sorted(set(seeds) - mine)
    for m in out["methods"]:
        if m["key"] == "ours":
            continue
        keys = sorted(set(hold_by["ours"]) & set(hold_by[m["key"]]))
        a, b = np.array([hold_by["ours"][k] for k in keys]), np.array([hold_by[m["key"]][k] for k in keys])
        p = float(wilcoxon(a, b, alternative="less", zero_method="zsplit").pvalue)
        m["vs_ours"] = {"faster": int((a < b).sum()), "of": len(keys), "p": round(p, 4),
                        "beats": bool(np.median(a) < np.median(b) and p < 0.05)}
        print(f"  ours vs {m['label']}: faster {m['vs_ours']['faster']}/{len(keys)}, p = {p:.4f}, {'BEATS' if m['vs_ours']['beats'] else 'tie'}")
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    print(out_path, f"{out_path.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
