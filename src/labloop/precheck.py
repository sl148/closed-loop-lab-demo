"""Pre-launch check (M9): before a campaign runs, does its plan hold up under the constraints?

1. Static: each constraint against the shortest the workflow can do on this deck and protocol
   (seeded cells cannot reach the incubator within 15 min if the dispense alone takes 19).
2. Motion: every arm route the runs used was checked for collisions in MuJoCo
   (results/sim3d/motion_costs.json, when the config uses the 3D calibration).
3. Monte Carlo: N simulated campaigns with the lab's failure model and duration variance. A
   random designer stands in for BO (cheap, and the plan's risk does not depend on which wells are
   chosen). The hidden physical limits are not used: the check knows only what the lab was told.
   Reports, per constraint, how often execution broke it, plus makespan and invalid plates.
4. Gate: launch when nothing conflicts, no route collides, and every hard constraint holds in at least
   1 - max_violation_rate of the plates (or switches) it covers.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np

from labloop.constraints.audit import audit
from labloop.constraints.compile import ORDER, conflicts_between, load_constraints
from labloop.constraints.spec import STEP_OP, DeadlineFromRelease, MaxBetween
from labloop.loop import run_campaign


def _run(args):
    cfg, seed = args
    return run_campaign(cfg, seed)


def step_minutes(runs: list[dict]) -> dict[str, float]:
    """Typical executed duration of each workflow step (median over plates and runs)."""
    durs: dict[str, list[int]] = {s: [] for s in ORDER}
    for run in runs:
        for r in run["schedule"]:
            job, _, suffix = r["op"].partition(".")
            if job in run["plates"]:
                for step, op in STEP_OP.items():
                    if suffix == op:
                        durs[step].append(r["end"] - r["start"])
    return {s: float(median(v)) for s, v in durs.items() if v}


def static_conflicts(specs: list, dur: dict[str, float]) -> list[dict]:
    """Constraints no schedule can meet: the steps they span take longer than they allow."""
    out = []
    for c in specs:
        need, why = None, ""
        if isinstance(c, MaxBetween):
            steps = ORDER[ORDER.index(c.a) + (0 if c.a_anchor == "start" else 1):ORDER.index(c.b)]
            need, why = sum(dur.get(s, 0) for s in steps), " + ".join(f"{s} ~{dur.get(s, 0):.0f}" for s in steps)
        elif isinstance(c, DeadlineFromRelease):
            steps = ORDER[:ORDER.index(c.step)]
            need, why = sum(dur.get(s, 0) for s in steps), " + ".join(f"{s} ~{dur.get(s, 0):.0f}" for s in steps)
        if need is not None and c.minutes < need:
            out.append({"constraint": c.id, "allows_min": c.minutes, "needs_min": round(need), "because": why,
                        "hard": c.hard})
    return out


def motion_check(cfg: dict, runs: list[dict]) -> dict:
    """Arm routes used by the runs vs the MuJoCo route checks."""
    used = sorted({r["op"].partition(".")[2] for run in runs for r in run["schedule"] if r["kind"] == "transfer"})
    if not cfg.get("sim3d"):
        return {"checked": False, "routes_used": len(used), "colliding": []}
    checks = json.loads((Path(cfg["sim3d"].get("dir", "results/sim3d")) / "motion_costs.json").read_text())["checks"]
    return {"checked": True, "routes_used": len(used), "routes_checked": len(checks),
            "colliding": sorted(r for r, c in checks.items() if c["collisions"])}


def precheck(cfg: dict, n: int = 10, seed0: int = 10_000, workers: int = 1, max_violation_rate: float = 0.05,
             designer: str = "random") -> dict:
    comp = load_constraints(cfg.get("constraints"))
    sim = {k: v for k, v in cfg.items() if k not in ("truth", "belief_log", "landscape_eval", "review")}
    sim["designer"] = designer
    sim["campaigns"] = [{k: v for k, v in c.items() if k != "designer"} for c in cfg.get("campaigns", [{}])]
    jobs = [(sim, seed0 + i) for i in range(n)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            runs = list(ex.map(_run, jobs))
    else:
        runs = [_run(j) for j in jobs]

    conflicts = static_conflicts(comp.specs, step_minutes(runs)) + [
        {"constraint": "pair", "because": why, "hard": True} for why in conflicts_between(comp.specs)]
    risk: dict[str, dict] = {}
    for run in runs:
        for cid, a in audit(comp.specs, run).items():
            r = risk.setdefault(cid, {"n": 0, "violated": 0, "worst_over_min": 0})
            r["n"] += a["n"]
            r["violated"] += a["violated"]
            r["worst_over_min"] = max(r["worst_over_min"], a["worst_over_min"])
    for r in risk.values():
        r["rate"] = round(r["violated"] / r["n"], 4) if r["n"] else 0.0
    hard = {c.id for c in comp.specs if c.hard}
    at_risk = sorted(cid for cid, r in risk.items() if cid in hard and r["rate"] > max_violation_rate)
    span = [run["makespan"] / 60 for run in runs]
    motion = motion_check(cfg, runs)
    return {
        "ok": not [c for c in conflicts if c["hard"]] and not at_risk and not motion["colliding"],
        "conflicts": conflicts,
        "at_risk": at_risk,
        "motion": motion,
        "risk": risk,
        "n_runs": n,
        "makespan_h": {"p50": round(float(np.percentile(span, 50)), 1), "p90": round(float(np.percentile(span, 90)), 1)},
        "invalid_plates": round(float(np.mean([run["invalid_plates"] for run in runs])), 2),
        "lag_slips": round(float(np.mean([sum(1 for e in run["events"] if e["kind"] == "lag_slip") for run in runs])), 1),
    }
