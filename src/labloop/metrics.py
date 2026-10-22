"""Run-level metrics and seed aggregation."""

from __future__ import annotations

import numpy as np


def time_to_target(progress: list[dict], optimum: float, frac: float = 0.9) -> tuple[float | None, int | None]:
    """(hours, plates read) until the reported best combination's true excess reaches
    `frac` of the optimum; None if never."""
    for p in progress:
        if p["best_true"] is not None and p["best_true"] >= frac * optimum:
            return p["t"] / 60.0, p["plates_read"]
    return None, None


def run_metrics(run: dict) -> dict:
    out = {"makespan_h": run["makespan"] / 60.0}
    util = run.get("utilization", {})
    for kind in ("lh", "reader", "arm", "incubator"):
        vals = [v for k, v in util.items() if k.startswith(kind)]
        if vals:
            out[f"util_{kind}"] = float(np.mean(vals))
    out["idle_h"] = sum(run.get("idle", {}).values()) / 60.0
    rs = run["reschedules"]
    out["n_resched"] = len(rs)
    out["resolve_ms"] = 1000 * float(np.mean([r["time_s"] for r in rs])) if rs else 0.0
    out["resolve_p95_ms"] = 1000 * float(np.percentile([r["time_s"] for r in rs], 95)) if rs else 0.0
    out["moved_ops"] = float(np.mean([r["moved"] for r in rs if r["reasons"] != ["new_batch"]] or [0]))
    out["n_failures"] = sum(1 for e in run["events"] if e["kind"] == "failure")
    out["invalid_plates"] = run.get("invalid_plates", 0)
    reads = [e for e in run["events"] if e["kind"] == "read"]
    out["corrupt_accepted"] = sum(e.get("corrupt_accepted", 0) for e in reads)
    out["wells_invalid"] = sum(e["n"] - e["n_valid"] for e in reads)
    fails = [e for e in run["events"] if e["kind"] == "failure"]
    out["recovery_min"] = sum(a["minutes"] for e in fails for a in e["decision"]["actions"]
                              if a["type"] in ("retry", "substitute", "human"))
    out["llm_fallbacks"] = sum(1 for e in fails if "->rules" in e["decision"]["policy"])
    regrets, ttt, ttp, plates, reagent, rmse = [], [], [], [], [], []
    for c in run["campaigns"].values():
        fb = c["final_best"]
        regrets.append(fb["regret"] if fb and fb["regret"] is not None else c["optimum"]["excess"])
        h, p = time_to_target(c["progress"], c["optimum"]["excess"])
        ttt.append(h if h is not None else np.nan)
        ttp.append(p if p is not None else np.nan)
        plates.append(c["plates"])
        reagent.append(c["reagent_nmol"])
        if c["landscape"] and c["landscape"][-1]["rmse"] is not None:
            rmse.append(c["landscape"][-1]["rmse"])
    out["regret"] = float(np.mean(regrets))
    out["ttt_h"] = float(np.nanmean(ttt)) if not np.all(np.isnan(ttt)) else np.nan
    out["ttt_plates"] = float(np.nanmean(ttp)) if not np.all(np.isnan(ttp)) else np.nan
    out["hit"] = float(np.mean([not np.isnan(x) for x in ttt]))
    out["plates"] = float(np.sum(plates))
    out["reagent_nmol"] = float(np.sum(reagent))
    if rmse:
        out["rmse"] = float(np.mean(rmse))
    return out


def summarize(runs: list[dict]) -> dict:
    """Mean (and std for the headline metrics) over seeds, rounded for printing."""
    ms = [run_metrics(r) for r in runs]
    keys = list(dict.fromkeys(k for m in ms for k in m))
    out = {}
    for k in keys:
        vals = np.array([m.get(k, np.nan) for m in ms], float)
        if np.all(np.isnan(vals)):
            out[k] = None
            continue
        out[k] = round(float(np.nanmean(vals)), 3)
        if k in ("makespan_h", "regret", "ttt_h", "resolve_ms", "rmse"):
            out[k + "_sd"] = round(float(np.nanstd(vals)), 3)
    return out
