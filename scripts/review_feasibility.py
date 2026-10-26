"""Can the data-side review see the hidden limits? Feasibility gate for M10 (run on the server).

    PYTHONPATH=src python scripts/review_feasibility.py [--variants blind,agent_open,gold]

For each Level 4 run in results/m9/level4/<variant>.json and each seed it reports
- the untreated-control SD (noise alone is about 0.04-0.05),
- how well the schedule factor "cell-line switch without a clean" and the score `over_frac` pick out
  the plates the hidden limit actually contaminated (AUC; the truth events are used here only),
- the association test on all plates (top tests and flags), and
- a prefix curve: after the first k plates read, the scan over every factor (`associate`) and the
  one-sided test of the agent's own changeover hypothesis (`check_hypotheses`); the review acts
  on the hypothesis once two consecutive checks are below alpha.
Writes results/m9/review_feasibility.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from labloop.review import stats
from labloop.review.stats import associate, check_hypotheses, plate_factors, plate_scores

# the agent's brainstorm proposed this in 3 of 3 open runs but found no number for it
CHANGEOVER = {"id": "liquid_handler_changeover_cell_line", "kind": "changeover", "steps": ["dispense"]}


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def seed_report(run: dict, B: int, alpha: float) -> dict:
    scores, factors = plate_scores(run), plate_factors(run)
    dirty = {e["plate"] for e in run["events"] if e["kind"] == "truth" and e.get("limit") == "changeover"}
    pids = [p for p in scores if p in factors]
    sw = {p: factors[p]["lh_switch_uncleaned"] for p in pids}
    tail = {p: scores[p]["over_frac"] for p in pids}
    tests = associate(scores, factors, B=B, alpha=alpha)
    order = sorted(pids, key=lambda p: factors[p]["read_t"])
    prefix = []
    for k in range(8, len(order) + 1, 4):
        sub = plate_scores(run, order[:k])
        a = associate(sub, factors, B=999, alpha=alpha)
        t = [x for x in a if x["score"] == "over_frac" and x["factor"] == "lh_switch_uncleaned"]
        h = check_hypotheses(sub, factors, [CHANGEOVER], alpha=alpha)
        prefix.append({"k": k, "p_holm": t[0]["p_holm"] if t else None, "flag": bool(t and t[0]["flag"]),
                       "any_flag": [f"{x['score']}~{x['factor']}" for x in a if x["flag"]],
                       "hyp_p": h[0]["p"] if h else None, "hyp_flag": bool(h and h[0]["flag"])})
    return {
        "plates": len(pids), "dirty": len(dirty & set(pids)),
        "scale_median": float(np.median([scores[p]["neg_sd"] for p in pids])),
        "scale_clean": float(np.median([scores[p]["neg_sd"] for p in pids if p not in dirty] or [np.nan])),
        "tail_dirty": float(np.mean([tail[p] for p in pids if p in dirty] or [np.nan])),
        "tail_clean": float(np.mean([tail[p] for p in pids if p not in dirty] or [np.nan])),
        "auc_factor": auc([sw[p] for p in pids if p in dirty], [sw[p] for p in pids if p not in dirty]),
        "auc_tail": auc([tail[p] for p in pids if p in dirty], [tail[p] for p in pids if p not in dirty]),
        "top": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t.items()} for t in tests[:6]],
        "flags": [f"{t['score']}~{t['factor']}" for t in tests if t["flag"]],
        "prefix": prefix,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="blind,agent_open,gold")
    ap.add_argument("--dir", default="results/m9/level4")
    ap.add_argument("--B", type=int, default=4999)
    ap.add_argument("--out", default="results/m9/review_feasibility.json")
    ap.add_argument("--k", type=float, default=stats.TAIL_K, help="control SDs above controls that count as implausible")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    stats.TAIL_K = args.k
    out = {}
    for v in args.variants.split(","):
        runs = json.loads((Path(args.dir) / f"{v}.json").read_text())
        out[v] = [seed_report(r, args.B, args.alpha) for r in runs]
        for i, r in enumerate(out[v]):
            first = next((p["k"] for p in r["prefix"] if p["flag"]), None)
            pre = r["prefix"]
            hyp = next((b["k"] for a, b in zip(pre, pre[1:]) if a["hyp_flag"] and b["hyp_flag"]), None)
            print(f"{v:11s} seed{i}: plates {r['plates']:2d} dirty {r['dirty']:2d} scale {r['scale_median']:.3f} "
                  f"over dirty/clean {r['tail_dirty']:.4f}/{r['tail_clean']:.4f} AUC factor {r['auc_factor']} "
                  f"over {r['auc_tail'] and round(r['auc_tail'], 3)} scan flags {r['flags']} (first at {first}) | hypothesis confirmed at {hyp} plates", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
