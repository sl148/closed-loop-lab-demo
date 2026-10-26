"""Per-rollout data behind the pre-launch check (M9 web view).

    PYTHONPATH=src python scripts/precheck_rollouts.py --config configs/level4.yaml \
        --set deck=configs/deck_level4.json --n 10 --workers 10

Runs the check's short rollouts twice, once with the lab told the gold constraints and once
without, and scores each rollout against the gold constraints. Writes
results/m9/precheck_rollouts.json: per condition, per constraint, the violation rate of each rollout.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from labloop.constraints.audit import audit
from labloop.constraints.compile import load_constraints
from labloop.precheck import _run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--gold", default="configs/constraints_gold.yaml")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/m9/precheck_rollouts.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    specs = load_constraints(args.gold).specs
    # The check's own setup (precheck.precheck): random designer, two rounds, no hidden limits.
    sim = {k: v for k, v in cfg.items() if k not in ("truth", "belief_log", "landscape_eval")}
    sim.update(designer="random",
               campaigns=[{**{k: v for k, v in c.items() if k != "designer"}, "rounds": min(2, c.get("rounds", 2))}
                          for c in cfg.get("campaigns", [{}])])
    out = {"n": args.n, "gate": 0.05, "constraints": {c.id: {"minutes": c.minutes, "kind": c.kind} for c in specs}}
    for name, cons in (("told", args.gold), ("not_told", None)):
        jobs = [({**sim, "constraints": cons}, 10_000 + i) for i in range(args.n)]
        with ProcessPoolExecutor(args.workers) as ex:
            runs = list(ex.map(_run, jobs))
        out[name] = {cid: [a["rate"] for a in (audit(specs, r)[cid] for r in runs)] for cid in out["constraints"]}
        out[name + "_n"] = {cid: [audit(specs, r)[cid]["n"] for r in runs] for cid in out["constraints"]}
        print(name, {k: sum(v) / len(v) for k, v in out[name].items()}, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
