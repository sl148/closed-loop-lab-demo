"""Contribution A at scale: online repair vs from-scratch CP-SAT as the live problem grows.

For N plates in flight (2 liquid handlers, 2 readers, 1 arm, 2 incubators): plan with CP-SAT,
execute until 30% of the plan, inject a new batch of N/4 plates, capture the live problem
(finished ops removed, running ops pinned), then solve it with
  - repair (1 s neighborhood limit, warm start, stability term)
  - full CP-SAT with 1 s / 5 s / 30 s limits (warm-started from FIFO)
and compare objective (10 x makespan + flow), wall time, and ops moved vs the previous plan.

    uv run python scripts/repair_scaling.py [--sizes 8 16 32 64] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import simpy
import yaml

from labloop.design.base import build_space
from labloop.execution.durations import DurationModel
from labloop.execution.lab import Lab
from labloop.execution.reader import PlateReader
from labloop.failures.policies.rules import RulePolicy
from labloop.layout.layouts import build_layout
from labloop.oracle.noise import NoiseModel
from labloop.oracle.surface import build_oracle
from labloop.schedule.cpsat import CPSATScheduler, Objective
from labloop.schedule.repair import RepairScheduler
from labloop.schedule.validator import violations
from labloop.schedule.workflow import build_resources
from labloop.types import PlateSpec

OBJ = Objective(makespan=10, flow=1)


def objective(prob, s) -> int:
    flow = sum(max(s.end[o.id] for o in prob.job_ops(j)) for j in prob.jobs())
    return 10 * s.makespan + flow


def plates_for(n: int, rng, layout, space, prefix: str):
    conds = []
    groups = [(cl, t) for cl in space.cell_lines for t in space.timepoints_h]
    out = []
    for i in range(n):
        g = groups[rng.integers(len(groups))]
        pool = [c for c in space.combos if c.group() == g]
        pick = [pool[k] for k in rng.choice(len(pool), min(layout.capacity, len(pool)), replace=False)]
        out += layout.place(pick, campaign=f"{prefix}{i}_")
    return out


def instance(n: int, seed: int):
    cfg = yaml.safe_load(Path("configs/level2.yaml").read_text())
    space = build_space(cfg["space"])
    rng = np.random.default_rng(seed)
    layout = build_layout("optimized", PlateSpec(16, 24), {}, seed)
    env = simpy.Environment()
    res = build_resources({"liquid_handlers": 2, "readers": 2, "arms": 1, "incubators": 2, "incubator_slots": 8})
    reader = PlateReader(build_oracle(cfg["oracle"]), NoiseModel())  # same cell lines as the plates
    lab = Lab(env, res, DurationModel(), reader, CPSATScheduler(10, OBJ), RulePolicy(), rng)
    noop = lambda *a: None
    lab.submit(plates_for(n, rng, layout, space, "A"), "A", noop, noop)
    env.run(until=1)  # initial plan
    horizon = lab.plan.makespan
    env.run(until=int(0.3 * horizon))
    lab.submit(plates_for(max(1, n // 4), rng, layout, space, "B"), "B", noop, noop)
    return lab.build_problem()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    rows = []
    for n in args.sizes:
        for seed in range(args.seeds):
            prob, prior = instance(n, seed)
            now = min(prob.start_lb.values())
            solvers = [("repair-1s", RepairScheduler(1.0, 240, 5.0, Objective(10, 1, 1)))] + \
                      [(f"full-{t}s", CPSATScheduler(t, OBJ)) for t in (1, 5, 30)]
            for name, sch in solvers:
                s = sch.schedule(prob, prior=prior, now=now)
                ok = not [e for e in violations(prob, s) if "->" not in e and "upper bound" not in e]
                moved = sum(1 for o in prior.start if o in s.start and o not in prob.pinned and s.start[o] != prior.start[o])
                rows.append({"plates": n, "seed": seed, "ops": len(prob.ops), "solver": name, "obj": objective(prob, s),
                             "makespan_h": round(s.makespan / 60, 2), "time_s": round(s.solve_time_s, 3),
                             "moved": moved, "status": s.status, "valid": ok})
                print(rows[-1], flush=True)
    out = Path("results/repair_scaling.json")
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)
    by = {}
    for r in rows:
        by.setdefault((r["plates"], r["solver"]), []).append(r)
    print(f"{'plates':>6} {'solver':>10} {'ops':>5} {'obj gap% vs best':>17} {'time s':>8} {'moved':>6}")
    for n in args.sizes:
        best = {s: min(r["obj"] for r in rows if r["plates"] == n and r["seed"] == s) for s in range(args.seeds)}
        for name in ("repair-1s", "full-1s", "full-5s", "full-30s"):
            rs = by[(n, name)]
            gap = np.mean([100 * (r["obj"] - best[r["seed"]]) / best[r["seed"]] for r in rs])
            print(f"{n:>6} {name:>10} {np.mean([r['ops'] for r in rs]):>5.0f} {gap:>17.2f} "
                  f"{np.mean([r['time_s'] for r in rs]):>8.2f} {np.mean([r['moved'] for r in rs]):>6.1f}")


if __name__ == "__main__":
    main()
