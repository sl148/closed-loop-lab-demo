"""Solver sanity on standard JSSP / FJSP instances: makespan vs known optimum.

    uv run python scripts/bench_solvers.py [--time 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from labloop.schedule.benchmarks import load_fjsp, load_jssp
from labloop.schedule.cpsat import CPSATScheduler
from labloop.schedule.serial import FIFOScheduler, PriorityScheduler
from labloop.schedule.validator import assert_valid

ROOT = Path(__file__).parents[1] / "benchmarks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=20.0)
    args = ap.parse_args()
    optima = json.loads((ROOT / "optima.json").read_text())
    rows = []
    for kind, loader, ext in (("jssp", load_jssp, "txt"), ("fjsp", load_fjsp, "fjs")):
        for name, opt in optima[kind].items():
            prob = loader(ROOT / kind / f"{name}.{ext}")
            for sched_ in (FIFOScheduler(), PriorityScheduler("lwr"), CPSATScheduler(args.time)):
                s = sched_.schedule(prob)
                assert_valid(prob, s)
                gap = 100 * (s.makespan - opt) / opt
                rows.append({"instance": name, "solver": s.solver, "makespan": s.makespan, "optimum": opt,
                             "gap_pct": round(gap, 2), "time_s": round(s.solve_time_s, 2), "status": s.status})
                print(f"{name:6} {s.solver:14} {s.makespan:6} opt={opt:5} gap={gap:6.2f}%  {s.solve_time_s:6.2f}s {s.status}")
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "bench_solvers.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
