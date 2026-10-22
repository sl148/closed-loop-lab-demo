"""CP-SAT scheduler: optional intervals per alternative resource, NoOverlap for unary
resources, Cumulative for capacity resources, min/max time lags, absolute bounds, downtime.

`build_and_solve` is shared with the repair scheduler, which adds fixed ops, hints, and a
stability term.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from labloop.schedule.model import Problem, Schedule
from labloop.schedule.serial import FIFOScheduler

LAG_PENALTY = 1000  # per minute of max-lag slack, only in soft mode


@dataclass
class Objective:
    makespan: int = 1
    flow: int = 0  # sum of job completion times (earlier reads -> earlier data)
    stability: int = 0  # sum |start - prior start| over free ops


def _horizon(problem: Problem, ub_hint: int | None) -> int:
    base = max([problem.earliest(o) for o in problem.ops] + [0])
    total = sum(max(problem.dur_range(o, r)[1] for r in problem.alternatives(o)) for o in problem.ops)
    total += sum(l.min_lag for l in problem.lags)
    h = base + total + max((b for w in problem.unavailable.values() for _, b in w), default=0)
    if ub_hint is not None:
        h = min(h, ub_hint)
    return max(h, base + 1) + 1


def build_and_solve(problem: Problem, *, name: str, time_limit_s: float, objective: Objective,
                    hint: Schedule | None = None, fixed: dict[str, tuple[int, int, str]] | None = None,
                    stability_ref: Schedule | None = None,
                    soft_lags: bool = False, workers: int = 8, seed: int = 0) -> Schedule:
    t_wall = time.perf_counter()
    fixed = fixed or {}
    ub_hint = int(hint.makespan * 1.5) + 600 if hint and hint.start else None
    H = _horizon(problem, None if soft_lags else ub_hint)
    if hint and hint.start:
        H = max(H, hint.makespan + 1)
    m = cp_model.CpModel()
    S, E, P = {}, {}, {}
    covered = problem.covered()
    by_res: dict[str, list] = {r: [] for r in problem.resources}

    for oid, op in problem.ops.items():
        lo = problem.earliest(oid)
        hi = problem.start_ub.get(oid, H)
        if oid in fixed:
            lo = hi = fixed[oid][0]
        s = m.NewIntVar(lo, max(lo, hi), f"s_{oid}")
        e = m.NewIntVar(0, H + 10**6 if soft_lags else max(H, lo + 1), f"e_{oid}")
        if oid in problem.end_lb:
            m.Add(e >= problem.end_lb[oid])
        alts = problem.alternatives(oid)
        if oid in fixed:
            alts = (fixed[oid][2],)
            m.Add(e == fixed[oid][1])
        lits = []
        for r in alts:
            dlo, dhi = problem.dur_range(oid, r)
            d = m.NewIntVar(dlo, dhi, f"d_{oid}_{r}")
            if len(alts) == 1:
                p = m.NewConstant(1)
                iv = m.NewIntervalVar(s, d, e, f"iv_{oid}_{r}")
            else:
                p = m.NewBoolVar(f"p_{oid}_{r}")
                iv = m.NewOptionalIntervalVar(s, d, e, p, f"iv_{oid}_{r}")
            lits.append(p)
            P[(oid, r)] = p
            if oid not in covered:  # a span holds this op's resource instead
                by_res[r].append(iv)
        if len(alts) > 1:
            m.AddExactlyOne(lits)
        S[oid], E[oid] = s, e

    for i, sp in enumerate(problem.spans):
        s0 = S[sp.first] if sp.first in problem.ops else problem.span_start[sp.link]
        size = m.NewIntVar(0, H + 10**6, f"span_d_{i}")
        if sp.res:
            by_res[sp.res].append(m.NewIntervalVar(s0, size, E[sp.last], f"span_{i}"))
            continue
        alts = (fixed[sp.link][2],) if sp.link in fixed else problem.alternatives(sp.link)
        for r in alts:
            by_res[r].append(m.NewOptionalIntervalVar(s0, size, E[sp.last], P[(sp.link, r)], f"span_{i}_{r}"))

    for r, windows in problem.unavailable.items():
        for i, (a, b) in enumerate(windows):
            for k in range(problem.resources[r].capacity if problem.resources[r].capacity > 1 else 1):
                by_res[r].append(m.NewIntervalVar(a, b - a, b, f"down_{r}_{i}_{k}"))

    for r, ivs in by_res.items():
        if not ivs:
            continue
        cap = problem.resources[r].capacity
        if cap == 1:
            m.AddNoOverlap(ivs)
        else:
            m.AddCumulative(ivs, [1] * len(ivs), cap)

    penalties = []
    for lag in problem.lags:
        gap = S[lag.after] - E[lag.before]
        m.Add(gap >= lag.min_lag)
        if lag.max_lag is not None:
            if soft_lags:
                slack = m.NewIntVar(0, H + 10**6, f"slack_{lag.before}")
                m.Add(gap <= lag.max_lag + slack)
                penalties.append(slack)
            else:
                m.Add(gap <= lag.max_lag)

    makespan = m.NewIntVar(0, H + 10**6, "makespan")
    m.AddMaxEquality(makespan, list(E.values()))
    terms = [objective.makespan * makespan]
    if objective.flow:
        for job in problem.jobs():
            last = problem.job_ops(job)[-1].id
            terms.append(objective.flow * E[last])
    if objective.stability and stability_ref:
        for oid in problem.ops:
            if oid in stability_ref.start and oid not in fixed and oid not in problem.pinned:
                dev = m.NewIntVar(0, H + 10**6, f"dev_{oid}")
                m.AddAbsEquality(dev, S[oid] - stability_ref.start[oid])
                terms.append(objective.stability * dev)
    if penalties:
        terms.append(LAG_PENALTY * sum(penalties))
    m.Minimize(sum(terms))

    if hint:
        for oid in problem.ops:
            if oid in hint.start and oid not in fixed:
                m.AddHint(S[oid], hint.start[oid])
                r = hint.assigned.get(oid)
                if r is not None and (oid, r) in P and len(problem.alternatives(oid)) > 1:
                    for (o2, r2), p in P.items():
                        if o2 == oid:
                            m.AddHint(p, 1 if r2 == r else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(m)
    out = Schedule(solver=name, status=solver.StatusName(status))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for oid in problem.ops:
            out.start[oid] = solver.Value(S[oid])
            out.end[oid] = solver.Value(E[oid])
            alts = (fixed[oid][2],) if oid in fixed else problem.alternatives(oid)
            out.assigned[oid] = next(r for r in alts if solver.Value(P[(oid, r)]))
        if penalties and any(solver.Value(p) for p in penalties):
            out.status += "+lag_violations"
    out.solve_time_s = time.perf_counter() - t_wall
    return out


class CPSATScheduler:
    """From-scratch CP-SAT solve, warm-started from a FIFO schedule. Falls back to soft
    max-lags when the hard model is infeasible, then to the FIFO schedule itself."""

    def __init__(self, time_limit_s: float = 10.0, objective: Objective | None = None, workers: int = 8):
        self.time_limit_s = time_limit_s
        self.objective = objective or Objective(makespan=1)
        self.workers = workers
        self.name = "cpsat"

    def schedule(self, problem: Problem, prior: Schedule | None = None, now: int = 0) -> Schedule:
        seed = FIFOScheduler().schedule(problem)
        hint = seed if seed.status == "feasible" else None
        out = build_and_solve(problem, name=self.name, time_limit_s=self.time_limit_s,
                              objective=self.objective, hint=hint, workers=self.workers)
        if not out.start:
            out = build_and_solve(problem, name=self.name, time_limit_s=self.time_limit_s,
                                  objective=self.objective, hint=hint, soft_lags=True, workers=self.workers)
        if not out.start:
            seed.solver = f"{self.name}->fifo"
            return seed
        out.solve_time_s += seed.solve_time_s
        return out
