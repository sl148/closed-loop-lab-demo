"""Online repair scheduler (contribution A).

On a new batch, an overrun, a failure, or a breakdown the lab hands over the live problem
(finished ops removed, running ops pinned) and the previous plan. Instead of re-solving from
scratch it:
1. frees a neighborhood: new jobs, jobs whose prior placement is now invalid (bounds,
   durations, downtime, broken lags), jobs in capacity conflicts with pinned work, and jobs
   due to start within `window_min` of now;
2. fixes every other op to its prior (start, end, device);
3. solves the neighborhood with CP-SAT, hinted with the prior plan, minimizing
   makespan + flow + stability (minutes moved);
4. falls back to a wider window, then a full warm-started solve, then the serial scheduler.
"""

from __future__ import annotations

import time

from labloop.schedule.cpsat import CPSATScheduler, Objective, build_and_solve
from labloop.schedule.model import Problem, Schedule
from labloop.schedule.serial import PriorityScheduler


def _prior_dur_ok(problem: Problem, prior: Schedule, oid: str) -> bool:
    res = prior.assigned.get(oid)
    if res not in problem.alternatives(oid):
        return False
    lo, hi = problem.dur_range(oid, res)
    return lo <= prior.end[oid] - prior.start[oid] <= hi


def invalid_jobs(problem: Problem, prior: Schedule, free: set[str]) -> set[str]:
    """Jobs whose prior placement can't be kept as is."""
    bad: set[str] = set()
    for oid, op in problem.ops.items():
        if op.job in free or oid in problem.pinned:
            continue
        if oid not in prior.start:
            bad.add(op.job)
            continue
        s, e, res = prior.start[oid], prior.end[oid], prior.assigned.get(oid)
        if s < problem.earliest(oid) or s > problem.start_ub.get(oid, s) or e < problem.end_lb.get(oid, e):
            bad.add(op.job)
        elif not _prior_dur_ok(problem, prior, oid):
            bad.add(op.job)
        elif any(s < b and e > a for a, b in problem.unavailable.get(res, [])):
            bad.add(op.job)
    for lag in problem.lags:
        a, b = lag.before, lag.after
        ja, jb = problem.ops[a].job, problem.ops[b].job
        if ja in free or jb in free:
            continue
        # Pinned ops keep their prior start but may have a new end estimate.
        ea = prior.end.get(a)
        if a in problem.pinned and a in problem.dur_override:
            ea = problem.start_lb[a] + problem.dur_override[a]
        if a in problem.end_lb and ea is not None:
            ea = max(ea, problem.end_lb[a])
        if lag.ref == "start":
            ea = problem.start_lb[a] if a in problem.pinned else prior.start.get(a)
        sb = prior.start.get(b)
        if ea is None or sb is None:
            bad |= {ja, jb}
            continue
        gap = sb - ea
        if gap < lag.min_lag or (lag.max_lag is not None and gap > lag.max_lag):
            bad.add(jb if b not in problem.pinned else ja)
    # Setups between kept ops (e.g. a new cell line now runs right after another on an LH).
    seqs: dict[str, list[str]] = {}
    for oid in problem.setup_ops():
        if problem.ops[oid].job not in free and oid in prior.start:
            seqs.setdefault(problem.pinned.get(oid) or prior.assigned.get(oid), []).append(oid)
    for r, oids in seqs.items():
        seq = sorted(oids, key=lambda o: prior.start[o])
        last = problem.res_last.get(r)
        chain = ([(last[0], last[1], None)] if last else []) + [(problem.family[o], prior.end[o], o) for o in seq]
        for (fa, ea, _), (fb, _, b) in zip(chain, chain[1:]):
            if fa != fb and prior.start[b] - ea < problem.setup_min[problem.ops[b].kind] and b not in problem.pinned:
                bad.add(problem.ops[b].job)
    return bad


def capacity_conflicts(problem: Problem, prior: Schedule, free: set[str]) -> set[str]:
    """Jobs with fixed ops overlapping an overloaded instant (pinned ops can't move)."""
    bad: set[str] = set()

    def placed(oid):  # (start, end, resource) a fixed op keeps
        if oid in problem.pinned:
            r, s = problem.pinned[oid], problem.start_lb[oid]
            d = problem.dur_override.get(oid, problem.dur_range(oid, r)[0])
            return s, max(s + d, problem.end_lb.get(oid, 0), prior.end.get(oid, 0)), r
        if oid in prior.start:
            return prior.start[oid], prior.end[oid], prior.assigned.get(oid)
        return None

    covered = problem.covered()
    spans = []
    for sp in problem.spans:
        if problem.ops[sp.last].job in free:
            continue
        a = placed(sp.first) if sp.first in problem.ops else (problem.span_start[sp.link], 0, None)
        b, k = placed(sp.last), placed(sp.link) if sp.link in problem.ops else None
        if a and b:
            spans.append((sp.res or (k[2] if k else None), a[0], b[1], sp.last))
    for rname, res in problem.resources.items():
        iv = [(s, e, o) for r, s, e, o in spans if r == rname]
        for oid, op in problem.ops.items():
            if op.job in free or oid in covered:
                continue
            p = placed(oid)
            if p is None or p[2] != rname:
                continue
            iv.append((p[0], p[1], oid))
        for a, b in problem.unavailable.get(rname, []):
            iv += [(a, b, None)] * res.capacity
        points = sorted({s for s, _, _ in iv})
        for t in points:
            active = [x for x in iv if x[0] <= t < x[1]]
            if len(active) > res.capacity:
                bad |= {problem.ops[o].job for _, _, o in active if o is not None and o not in problem.pinned}
    return bad


class RepairScheduler:
    def __init__(self, time_limit_s: float = 1.0, window_min: int = 240, full_time_limit_s: float = 5.0,
                 objective: Objective | None = None, workers: int = 8, deterministic: bool = False):
        self.time_limit_s, self.window_min, self.full_time_limit_s = time_limit_s, window_min, full_time_limit_s
        self.objective = objective or Objective(makespan=10, flow=1, stability=1)
        self.workers = workers
        self.deterministic = deterministic  # reproducible solves (cpsat.build_and_solve), for replayable runs
        self.name = "repair"
        self.last_neighborhood = 0

    def neighborhood(self, problem: Problem, prior: Schedule, now: int, window: int) -> set[str]:
        free = {op.job for oid, op in problem.ops.items() if oid not in prior.start}
        # Near-term jobs: the next op that has not started is due inside the window.
        for job in problem.jobs():
            nxt = [o for o in problem.job_ops(job) if o.id not in problem.pinned]
            if nxt and nxt[0].id in prior.start and prior.start[nxt[0].id] <= now + window:
                free.add(job)
        while True:
            more = invalid_jobs(problem, prior, free) | capacity_conflicts(problem, prior, free)
            if more <= free:
                return free
            free |= more

    def _solve(self, problem: Problem, prior: Schedule, free: set[str], limit: float) -> Schedule:
        fixed = {oid: (prior.start[oid], prior.end[oid], prior.assigned[oid])
                 for oid, op in problem.ops.items()
                 if op.job not in free and oid not in problem.pinned and oid in prior.start}
        return build_and_solve(problem, name=self.name, time_limit_s=limit, objective=self.objective,
                               hint=prior, fixed=fixed, stability_ref=prior, workers=self.workers,
                               deterministic=self.deterministic)

    def schedule(self, problem: Problem, prior: Schedule | None = None, now: int = 0) -> Schedule:
        t0 = time.perf_counter()
        if prior is None or not prior.start:
            out = CPSATScheduler(self.full_time_limit_s, self.objective, self.workers, self.deterministic).schedule(problem)
            out.solver = f"{self.name}:full"
            self.last_neighborhood = len(problem.jobs())
            return out
        for window, label in ((self.window_min, "lns"), (self.window_min * 4, "lns-wide")):
            free = self.neighborhood(problem, prior, now, window)
            self.last_neighborhood = len(free)
            out = self._solve(problem, prior, free, self.time_limit_s)
            if out.start and "lag_violations" not in out.status:
                out.solver = f"{self.name}:{label}"
                out.solve_time_s = time.perf_counter() - t0
                return out
        out = CPSATScheduler(self.full_time_limit_s, self.objective, self.workers, self.deterministic).schedule(problem)
        if out.start:
            out.solver = f"{self.name}:full"
        else:
            out = PriorityScheduler("deadline").schedule(problem)
            out.solver = f"{self.name}:serial"
        self.last_neighborhood = len(problem.jobs())
        out.solve_time_s = time.perf_counter() - t0
        return out
