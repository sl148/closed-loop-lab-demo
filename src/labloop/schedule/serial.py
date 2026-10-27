"""Serial schedule generation: place whole job chains one at a time, earliest-feasible
(with gap filling), in a given job order. FIFO and priority-list schedulers differ only in
the order they pass in. Each op goes to the alternative resource that finishes it first."""

from __future__ import annotations

import time

from labloop.schedule.model import Op, Problem, Schedule

SHIFT_STEP = 15  # minutes to delay a job's start when an incubation stretch does not fit


class Timeline:
    """Committed intervals per resource; supports cumulative capacity, downtime and
    sequence-dependent setups between op families."""

    def __init__(self, problem: Problem):
        self.cap = {r: res.capacity for r, res in problem.resources.items()}
        self.busy: dict[str, list[tuple[int, int]]] = {r: [] for r in problem.resources}
        for r, windows in problem.unavailable.items():
            for a, b in windows:
                self.busy[r] += [(a, b)] * self.cap[r]
        self.fam: dict[str, list[tuple[int, int, str]]] = {r: [] for r in problem.resources}
        for r, (f, t) in problem.res_last.items():
            self.fam[r].append((t, t, f))

    def setup_ok(self, res: str, s: int, e: int, fam: str | None, c: int) -> bool:
        return fam is None or all(g == fam or b + c <= s or e + c <= a for a, b, g in self.fam[res])

    def earliest_fit_setup(self, res: str, t: int, dur: int, fam: str | None, c: int) -> int:
        if fam is None:
            return self.earliest_fit(res, t, dur)
        cands = {t, *(b for _, b in self.busy[res] if b > t), *(b + c for _, b, g in self.fam[res] if g != fam and b + c > t)}
        for s in sorted(cands):
            if self.fits(res, s, s + dur) and self.setup_ok(res, s, s + dur, fam, c):
                return s
        raise RuntimeError("unreachable: the last candidate clears every interval")

    def fits(self, res: str, s: int, e: int) -> bool:
        if e <= s:
            return True
        overlapping = [(a, b) for a, b in self.busy[res] if a < e and b > s]
        if len(overlapping) < self.cap[res]:
            return True
        # Peak load inside [s, e) is reached at s or at some interval start.
        for t in {s, *(a for a, _ in overlapping if a > s)}:
            if sum(1 for a, b in overlapping if a <= t < b) >= self.cap[res]:
                return False
        return True

    def earliest_fit(self, res: str, t: int, dur: int) -> int:
        # Feasibility can only change at interval ends, so those are the only candidates.
        for s in sorted({t, *(b for _, b in self.busy[res] if b > t)}):
            if self.fits(res, s, s + dur):
                return s
        raise RuntimeError("unreachable: the last interval end always fits")

    def commit(self, res: str, s: int, e: int) -> None:
        if e > s:
            self.busy[res].append((s, e))


def _place_chain(problem: Problem, tl: Timeline, ops: list[Op], t0: int, strict: bool):
    """Place a job's remaining ops in chain order, the first no earlier than t0.
    Returns (start, end, assigned, retry_t0). retry_t0 is set when a max-lag or an
    incubation stretch fails and strict=True; with strict=False the chain is placed anyway."""
    start: dict[str, int] = {}
    end: dict[str, int] = {}
    assigned: dict[str, str] = {}
    cursor = t0
    prev: Op | None = None
    for op in ops:
        lag = problem.lag_between(prev.id, op.id) if prev else None
        min_lag = lag.min_lag if lag else 0
        max_lag = lag.max_lag if lag else None
        lo = max(cursor + min_lag, problem.earliest(op.id))
        # Lags from earlier ops of the chain other than the previous one (e.g. dispense start ->
        # incubator entry).
        far = [l for l in problem.lags if l.after == op.id and l is not lag and l.before in start]
        for l in far:
            lo = max(lo, l.anchor(start, end) + l.min_lag)
        fam = problem.family.get(op.id) if op.kind in problem.setup_min else None
        c = problem.setup_min.get(op.kind, 0)
        if op.id in problem.pinned:
            res = problem.pinned[op.id]
            s = problem.start_lb[op.id]
        else:
            best = None
            for r in problem.alternatives(op.id):
                d = problem.dur_range(op.id, r)[0]
                s_r = tl.earliest_fit_setup(r, lo, d, fam, c)
                if best is None or s_r + d < best[0] + best[2]:
                    best = (s_r, r, d)
            s, res, _ = best
        for l in far:
            if strict and l.max_lag is not None and s - l.anchor(start, end) > l.max_lag:
                return None, None, None, t0 + SHIFT_STEP

        if prev is not None and not prev.fixed:
            # A plate waiting on this op stays in the incubator, up to its max duration.
            ps = start[prev.id]
            hi = ps + problem.dur_range(prev.id, assigned[prev.id])[1]
            pe = max(min(s - min_lag, hi), end[prev.id])
            if not tl.fits(assigned[prev.id], ps, pe) and strict:
                return None, None, None, t0 + SHIFT_STEP
            end[prev.id] = pe
        if strict and prev is not None and max_lag is not None and s - end[prev.id] > max_lag:
            return None, None, None, t0 + (s - end[prev.id] - max_lag)
        if strict and op.id in problem.start_ub and s > problem.start_ub[op.id]:
            return None, None, None, None  # cannot be fixed by shifting later

        dmin = problem.dur_range(op.id, res)[0]
        start[op.id], assigned[op.id] = s, res
        end[op.id] = max(s + dmin, problem.end_lb.get(op.id, 0))
        cursor = end[op.id]
        prev = op
    return start, end, assigned, None


def _fit_spans(problem: Problem, tl: Timeline, spans, start, end, assigned) -> bool:
    """Give each span a resource free over its whole interval (rewriting the link op's
    assignment); False if some span has none."""
    for sp in spans:
        s, e = problem.span_interval(sp, start, end)
        alts = (sp.res,) if sp.res else problem.alternatives(sp.link)
        pref = sorted(alts, key=lambda r: r != assigned.get(sp.link))
        link = problem.ops.get(sp.link)
        fam = problem.family.get(sp.link) if link and link.kind in problem.setup_min else None
        c = problem.setup_min.get(link.kind, 0) if link else 0
        setup = lambda r: fam is None or tl.setup_ok(r, start[sp.link], end[sp.link], fam, c)
        ok = next((r for r in pref if tl.fits(r, s, e) and setup(r)), None)
        if ok is None:
            return False
        if sp.link in problem.ops:
            assigned[sp.link] = ok
    return True


def serial_schedule(problem: Problem, job_order: list[str], name: str, max_tries: int = 5000) -> Schedule:
    t_wall = time.perf_counter()
    tl = Timeline(problem)
    sched = Schedule(solver=name, status="feasible")
    # Jobs with running (pinned) ops go first: their timing is not negotiable. So do jobs whose plate already holds
    # a deck zone (its span started): every other plate needs that zone, which frees only when this one moves on.
    held = {problem.ops[sp.link].job for sp in problem.spans if sp.link in problem.span_start and sp.link in problem.ops}
    pinned_jobs = [j for j in job_order if j in held or any(o.id in problem.pinned for o in problem.job_ops(j))]
    order = pinned_jobs + [j for j in job_order if j not in pinned_jobs]
    for job in order:
        ops = problem.job_ops(job)
        t0 = problem.earliest(ops[0].id)
        shiftable = ops[0].id not in problem.pinned
        placed = None
        spans = [sp for sp in problem.spans if problem.ops[sp.last].job == job]
        for _ in range(max_tries if shiftable else 1):
            start, end, assigned, retry = _place_chain(problem, tl, ops, t0, strict=True)
            if start is not None and _fit_spans(problem, tl, spans, start, end, assigned):
                placed = (start, end, assigned)
                break
            if start is not None:  # deck zone taken: try a later start
                retry = t0 + SHIFT_STEP
            if retry is None:
                break
            t0 = retry
        if placed is None:
            # Best effort: keep the earliest placement and let the validator flag the lag.
            start, end, assigned, _ = _place_chain(problem, tl, ops, problem.earliest(ops[0].id), strict=False)
            placed = (start, end, assigned)
            sched.status = "lag_violations"
        start, end, assigned = placed
        for op in ops:
            if op.id not in problem.covered():
                tl.commit(assigned[op.id], start[op.id], end[op.id])
        for sp in spans:
            tl.commit(problem.span_res(sp, assigned), *problem.span_interval(sp, start, end))
        for op in ops:
            if op.id in problem.family and op.kind in problem.setup_min:
                tl.fam[assigned[op.id]].append((start[op.id], end[op.id], problem.family[op.id]))
        sched.start.update(start)
        sched.end.update(end)
        sched.assigned.update(assigned)
    sched.solve_time_s = time.perf_counter() - t_wall
    return sched


def predecessors_first(problem: Problem, order: list[str]) -> list[str]:
    """Jobs another job waits on (a cross-job lag, e.g. a tip box or dilution batch a plate needs) go first, in
    the given order: placed after the plates that need them, they would start late and the plates with them."""
    feeds = {problem.ops[l.before].job for l in problem.lags
             if l.before in problem.ops and l.after in problem.ops and problem.ops[l.before].job != problem.ops[l.after].job}
    return [j for j in order if j in feeds] + [j for j in order if j not in feeds]


class FIFOScheduler:
    """Jobs in release order; ties by arrival (insertion) order."""

    name = "fifo"

    def schedule(self, problem: Problem, prior: Schedule | None = None, now: int = 0) -> Schedule:
        jobs = problem.jobs()
        order = sorted(jobs, key=lambda j: (problem.release.get(j, 0), jobs.index(j)))
        return serial_schedule(problem, predecessors_first(problem, order), self.name)


class PriorityScheduler:
    """Priority-list scheduling. Rules (ties broken by release, then arrival):
    - deadline: jobs whose next op has the tightest start upper bound first
    - lwr: longest work remaining (sum of minimum durations) first
    - mtd: most time-lag-constrained first (tightest max-lag sum), then lwr
    """

    def __init__(self, rule: str = "lwr"):
        self.rule = rule
        self.name = f"priority-{rule}"

    def _key(self, problem: Problem, job: str):
        ops = problem.job_ops(job)
        work = sum(min(problem.dur_range(o.id, r)[0] for r in problem.alternatives(o.id)) for o in ops)
        deadline = min((problem.start_ub[o.id] for o in ops if o.id in problem.start_ub), default=10**9)
        tight = sum(l.max_lag for l in problem.lags if l.max_lag is not None and problem.ops[l.after].job == job)
        if self.rule == "deadline":
            return (deadline,)
        if self.rule == "mtd":
            return (deadline, tight, -work)
        return (-work,)

    def schedule(self, problem: Problem, prior: Schedule | None = None, now: int = 0) -> Schedule:
        jobs = problem.jobs()
        order = sorted(jobs, key=lambda j: (*self._key(problem, j), problem.release.get(j, 0), jobs.index(j)))
        return serial_schedule(problem, predecessors_first(problem, order), self.name)
