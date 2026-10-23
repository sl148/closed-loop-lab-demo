"""Independent constraint checker. Shares no code with any scheduler; every produced
schedule goes through here."""

from __future__ import annotations

from labloop.schedule.model import Problem, Schedule


def violations(problem: Problem, sched: Schedule, check_lags: bool = True) -> list[str]:
    errs: list[str] = []
    for oid, op in problem.ops.items():
        if oid not in sched.start or oid not in sched.end:
            errs.append(f"{oid}: not scheduled")
            continue
        res = sched.assigned.get(oid, op.resources[0])
        allowed = (problem.pinned[oid],) if oid in problem.pinned else op.resources
        if res not in allowed:
            errs.append(f"{oid}: resource {res} not in {allowed}")
            continue
        d = sched.end[oid] - sched.start[oid]
        if oid in problem.dur_override:
            lo = hi = problem.dur_override[oid]
        elif op.durs:
            lo = hi = op.durs[op.resources.index(res)]
        else:
            lo, hi = op.dur_min, (op.dur_max if op.dur_max is not None else op.dur_min)
        if not (lo <= d <= hi):
            errs.append(f"{oid}: duration {d} outside [{lo}, {hi}]")
        s = sched.start[oid]
        if s < problem.release.get(op.job, 0) or s < problem.start_lb.get(oid, s):
            errs.append(f"{oid}: starts {s} before its lower bound")
        if oid in problem.start_ub and s > problem.start_ub[oid]:
            errs.append(f"{oid}: starts {s} after its upper bound {problem.start_ub[oid]}")
        if oid in problem.end_lb and sched.end[oid] < problem.end_lb[oid]:
            errs.append(f"{oid}: ends {sched.end[oid]} before {problem.end_lb[oid]}")
        for a, b in problem.unavailable.get(res, []):
            if s < b and sched.end[oid] > a:
                errs.append(f"{oid}: runs on {res} during downtime [{a}, {b})")
    if errs:
        return errs

    if check_lags:
        for lag in problem.lags:
            gap = sched.start[lag.after] - sched.end[lag.before]
            if gap < lag.min_lag:
                errs.append(f"{lag.before}->{lag.after}: gap {gap} < min {lag.min_lag}")
            if lag.max_lag is not None and gap > lag.max_lag:
                errs.append(f"{lag.before}->{lag.after}: gap {gap} > max {lag.max_lag}")

    # A span (deck zone held from loading to unloading) replaces its link op's interval.
    covered = problem.covered()
    spans = []
    for sp in problem.spans:
        s = sched.start[sp.first] if sp.first in problem.ops else problem.span_start[sp.link]
        res = sp.res or sched.assigned.get(sp.link)
        if res is None or (sp.link in problem.ops and sched.assigned.get(sp.link) != res):
            errs.append(f"span {sp.link}: zone resource {res} differs from the op's")
            continue
        spans.append((res, s, sched.end[sp.last]))
    for rname, res in problem.resources.items():
        events = []
        for oid, op in problem.ops.items():
            if oid in covered:
                continue
            if sched.assigned.get(oid, op.resources[0]) == rname and sched.end[oid] > sched.start[oid]:
                events += [(sched.start[oid], 1), (sched.end[oid], -1)]
        for r, s, e in spans:
            if r == rname and e > s:
                events += [(s, 1), (e, -1)]
        load = 0
        # Releases sort before acquisitions at the same instant.
        for t, delta in sorted(events, key=lambda e: (e[0], e[1])):
            load += delta
            if load > res.capacity:
                errs.append(f"{rname}: load {load} > capacity {res.capacity} at t={t}")
                break
    return errs


def assert_valid(problem: Problem, sched: Schedule) -> None:
    errs = violations(problem, sched)
    if errs:
        raise ValueError(f"invalid schedule ({sched.solver}): " + "; ".join(errs[:5]))
