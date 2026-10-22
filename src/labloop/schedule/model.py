"""Scheduling problem: flexible job shop with min/max time lags and cumulative resources.

All times are integer minutes. Each op runs on one of its alternative resources. Incubation is
an op with a duration window [dur_min, dur_max] on a cumulative resource (incubator slots);
handling ops have fixed durations, optionally per alternative (FJSP).

Online repair adds absolute bounds: ops already finished are dropped, and their effect on
successors becomes start bounds; running ops are pinned (start, resource) with an end bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Resource:
    name: str
    capacity: int = 1


@dataclass(frozen=True)
class Op:
    id: str
    job: str
    kind: str  # dispense | incubate | read | transfer | (benchmark ops)
    resources: tuple[str, ...]  # alternatives
    dur_min: int
    dur_max: int | None = None  # None -> fixed duration
    durs: tuple[int, ...] | None = None  # per-alternative fixed duration (FJSP)

    @property
    def fixed(self) -> bool:
        return self.dur_max is None or self.dur_max == self.dur_min

    @property
    def dur_hi(self) -> int:
        return self.dur_min if self.dur_max is None else self.dur_max

    def dur_on(self, res: str) -> int:
        return self.durs[self.resources.index(res)] if self.durs else self.dur_min

    def dur_range_on(self, res: str) -> tuple[int, int]:
        if self.durs:
            d = self.dur_on(res)
            return d, d
        return self.dur_min, self.dur_hi


@dataclass(frozen=True)
class Lag:
    """start(after) - end(before) in [min_lag, max_lag]; max_lag None = unbounded."""

    before: str
    after: str
    min_lag: int = 0
    max_lag: int | None = None


@dataclass(frozen=True)
class Span:
    """The resource `link` runs on is held from start(first) to end(last). Used for a liquid
    handler's deck site and gantry zone: held from loading a plate (arm) through the dispense
    to unloading it (arm), so the arm and the gantry never work in the zone at once and a
    second plate cannot land on an occupied site. `res` fixes the resource (online repair)."""

    first: str
    last: str
    link: str
    res: str | None = None


@dataclass
class Problem:
    resources: dict[str, Resource]
    ops: dict[str, Op] = field(default_factory=dict)  # insertion order = chain order per job
    lags: list[Lag] = field(default_factory=list)
    release: dict[str, int] = field(default_factory=dict)  # job -> earliest start of any op
    start_lb: dict[str, int] = field(default_factory=dict)
    start_ub: dict[str, int] = field(default_factory=dict)
    end_lb: dict[str, int] = field(default_factory=dict)
    pinned: dict[str, str] = field(default_factory=dict)  # op -> resource
    dur_override: dict[str, int] = field(default_factory=dict)  # running fixed op, re-estimated
    unavailable: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # breakdowns
    spans: list[Span] = field(default_factory=list)
    fixed_res: dict[str, str] = field(default_factory=dict)  # op -> resource (timing still free)
    span_start: dict[str, int] = field(default_factory=dict)  # link -> held since (first op done)

    def add_op(self, op: Op) -> None:
        self.ops[op.id] = op

    def job_ops(self, job: str) -> list[Op]:
        return [o for o in self.ops.values() if o.job == job]

    def jobs(self) -> list[str]:
        return list(dict.fromkeys(o.job for o in self.ops.values()))

    def lag_between(self, before: str, after: str) -> Lag | None:
        return next((l for l in self.lags if l.before == before and l.after == after), None)

    def alternatives(self, oid: str) -> tuple[str, ...]:
        if oid in self.pinned:
            return (self.pinned[oid],)
        if oid in self.fixed_res:
            return (self.fixed_res[oid],)
        return self.ops[oid].resources

    def covered(self) -> set[str]:
        """Ops whose resource time is represented by a span instead of their own interval."""
        return {sp.link for sp in self.spans}

    def span_interval(self, sp: Span, start: dict, end: dict) -> tuple[int, int]:
        s = start[sp.first] if sp.first in self.ops else self.span_start[sp.link]
        return s, end[sp.last]

    def span_res(self, sp: Span, assigned: dict) -> str:
        if sp.res:
            return sp.res
        return assigned.get(sp.link) or self.alternatives(sp.link)[0]

    def dur_range(self, oid: str, res: str) -> tuple[int, int]:
        if oid in self.dur_override:
            d = self.dur_override[oid]
            return d, d
        return self.ops[oid].dur_range_on(res)

    def earliest(self, oid: str) -> int:
        return max(self.release.get(self.ops[oid].job, 0), self.start_lb.get(oid, 0))


@dataclass
class Schedule:
    start: dict[str, int] = field(default_factory=dict)
    end: dict[str, int] = field(default_factory=dict)
    assigned: dict[str, str] = field(default_factory=dict)
    solver: str = ""
    solve_time_s: float = 0.0
    status: str = ""

    @property
    def makespan(self) -> int:
        return max(self.end.values(), default=0)

    def resource_of(self, problem: Problem, oid: str) -> str:
        return self.assigned.get(oid) or problem.alternatives(oid)[0]

    def to_rows(self, problem: Problem) -> list[dict]:
        return [
            {"op": oid, "job": op.job, "kind": op.kind, "resource": self.resource_of(problem, oid),
             "start": self.start[oid], "end": self.end[oid]}
            for oid, op in problem.ops.items() if oid in self.start
        ]


def utilization(problem: Problem, sched: Schedule, t0: int = 0) -> dict[str, float]:
    """Busy fraction per resource over [t0, makespan], normalized by capacity."""
    horizon = max(sched.makespan - t0, 1)
    busy: dict[str, int] = {r: 0 for r in problem.resources}
    for oid in problem.ops:
        busy[sched.resource_of(problem, oid)] += sched.end[oid] - sched.start[oid]
    return {r: busy[r] / (horizon * problem.resources[r].capacity) for r in busy}


def idle_time(problem: Problem, sched: Schedule, t0: int = 0) -> dict[str, int]:
    """Unary-resource idle minutes between the first and last op on that resource."""
    out = {}
    for r, res in problem.resources.items():
        iv = sorted((sched.start[o], sched.end[o]) for o in problem.ops if sched.resource_of(problem, o) == r)
        if res.capacity != 1 or not iv:
            continue
        out[r] = sum(max(0, b[0] - a[1]) for a, b in zip(iv, iv[1:]))
    return out
