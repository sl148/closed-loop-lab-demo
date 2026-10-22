"""Loaders for standard JSSP (Taillard / Fisher-Thompson / Lawrence, JSPLIB format) and
FJSP (Brandimarte .fjs, 0-based machines) instances into the shared Problem model."""

from __future__ import annotations

from pathlib import Path

from labloop.schedule.model import Lag, Op, Problem, Resource


def _numbers(path: Path) -> list[int]:
    lines = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    return [int(float(x)) for l in lines for x in l.split()]


def load_jssp(path: Path) -> Problem:
    nums = _numbers(path)
    n_jobs, n_machines = nums[0], nums[1]
    it = iter(nums[2:])
    prob = Problem(resources={f"M{m}": Resource(f"M{m}") for m in range(n_machines)})
    for j in range(n_jobs):
        prev = None
        for k in range(n_machines):
            m, d = next(it), next(it)
            op = Op(f"J{j}.{k}", f"J{j}", "op", (f"M{m}",), d)
            prob.add_op(op)
            if prev:
                prob.lags.append(Lag(prev.id, op.id))
            prev = op
    return prob


def load_fjsp(path: Path) -> Problem:
    nums = _numbers(path)
    n_jobs, n_machines = nums[0], nums[1]
    it = iter(nums[2:])
    prob = Problem(resources={f"M{m}": Resource(f"M{m}") for m in range(n_machines)})
    for j in range(n_jobs):
        prev = None
        for k in range(next(it)):
            alts = [(next(it), next(it)) for _ in range(next(it))]
            op = Op(f"J{j}.{k}", f"J{j}", "op", tuple(f"M{m}" for m, _ in alts),
                    min(d for _, d in alts), durs=tuple(d for _, d in alts))
            prob.add_op(op)
            if prev:
                prob.lags.append(Lag(prev.id, op.id))
            prev = op
    return prob
