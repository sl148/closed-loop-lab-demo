"""Compiles constraint specs into scheduler primitives (schedule/model.py):

- max_between / min_between   -> Lag (ref = start or end of the earlier step)
- deadline_from_release       -> start_ub on the step
- read_window                 -> the incubation's max duration
- changeover                  -> op family + setup minutes for that op kind

The same compiled set is applied to every plate the lab accepts, so the scheduler, the
pre-launch check and the validator all see the same limits.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from labloop.constraints.spec import (STEP_OP, Changeover, ConstraintSet, DeadlineFromRelease, MaxBetween,
                                      MinBetween, ReadWindow)
from labloop.schedule.model import Lag, Problem
from labloop.types import PlateMap

ORDER = list(STEP_OP)
STEP_KIND = {"dispense": "dispense", "incubate": "incubate", "read": "read"}  # other steps are arm transfers


class ConstraintError(ValueError):
    pass


def conflicts_between(specs: list) -> list[str]:
    """Pairs of specs no schedule can satisfy together (a minimum wait above a maximum one)."""
    out = []
    for a in specs:
        for b in specs:
            if isinstance(a, MinBetween) and isinstance(b, MaxBetween) and (a.a, a.a_anchor, a.b) == (b.a, b.a_anchor, b.b) \
                    and a.minutes > b.minutes:
                out.append(f"{a.id} needs at least {a.minutes} min but {b.id} allows at most {b.minutes}")
    return out


def check(spec) -> None:
    """Rejects specs that cannot hold in the plate workflow (raised to the agent as a tool error)."""
    if isinstance(spec, (MaxBetween, MinBetween)):
        if ORDER.index(spec.a) >= ORDER.index(spec.b) and not (spec.a == spec.b and spec.a_anchor == "start"):
            raise ConstraintError(f"{spec.id}: {spec.a} does not come before {spec.b}")
    if getattr(spec, "minutes", 0) < 0:
        raise ConstraintError(f"{spec.id}: negative minutes")


@dataclass
class Compiled:
    specs: list = field(default_factory=list)

    @property
    def setup_min(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.specs:
            if isinstance(c, Changeover):
                k = STEP_KIND.get(c.step, "transfer")
                out[k] = max(out.get(k, 0), c.minutes)
        return out

    def apply(self, prob: Problem, pm: PlateMap) -> None:
        """Adds this plate's constraints to a problem that already holds its job (plate_job)."""
        j = pm.plate_id
        op = lambda step: f"{j}.{STEP_OP[step]}"
        for c in self.specs:
            if not c.scope.covers(pm.cell_line, pm.timepoint_h):
                continue
            if isinstance(c, MaxBetween):
                prob.lags.append(Lag(op(c.a), op(c.b), 0, c.minutes, ref=c.a_anchor, tag=c.id))
            elif isinstance(c, MinBetween):
                prob.lags.append(Lag(op(c.a), op(c.b), c.minutes, None, ref=c.a_anchor, tag=c.id))
                # The plate waits on purpose (e.g. equilibrating before the read): the lab's own
                # limit on waiting between those stations must leave room for it.
                inside = {op(s) for s in ORDER[ORDER.index(c.a):ORDER.index(c.b) + 1]}
                for i, l in enumerate(prob.lags):
                    if l.ref == "end" and l.before in inside and l.after in inside and l.max_lag is not None \
                            and l.max_lag < c.minutes:
                        prob.lags[i] = dataclasses.replace(l, max_lag=c.minutes + l.max_lag)
            elif isinstance(c, DeadlineFromRelease):
                o, ub = op(c.step), prob.release[j] + c.minutes
                prob.start_ub[o] = min(prob.start_ub.get(o, ub), ub)
            elif isinstance(c, ReadWindow):
                inc = prob.ops[op("incubate")]
                if inc.dur_min + c.minutes < inc.dur_hi:
                    prob.ops[inc.id] = dataclasses.replace(inc, dur_max=inc.dur_min + c.minutes)
            elif isinstance(c, Changeover):
                prob.family[op(c.step)] = getattr(pm, c.by)
                k = STEP_KIND.get(c.step, "transfer")
                prob.setup_min[k] = max(prob.setup_min.get(k, 0), c.minutes)


def compile_constraints(cs: ConstraintSet | list) -> Compiled:
    specs = cs.constraints if isinstance(cs, ConstraintSet) else list(cs)
    for c in specs:
        check(c)
    return Compiled(specs)


def load_constraints(src: str | dict | list | None) -> Compiled:
    """From a path (.yaml/.json), a ConstraintSet dict, or a list of spec dicts."""
    if not src:
        return Compiled()
    if isinstance(src, str):
        text = Path(src).read_text()
        src = json.loads(text) if src.endswith(".json") else yaml.safe_load(text)
    if isinstance(src, list):
        src = {"constraints": src}
    return compile_constraints(ConstraintSet.model_validate(src))
