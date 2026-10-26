"""Schedule-side review triggers: when should a constraint be looked up again?

(a) costly: a minimum wait (or cleaning gap) with an assumed value holds most plates exactly at its
    limit, so that guess decides the timing; a documented value could be shorter. Values already
    taken from a quote are not re-reviewed.
(b) infeasible: the scheduler keeps failing to plan (falls back to FIFO or a serial plan). Soft-lag
    repairs ("lag_violations") are routine under overruns and do not count.
"""

from __future__ import annotations

import dataclasses

from labloop.constraints.audit import audit
from labloop.constraints.spec import Changeover, MinBetween
from labloop.schedule.serial import FIFOScheduler

FALLBACK = ("->fifo", ":serial")


def since_unchanged(lab, c) -> int:
    """Minute from which constraint `c` has had its current value."""
    t = 0
    for h in reversed(lab.constraint_history):
        same = any(s["id"] == c.id and s.get("minutes") == c.minutes for s in h["specs"])
        if not same:
            break
        t = h["t"]
    return t


def fifo_delta(lab, c) -> int:
    """Minutes the live plan's FIFO makespan would shrink without `c`'s waits (a quick cost probe)."""
    prob, _ = lab.build_problem()
    base = FIFOScheduler().schedule(prob).makespan
    prob.lags = [dataclasses.replace(l, min_lag=0) if l.tag == c.id else l for l in prob.lags]
    if isinstance(c, Changeover):
        prob.setup_min = {}
    return int(base - FIFOScheduler().schedule(prob).makespan)


def costly(lab, view: dict, done: set, min_n: int = 10, frac: float = 0.5, min_minutes: int = 30) -> list[dict]:
    out = []
    for c in lab.cons.specs:
        if not c.value_assumed or not isinstance(c, (MinBetween, Changeover)) or (c.id, c.minutes) in done:
            continue
        a = audit([c], view, since=since_unchanged(lab, c))[c.id]
        if a["n"] < min_n or a["tight"] / a["n"] <= frac or c.minutes < min_minutes:
            continue
        out.append({"trigger": "costly", "constraint_id": c.id, "kind": c.kind, "minutes": c.minutes,
                    "value_assumed": c.value_assumed, "hard": c.hard, "plates": a["n"], "at_limit": a["tight"],
                    "at_limit_frac": a["tight"] / a["n"], "broken": a["violated"],
                    "fifo_makespan_saved_min_without_it": fifo_delta(lab, c),
                    "source": c.source.model_dump() if c.source else None})
    return out


def infeasible(lab, last_t: int, min_count: int = 3) -> dict | None:
    bad = [r for r in lab.reschedules if r["t"] > last_t and (r["solver"].endswith(FALLBACK) or "wall_cap" in r.get("status", ""))]
    if len(bad) < min_count:
        return None
    return {"trigger": "infeasible", "fallbacks": len(bad),
            "recent": [{k: r.get(k) for k in ("t", "reasons", "solver", "status")} for r in bad[-5:]],
            "soft_constraints": [c.id for c in lab.cons.specs if not c.hard]}
