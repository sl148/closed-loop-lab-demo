"""Turns plate maps into scheduling jobs.

Per plate: hotel>lh (arm) -> dispense (LH) -> lh>incubator (arm) -> incubate (incubator,
flexible) -> incubator>reader (arm) -> read (reader) -> reader>hotel (arm).
Consecutive ops are linked by a max transfer lag; incubation stretches until the arm picks
the plate up, bounded by the read window. Identical devices are alternatives (LH1, LH2, ...).
"""

from __future__ import annotations

from labloop.execution.durations import DurationModel
from labloop.schedule.model import Lag, Op, Problem, Resource, Span
from labloop.types import PlateMap

LH, INCUBATOR, READER, ARM = "lh", "incubator", "reader", "arm"
ROUTES = ("hotel>lh", "lh>incubator", "incubator>reader", "reader>hotel")


def build_resources(cfg: dict) -> dict[str, Resource]:
    res: dict[str, Resource] = {}
    for kind, key, default in ((LH, "liquid_handlers", 1), (READER, "readers", 1), (ARM, "arms", 1)):
        for i in range(cfg.get(key, default)):
            res[f"{kind}{i + 1}"] = Resource(f"{kind}{i + 1}")
    for i in range(cfg.get("incubators", 1)):
        name = f"{INCUBATOR}{i + 1}"
        res[name] = Resource(name, cfg.get("incubator_slots", 10))
    return res


def devices(resources: dict[str, Resource], kind: str) -> tuple[str, ...]:
    return tuple(r for r in resources if r.rstrip("0123456789") == kind)


def plate_job(prob: Problem, pm: PlateMap, durations: DurationModel, release: int) -> list[Op]:
    j, R = pm.plate_id, prob.resources
    inc_lo, inc_hi = durations.incubation_window(pm)
    arm = devices(R, ARM)
    ops = [
        Op(f"{j}.mv_hotel_lh", j, "transfer", arm, durations.arm("hotel>lh")),
        Op(f"{j}.dispense", j, "dispense", devices(R, LH), durations.dispense(pm)),
        Op(f"{j}.mv_lh_incubator", j, "transfer", arm, durations.arm("lh>incubator")),
        Op(f"{j}.incubate", j, "incubate", devices(R, INCUBATOR), inc_lo, inc_hi),
        Op(f"{j}.mv_incubator_reader", j, "transfer", arm, durations.arm("incubator>reader")),
        Op(f"{j}.read", j, "read", devices(R, READER), durations.read(pm)),
        Op(f"{j}.mv_reader_hotel", j, "transfer", arm, durations.arm("reader>hotel")),
    ]
    for op in ops:
        prob.add_op(op)
    for a, b in zip(ops, ops[1:]):
        prob.lags.append(Lag(a.id, b.id, 0, durations.transfer_max_lag_min))
    # The LH deck site and gantry zone are held from loading the plate to unloading it.
    prob.spans.append(Span(ops[0].id, ops[2].id, ops[1].id))
    prob.release[j] = release
    if durations.reagent_expiry_min is not None:
        prob.start_ub[f"{j}.dispense"] = release + durations.reagent_expiry_min
    return ops


def plate_problem(plates: list[PlateMap], resources: dict[str, Resource],
                  durations: DurationModel, release: int) -> Problem:
    prob = Problem(resources=resources)
    for pm in plates:
        plate_job(prob, pm, durations, release)
    return prob
