"""Constraint specs -> scheduler primitives (constraints/), honored by every scheduler and checked
by the validator; in a run, the lab cleans instead of contaminating once it knows."""

from pathlib import Path

import pytest
import yaml

from labloop.constraints.compile import ConstraintError, compile_constraints, load_constraints
from labloop.constraints.spec import Changeover, MaxBetween
from labloop.execution.durations import DurationModel
from labloop.loop import run_campaign
from labloop.schedule.cpsat import CPSATScheduler
from labloop.schedule.model import Schedule
from labloop.schedule.repair import RepairScheduler
from labloop.schedule.serial import FIFOScheduler, PriorityScheduler
from labloop.schedule.validator import violations
from labloop.schedule.workflow import build_resources, plate_job
from labloop.schedule.model import Problem
from labloop.types import CONTROL_NEG, Condition, PlateMap, PlateSpec

ROOT = Path(__file__).parents[1]
LEVEL1 = yaml.safe_load((ROOT / "configs" / "level1.yaml").read_text())
SCHEDULERS = [FIFOScheduler(), PriorityScheduler("deadline"), CPSATScheduler(5)]


def plates(lines):
    out = []
    for i, line in enumerate(lines):
        pm = PlateMap(f"P{i}", PlateSpec(), line, 24)
        for k in range(40):
            pm.wells[divmod(k, 12)] = Condition(line, (1.0, float(k)), 24) if k < 36 else CONTROL_NEG
        out.append(pm)
    return out


def problem(lines, specs, **res):
    prob = Problem(resources=build_resources(res))
    comp = compile_constraints(specs)
    for pm in plates(lines):
        plate_job(prob, pm, DurationModel(), release=0)
        comp.apply(prob, pm)
    return prob


def gaps(prob, sched, res):
    seq = sorted((o for o in prob.setup_ops() if sched.assigned[o] == res), key=lambda o: sched.start[o])
    return [(prob.family[a] != prob.family[b], sched.start[b] - sched.end[a]) for a, b in zip(seq, seq[1:])]


def test_bad_order_is_rejected():
    with pytest.raises(ConstraintError):
        compile_constraints([MaxBetween(id="x", a="read", b="dispense", minutes=5)])


def test_gold_file_compiles():
    comp = load_constraints(str(ROOT / "configs" / "constraints_gold.yaml"))
    assert len(comp.specs) == 3 and comp.setup_min == {"dispense": 15}
    assert all(c.source and c.source.quote for c in comp.specs)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_changeover_is_honored(sched):
    prob = problem(["A", "B", "A", "B"], [Changeover(id="c", minutes=25)], liquid_handlers=1)
    out = sched.schedule(prob)
    assert violations(prob, out) == []
    g = gaps(prob, out, "lh1")
    assert any(switch for switch, _ in g) and all(gap >= 25 for switch, gap in g if switch)


def test_validator_catches_a_short_changeover():
    prob = problem(["A", "B"], [Changeover(id="c", minutes=25)], liquid_handlers=1)
    out = CPSATScheduler(5).schedule(prob)
    a, b = sorted(prob.setup_ops(), key=lambda o: out.start[o])
    bad = Schedule(dict(out.start), dict(out.end), dict(out.assigned))
    shift = out.start[b] - (out.end[a] + 5)  # pull b's whole chain forward to 5 min after a
    for o in prob.ops:
        if prob.ops[o].job == prob.ops[b].job:
            bad.start[o] -= shift
            bad.end[o] -= shift
    assert any("setup" in e for e in violations(prob, bad))


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_start_anchored_max_lag(sched):
    d = DurationModel().dispense(plates(["A"])[0])
    spec = MaxBetween(id="m", a="dispense", a_anchor="start", b="incubate", minutes=d + 4)
    prob = problem(["A", "A", "A"], [spec])
    out = sched.schedule(prob)
    assert violations(prob, out) == []
    for j in prob.jobs():
        assert out.start[f"{j}.incubate"] - out.start[f"{j}.dispense"] <= d + 4


def test_repair_keeps_setups():
    prob = problem(["A", "B", "A"], [Changeover(id="c", minutes=25)], liquid_handlers=1)
    prior = CPSATScheduler(5).schedule(prob)
    out = RepairScheduler(time_limit_s=2).schedule(prob, prior=prior, now=0)
    assert violations(prob, out) == []


@pytest.mark.parametrize("scheduler", ["fifo", "repair"])
def test_lab_cleans_once_told(scheduler):
    cfg = dict(LEVEL1, rounds=2, scheduler=scheduler, truth={"changeover_wells": 8})
    blind = run_campaign(cfg, 0)
    told = run_campaign(dict(cfg, constraints=[{"id": "c", "kind": "changeover", "minutes": 15}]), 0)
    assert blind["truth"].get("changeover", 0) >= 1
    assert told["truth"].get("changeover", 0) == 0
    assert any(e["kind"] == "clean" for e in told["events"])


def test_min_wait_widens_the_labs_station_limit():
    from labloop.constraints.compile import conflicts_between
    from labloop.constraints.spec import MinBetween

    spec = MinBetween(id="eq", a="to_reader", a_anchor="start", b="read", minutes=30)
    prob = problem(["A", "A"], [spec])
    out = CPSATScheduler(5).schedule(prob)
    assert violations(prob, out) == []
    for j in prob.jobs():
        assert out.start[f"{j}.read"] - out.start[f"{j}.mv_incubator_reader"] >= 30
    assert conflicts_between([spec, MaxBetween(id="m", a="to_reader", a_anchor="start", b="read", minutes=20)])
