import pytest

from labloop.execution.durations import DurationModel
from labloop.schedule.cpsat import CPSATScheduler
from labloop.schedule.model import Lag, Op, Problem, Resource, Schedule
from labloop.schedule.repair import RepairScheduler
from labloop.schedule.serial import FIFOScheduler, PriorityScheduler, Timeline
from labloop.schedule.validator import violations
from labloop.schedule.workflow import build_resources, plate_problem
from labloop.types import CONTROL_NEG, Condition, PlateMap, PlateSpec


def plates(n: int, timepoint_h: int = 48) -> list[PlateMap]:
    out = []
    for i in range(n):
        pm = PlateMap(f"P{i}", PlateSpec(), "L1", timepoint_h)
        for k in range(72):
            pm.wells[divmod(k, 12)] = Condition("L1", (1.0, float(k)), timepoint_h) if k < 64 else CONTROL_NEG
        out.append(pm)
    return out


def lab_problem(n=6, **res):
    return plate_problem(plates(n), build_resources(res), DurationModel(), release=0)


def test_timeline_cumulative_capacity():
    p = Problem(resources={"inc": Resource("inc", 2)})
    tl = Timeline(p)
    tl.commit("inc", 0, 10)
    tl.commit("inc", 5, 15)
    assert not tl.fits("inc", 6, 8)
    assert tl.earliest_fit("inc", 0, 3) == 0  # load 1 of 2 on [0, 3)
    assert tl.earliest_fit("inc", 4, 3) == 10  # [4, 7) hits load 2 at t=5


@pytest.mark.parametrize("sched", [FIFOScheduler(), PriorityScheduler("lwr"), CPSATScheduler(5)])
@pytest.mark.parametrize("n,slots", [(1, 10), (6, 10), (8, 3)])
def test_schedulers_validate(sched, n, slots):
    prob = lab_problem(n, incubator_slots=slots)
    assert violations(prob, sched.schedule(prob)) == []


def test_two_devices_are_alternatives():
    prob = lab_problem(6, liquid_handlers=2, readers=2)
    s = CPSATScheduler(5).schedule(prob)
    assert violations(prob, s) == []
    assert {s.assigned[o] for o in s.start if o.endswith("dispense")} == {"lh1", "lh2"}


def test_cpsat_not_worse_than_fifo():
    prob = lab_problem(8, incubator_slots=3)
    assert CPSATScheduler(5).schedule(prob).makespan <= FIFOScheduler().schedule(prob).makespan


def test_downtime_respected():
    prob = lab_problem(3)
    prob.unavailable = {"lh1": [(0, 120)]}
    for sched in (FIFOScheduler(), CPSATScheduler(5)):
        s = sched.schedule(prob)
        assert violations(prob, s) == []
        assert min(s.start[o] for o in s.start if o.endswith("dispense")) >= 120


def test_repair_keeps_prior_and_adds_new_jobs():
    prob = lab_problem(4)
    prior = CPSATScheduler(5).schedule(prob)
    bigger = plate_problem(plates(6), build_resources({}), DurationModel(), release=0)
    rep = RepairScheduler(time_limit_s=2, window_min=0)
    s = rep.schedule(bigger, prior=prior, now=0)
    assert violations(bigger, s) == []
    assert rep.last_neighborhood < len(bigger.jobs())


def test_validator_catches_violations():
    p = Problem(resources={"m": Resource("m", 1)})
    p.add_op(Op("a", "j", "x", ("m",), 10))
    p.add_op(Op("b", "j", "x", ("m",), 10))
    p.lags.append(Lag("a", "b", 0, 5))
    assert any("capacity" in e for e in violations(p, Schedule(start={"a": 0, "b": 5}, end={"a": 10, "b": 15})))
    assert any("max" in e for e in violations(p, Schedule(start={"a": 0, "b": 30}, end={"a": 10, "b": 40})))
    assert any("duration" in e for e in violations(p, Schedule(start={"a": 0, "b": 10}, end={"a": 9, "b": 20})))
