"""Tiered recovery (M9): dropped plates, rules first, Claude only on escalation, and the MuJoCo
re-placement of a misplaced plate."""

import math
from pathlib import Path

import pytest
import yaml

from labloop.failures.events import ActionType as A
from labloop.failures.events import FailureEvent, FailureType as F, LabState
from labloop.failures.policies.rules import RulePolicy
from labloop.failures.policies.tiered import escalation
from labloop.loop import run_campaign

LEVEL1 = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
STATE = LabState(0, {}, None, {}, [])


def test_dropped_plate_is_voided_rerun_and_cleaned_up():
    d = RulePolicy().decide(FailureEvent(F.PLATE_DROPPED, 0, plate="P"), STATE)
    assert [a.type for a in d.actions] == [A.INVALIDATE_PLATE, A.RETEST, A.HUMAN]
    assert escalation(d, FailureEvent(F.PLATE_DROPPED, 0, plate="P"), STATE) is None  # the spill needs a person anyway


def test_drops_in_a_run():
    run = run_campaign(dict(LEVEL1, rounds=2, failures={"p_drop": 0.15}), 3)
    drops = [e for e in run["events"] if e["kind"] == "plate_dropped"]
    assert drops, "expected at least one drop at p=0.15 per transfer"
    for e in drops:
        ran = {r["op"] for r in run["schedule"] if r["job"] == e["plate"]}
        if e["after_read"]:  # data already taken: the plate keeps its read
            assert f"{e['plate']}.read" in ran and run["plates"][e["plate"]]["status"] != "invalid"
        else:
            assert run["plates"][e["plate"]]["status"] == "invalid"
            assert f"{e['plate']}.read" not in ran  # the plate's remaining steps never ran
    assert run["campaigns"]["C0"]["progress"]  # the campaign still finished


def test_escalation_rules():
    rules = RulePolicy()
    first = FailureEvent(F.PLATE_MISPLACED, 0, plate="P", attempt=1)
    last = FailureEvent(F.PLATE_MISPLACED, 0, plate="P", attempt=2)
    assert escalation(rules.decide(first, STATE), first, STATE) is None
    assert "exhausted" in escalation(rules.decide(last, STATE), last, STATE)
    busy = LabState(0, {}, None, {}, [{"event": {"plate": "P", "op": "P.dispense"}}, {"event": {"plate": "P", "op": "P.read"}}])
    assert "repeated" in escalation(rules.decide(first, busy), first, busy)


def test_tiered_calls_the_llm_only_on_escalation():
    cfg = dict(LEVEL1, rounds=2, failures={"p_misplace": 0.3, "p_tip_pickup": 0.2, "p_measurement": 0.2},
               policy="tiered", policy_cfg={"provider": "scripted"})
    rec = run_campaign(cfg, 0)["recovery"]
    assert rec.get("tiered:rules", 0) > 0 and rec.get("tiered:llm", 0) > 0
    assert rec["tiered:rules"] > rec["tiered:llm"]


def test_replace_plan_recenters_a_misplaced_plate():
    mujoco = pytest.importorskip("mujoco")
    from labloop.sim3d.cell import load
    from labloop.sim3d.motion import playback, replace_plan, set_joints

    spec, m, d = load(plates={"plate0": "lh"})
    c = spec.stations["lh"].center
    plan = replace_plan(spec, "plate0", "lh", c)
    set_joints(m, d, plan[0].target)
    for _ in range(300):
        mujoco.mj_step(m, d)
    adr = m.jnt_qposadr[m.body("plate0").jntadr[0]]
    d.qpos[adr] += 0.003  # 3 mm off along x
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    est = (float(d.qpos[adr]), float(d.qpos[adr + 1]))
    pb = playback(m, d, replace_plan(spec, "plate0", "lh", est), record=False)
    p = d.xpos[m.body("plate0").id]
    assert pb.collisions == []
    assert 1000 * math.hypot(p[0] - c[0], p[1] - c[1]) < 1.5
