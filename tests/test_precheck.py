"""Constraint audit of executed runs, and the pre-launch gate."""

from pathlib import Path

import yaml

from labloop.constraints.audit import audit
from labloop.constraints.compile import load_constraints
from labloop.loop import run_campaign
from labloop.precheck import precheck

LEVEL1 = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
CFG = dict(LEVEL1, rounds=2)


def test_audit_scores_runs_against_any_constraint_set():
    specs = [{"id": "c", "kind": "changeover", "minutes": 15},
             {"id": "w", "kind": "read_window", "minutes": 120},
             {"id": "tight", "kind": "max_between", "a": "dispense", "a_anchor": "start", "b": "incubate", "minutes": 1}]
    comp = load_constraints(specs)
    blind = audit(comp.specs, run_campaign(CFG, 0))
    told = audit(comp.specs, run_campaign(dict(CFG, constraints=specs[:2]), 0))
    assert blind["c"]["violated"] >= 1 and told["c"]["violated"] == 0  # the scheduler honored what it was told
    assert blind["c"]["rate"] == 1.0  # never cleaned: idle time between plates does not count
    assert told["w"]["violated"] == 0
    assert blind["tight"]["rate"] == 1.0  # nothing reaches the incubator 1 min after dispense starts


def test_wait_after_incubation_counts_from_leaving_the_incubator():
    # Under contention a plate can sit in the incubator after its time is up; a wait "from the end of
    # incubation" must start when it leaves, as the plan counts it, or reads start too early.
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level4.yaml").read_text())
    spec = [{"id": "rest", "kind": "min_between", "a": "incubate", "a_anchor": "end", "b": "read", "minutes": 60}]
    cfg.update(constraints=spec, scheduler="repair", deck="configs/deck_level4.json")
    for c in cfg["campaigns"]:
        c["rounds"] = 2
    a = audit(load_constraints(spec).specs, run_campaign(cfg, 0))["rest"]
    assert a["n"] > 0 and a["violated"] == 0 and a["tight"] > 0  # kept everywhere, and it binds


def test_gate_flags_impossible_and_passes_loose():
    bad = precheck(dict(CFG, constraints=[{"id": "tight", "kind": "max_between", "a": "dispense", "a_anchor": "start",
                                           "b": "incubate", "minutes": 1}]), n=2)
    assert not bad["ok"] and bad["conflicts"][0]["constraint"] == "tight" and bad["conflicts"][0]["needs_min"] > 1
    good = precheck(dict(CFG, constraints=[{"id": "c", "kind": "changeover", "minutes": 15}]), n=2)
    assert good["ok"] and good["risk"]["c"]["rate"] == 0.0 and good["conflicts"] == []
