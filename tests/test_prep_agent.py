"""Prep phase (agent/prep.py): what the lab checks in code. No API calls."""

import json
from pathlib import Path

import pytest
import yaml

from labloop.execution.supplies import PrepPlan

ROOT = Path(__file__).parents[1]
LEVEL4 = yaml.safe_load((ROOT / "configs" / "level4.yaml").read_text())
SOL = "The solubility of sorafenib in these solvents is approximately 20 mg/ml."


@pytest.fixture
def box(monkeypatch):
    if not (ROOT / "data" / "kb" / "chunks.jsonl").exists():
        pytest.skip("document archive not present")
    monkeypatch.chdir(ROOT)
    from labloop.agent.prep import PrepBox
    from labloop.kb.archive import Archive

    return PrepBox(Archive(), LEVEL4)


PIP = {"fill": {"tip": "1000uL", "mode": "jet"}, "stock": {"tip": "50uL", "mode": "surface"},
       "intermediate": {"tip": "300uL", "mode": "surface"}, "top": {"tip": "300uL", "mode": "surface"},
       "series": {"tip": "300uL", "mode": "jet"}}


def plan(**kw):  # 10 mM stocks; vinblastine (0.48 uM top) needs an intermediate step
    return {**PrepPlan.default(LEVEL4["oracle"]["drugs"]).to_dict(), "max_age_min": 1200, "ul_per_well": 600,
            "dilution": {"VINBLASTINE": [20]}, "pipetting": PIP, **kw}


def test_calc_plan_works_the_numbers(box):
    n = json.loads(box.calc_plan(plan()))
    s = n["drugs"]["SORAFENIB"]
    assert abs(s["stock_mM"] - 10.0) < 0.05 and s["top_working_uM"] == 240.0 and s["powder_mg_on_hand"] == 10
    assert n["batches_expected"] > 1 and 0 < n["dmso_final_pct_worst_well"] < 1
    assert box.calc_plan({"stocks": {}}).startswith("error")


def test_propose_prep_checks_what_the_lab_knows(box):
    greedy = plan(stocks={**plan()["stocks"], "VINBLASTINE": {"mg": 3, "dmso_ul": 300}})
    assert "1 mg on hand" in box.propose_prep(greedy, [])  # more powder than the freezer holds
    assert "2 ml tube" in box.propose_prep(plan(stocks={**plan()["stocks"], "SORAFENIB": {"mg": 10, "dmso_ul": 5000}}), [])
    assert "SRC-DIL" in box.propose_prep(plan(ul_per_well=5000), [])
    bad = [{"claim": "x", "doc": "cayman_sorafenib", "page": 1, "quote": "not on the page at all"}]
    assert "quotes not found" in box.propose_prep(plan(), bad)
    good = [{"claim": "sorafenib DMSO solubility", "doc": "cayman_sorafenib", "page": 1, "quote": SOL}]
    assert box.propose_prep(plan(), good).startswith("accepted") and box.plan["max_age_min"] == 1200


def test_dilution_protocol_is_held_to_liquid_classes_and_pyLabRobot(box):
    n = json.loads(box.calc_plan(plan(dilution={})))  # vinblastine straight from a 10 mM stock to 0.48 uM
    assert any("VINBLASTINE stock" in x and "outside" in x for x in n["protocol"]["problems"])
    assert "outside" in box.propose_prep(plan(dilution={}), [])
    ok = json.loads(box.calc_plan(plan()))
    assert ok["protocol"]["problems"] == [] and ok["protocol"]["minutes_per_batch"] > 0
