"""Labware phase (agent/labware.py): tool guardrails and the rule-limited deck search. No API calls."""

import json
from pathlib import Path

import pytest
import yaml

from labloop.protocol.optimizer import optimize_deck

ROOT = Path(__file__).parents[1]
LEVEL4 = yaml.safe_load((ROOT / "configs" / "level4.yaml").read_text())
HAS_KB = (ROOT / "data" / "kb" / "chunks.jsonl").exists()
WHITE = "PL-384-WHT  Solid white opaque walls and white bottom. Reflects light; for luminescence assays."
TIP50 = "50 ul tip: medium jet 10 to 50 ul; medium surface 1 to 50 ul; dmso jet 10 to 50 ul; dmso surface 1 to 50 ul."
TIP300 = "300 ul tip: medium jet 20 to 300 ul; medium surface 0.5 to 300 ul; dmso jet 20 to 300 ul; dmso surface 1 to 300 ul."


@pytest.fixture
def box(monkeypatch):
    if not HAS_KB:
        pytest.skip("document archive not present")
    monkeypatch.chdir(ROOT)
    from labloop.agent.labware import LabwareBox
    from labloop.kb.archive import Archive

    return LabwareBox(Archive(), LEVEL4)


def test_describe_names_the_stock_and_volumes_not_the_answers(box):
    d = json.loads(box.describe_labware())
    assert d["dispenses_ul_per_well"] == {"cells": 16.0, "drug": 2.0}
    assert set(d["assay_plates_stocked"]) == {"clear", "white", "black"} and "50uL" in d["tips_stocked"]
    assert "liquid class" not in json.dumps(d).lower() and " to 50 ul" not in json.dumps(d)  # the ranges are in the archive


def test_propose_guardrails(box):
    assert box.propose_labware("assay_plate", "gold", "lab_catalog", 1, WHITE).startswith("rejected")
    assert box.propose_labware("assay_plate", "white", "lab_catalog", 2, WHITE).startswith("rejected")  # wrong page
    assert box.propose_labware("assay_plate", "black", "lab_catalog", 1, WHITE).startswith("rejected")  # quote is about white
    assert box.propose_labware("assay_plate", "white", "lab_catalog", 1, WHITE).startswith("accepted")
    r = box.propose_labware("tips.drug", ["300uL"], "lab_catalog", 2, TIP300)
    assert r.startswith("accepted")  # 2 ul is inside the 300 ul tip's surface range; how it is dispensed is pipetting.*
    assert box.propose_labware("tips.drug", ["50uL"], "lab_catalog", 2, TIP50).startswith("accepted")
    assert box.propose_labware("tips.cells", ["300uL"], "lab_catalog", 2, TIP300).startswith("accepted")
    from labloop.agent.labware import deck_rules

    assert deck_rules(box.rules) == {"assay_plate": "white", "cell_tips": ["300uL"], "drug_tips": ["50uL"]}
    assert box.abstain_labware("tips.cells", "n/a").endswith("default") and "tips.cells" in box.abstained


def test_deck_search_keeps_the_rules():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from optimize_deck import sample_plates

    sample = sample_plates(LEVEL4, 2)
    fast = optimize_deck(sample)[0]
    best = optimize_deck(sample, rules={"assay_plate": "white", "cell_tips": ["300uL"], "drug_tips": ["50uL"]})[0]
    assert (best.design.cell_tip, best.design.drug_tip, best.design.assay_plate) == ("300uL", "50uL", "white")
    assert best.total_s >= fast.total_s  # rules can only cost time
    with pytest.raises(ValueError):
        optimize_deck(sample, rules={"drug_tips": ["1000uL"], "cell_tips": ["50uL"]})


def test_pipetting_choices_hold_to_the_liquid_classes(box):
    r = box.propose_labware("pipetting.drug", "", "lab_catalog", 2, TIP50, tip="50uL", mode="jet")
    assert r.startswith("rejected") and "10 to 50" in r  # 2 ul by jet: below the liquid class
    assert box.propose_labware("pipetting.drug", "", "lab_catalog", 2, TIP50, tip="50uL", mode="surface", reuse=True).startswith("accepted")
    assert box.propose_labware("pipetting.cells", "", "lab_catalog", 2, TIP300, tip="300uL", mode="jet").startswith("rejected")
    assert box.propose_labware("pipetting.cells", "", "lab_catalog", 2, TIP50, tip="50uL", mode="jet").startswith("accepted")
    assert box.propose_labware("order", "drugs_first", "lab_catalog", 2, TIP50, value_assumed=True).startswith("accepted")
    from labloop.agent.labware import deck_rules

    dr = deck_rules(box.rules)
    assert dr["drug_tips"] == ["50uL"] and dr["cell_tips"] == ["50uL"]
    assert dr["pipetting"] == {"cell_mode": "jet", "cell_reuse": True, "drug_mode": "surface", "drug_reuse": True, "drugs_first": True}
