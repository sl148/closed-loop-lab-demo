"""Hidden labware limits (execution/truth.py): the assay plate wall and tips below their range."""

import json
from pathlib import Path

import numpy as np
import yaml

from labloop.execution.reader import PlateReader
from labloop.execution.truth import Truth
from labloop.oracle.noise import NoiseModel
from labloop.types import CONTROL_NEG, CONTROL_POS, Condition, PlateMap, PlateSpec
from labloop.loop import run_campaign

LEVEL1 = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
LABWARE = {k: v for k, v in yaml.safe_load((Path(__file__).parents[1] / "configs" / "truth_labware.yaml").read_text()).items()
           if k in ("plates", "under_range_cv", "under_range_bias")}
DECK = {"tip_rail": 25, "source_rail": 13, "dest_rail": 19, "racks_on_deck": 5, "cell_source": "trough", "tip_reuse": True}
# Hamilton liquid classes: 2 ul only by surface (contact) dispensing, 16 ul by jet from a 50 ul tip. Drugs go into
# the empty plate first, so the reused contact tip only ever touches dry plastic.
RIGHT = dict(DECK, cell_tip="50uL", drug_tip="50uL", assay_plate="white", drug_mode="surface", drugs_first=True)
WRONG = dict(DECK, cell_tip="300uL", drug_tip="300uL", assay_plate="clear")  # 2 ul and 16 ul by jet: out of range


def test_labware_effects():
    t = Truth(**LABWARE)
    assert t.on and t.labware_on
    assert t.labware_effects(RIGHT, 384) == {}
    fx = t.labware_effects(WRONG, 384)  # 2 ul of drug from a 300 ul tip (validated from 10 ul)
    assert fx["crosstalk"] == 0.05 and fx["tip_under_range"] == {"drug": "300uL jet", "cells": "300uL jet"}
    assert fx["dose_cv"] == 0.25 and fx["dose_bias"] == -0.25
    fx = t.labware_effects(dict(RIGHT, cell_tip="300uL", assay_plate="black"), 384)  # 16 ul by jet from a 300 ul tip (from 20 ul)
    assert fx["tip_under_range"] == {"cells": "300uL jet"} and abs(fx["extra_noise"] - 0.29) < 1e-12
    assert t.labware_effects(None, 384) == {} and Truth().labware_effects(WRONG, 384) == {}


def _drug_wells(run):
    return {(pid, name): w["viability"] for pid, p in run["plates"].items() for name, w in p["reads"].items()
            if w["valid"] and isinstance(p["wells"].get(name), dict)}


def test_right_labware_leaves_the_run_unchanged():
    cfg = dict(LEVEL1, rounds=2, deck=RIGHT)
    a, b = run_campaign(cfg, 0), run_campaign(dict(cfg, truth=LABWARE), 0)
    strip = lambda r: json.dumps({k: v for k, v in r.items() if k not in ("reschedules", "truth")}, sort_keys=True)
    assert strip(a) == strip(b)


class _Oracle:  # kills a well whenever drug 0 is present; records the doses it was given
    def __init__(self):
        self.seen = []

    def viability(self, cell_line, doses, tp):
        self.seen.append(doses[0])
        return 0.0 if doses[0] > 0 else 1.0


def _plate():
    spec = PlateSpec()
    wells = {(r, 0): CONTROL_NEG for r in range(spec.rows)} | {(r, spec.cols - 1): CONTROL_POS for r in range(spec.rows)}
    for r in range(spec.rows):
        for c in range(1, spec.cols - 1):
            wells[(r, c)] = Condition("X", (1.0 if (r + c) % 2 else 0.0,), 24)
    return PlateMap("P", spec, "X", 24, wells=wells)


def test_reader_crosstalk_and_underdosing():
    pm, rng = _plate(), np.random.default_rng(0)
    base = {w.well: w.value for w in PlateReader(_Oracle(), NoiseModel()).read(pm, 0, rng)}
    leak = {w.well: w.value for w in PlateReader(_Oracle(), NoiseModel()).read(pm, 0, rng, effects={"crosstalk": 0.05})}
    name = pm.spec.well_name
    r, c = 3, 4  # a killed well with four live neighbours
    assert base[name(r, c)] == 0.0 and abs(leak[name(r, c)] - 0.05) < 1e-12
    assert abs(leak[name(r, c + 1)] - 1.0) < 1e-12  # a live well among killed ones gains nothing
    o = _Oracle()
    PlateReader(o, NoiseModel()).read(pm, 0, rng, effects={"dose_bias": -0.25})
    assert max(o.seen) == 0.75  # 25% less drug than asked


def test_tip_below_its_range_underdoses():
    cfg = dict(LEVEL1, rounds=2, noise={"rel_sigma": 0.02}, truth=LABWARE)
    right, wrong = run_campaign(dict(cfg, deck=RIGHT), 1), run_campaign(dict(cfg, deck=dict(RIGHT, drug_mode="jet")), 1)
    w, c = _drug_wells(right), _drug_wells(wrong)
    assert sum(c.values()) / len(c) > sum(w.values()) / len(w) + 0.01  # less drug: more cells survive


def test_contact_tip_reused_into_wet_wells_carries_over():
    t = Truth(carryover_wells=20)
    assert t.carryover(RIGHT) is None  # drugs into dry wells first
    assert t.carryover(dict(RIGHT, drugs_first=False)) == "drug"  # 2 ul touched into wells that hold cells
    assert t.carryover(dict(RIGHT, drugs_first=False, drug_reuse=False)) is None  # fresh tip every column
