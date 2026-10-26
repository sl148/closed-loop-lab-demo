"""Hidden physical limits (execution/truth.py): math, and their effect on a run."""

import json
from pathlib import Path

import yaml

from labloop.execution.truth import Truth, z_factor
from labloop.loop import run_campaign

LEVEL1 = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())


def test_plate_effects():
    t = Truth(out_of_incubator_min=30, out_of_incubator_noise_per_min=0.01, drug_rt_stable_min=100,
              drug_halflife_min=50, actual_timepoint=True)
    fx = t.plate_effects(t_release=0, t_dispense=200, t_incubate=240, incubated_min=48 * 60 + 90)
    assert fx["out_of_incubator_over_min"] == 10 and abs(fx["extra_noise"] - 0.1) < 1e-12
    assert fx["drug_rt_over_min"] == 100 and abs(fx["dose_scale"] - 0.25) < 1e-12
    assert fx["timepoint_h"] == 49.5
    assert Truth(out_of_incubator_min=30).plate_effects(0, 0, 30, 60) == {}  # at the limit: fine


def test_read_equilibration():
    t = Truth(read_equilibration_min=30, read_equilibration_noise=0.3)
    fx = t.plate_effects(0, 0, 10, 60, t_out=100, t_read=110)
    assert fx["read_equilibration_short_min"] == 20 and abs(fx["extra_noise"] - 0.2) < 1e-12
    assert t.plate_effects(0, 0, 10, 60, t_out=100, t_read=130) == {}


def test_z_factor():
    assert z_factor([0.0, 0.0], [1.0, 1.0]) == 1.0
    assert z_factor([0.0], [1.0, 1.0]) is None
    assert z_factor([0.0, 0.1], [0.5, 1.0]) < 0.5


def test_truth_off_leaves_the_run_unchanged():
    a = run_campaign(dict(LEVEL1, rounds=2), 0)
    b = run_campaign(dict(LEVEL1, rounds=2, truth={}), 0)
    strip = lambda r: json.dumps({k: v for k, v in r.items() if k not in ("reschedules",)}, sort_keys=True)
    assert strip(a) == strip(b) and b["truth"] == {}


def test_tight_limits_bias_reads_and_are_logged():
    # Limits no plan can meet: every plate breaks them, and a dirty cell-line switch contaminates wells.
    tight = {"out_of_incubator_min": 0, "drug_rt_stable_min": 0, "drug_halflife_min": 30,
             "actual_timepoint": True, "changeover_wells": 8}
    cfg = dict(LEVEL1, rounds=2, noise={"rel_sigma": 0.02})
    base, hit = run_campaign(cfg, 1), run_campaign(dict(cfg, truth=tight), 1)
    n = len(hit["plates"])
    assert hit["truth"]["out_of_incubator"] == n and hit["truth"]["drug_rt"] == n
    assert hit["truth"].get("changeover", 0) >= 1  # two cell lines share one liquid handler
    # Potency loss weakens the drugs: mean viability over drug wells goes up.
    def mean_v(run):
        vs = [w["viability"] for p in run["plates"].values() for name, w in p["reads"].items()
              if w["valid"] and isinstance(p["wells"].get(name), dict)]
        return sum(vs) / len(vs)
    assert mean_v(hit) > mean_v(base) + 0.05


def test_qc_rejects_stressed_plates():
    cfg = dict(LEVEL1, rounds=1, noise={"rel_sigma": 0.05})
    run = run_campaign(dict(cfg, truth={"out_of_incubator_min": 0, "out_of_incubator_noise_per_min": 0.05,
                                        "qc_z_min": 0.5}), 2)
    assert run["truth"].get("qc_fail", 0) >= 1
    assert run["invalid_plates"] == run["truth"]["qc_fail"]
