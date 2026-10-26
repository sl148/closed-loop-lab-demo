"""The deck scores the web page's JavaScript port reproduces (tests/fixtures/deck_cases.json) still
match the Python planner and optimizer."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_fixture_matches_python():
    fx = ROOT / "tests/fixtures/deck_cases.json"
    if not fx.exists():
        pytest.skip("run scripts/export_deck_live.py")
    from export_deck_live import cases
    from optimize_deck import sample_plates

    sample = sample_plates(yaml.safe_load((ROOT / "configs/level4.yaml").read_text()))
    want, got = json.loads(fx.read_text()), cases(sample)
    assert got["n_candidates"] == want["n_candidates"]
    same = lambda x, y: {k: x[k] for k in y} == y  # the page port knows the fixture's fields; new ones keep their defaults
    assert same(got["best"]["design"], want["best"]["design"]) and same(got["baseline"]["design"], want["baseline"]["design"])
    for k in ("total_s", "protocol_s", "swaps"):
        assert abs(got["best"][k] - want["best"][k]) < 1e-6 and abs(got["baseline"][k] - want["baseline"][k]) < 1e-6
    for a, b in zip(got["designs"], want["designs"]):
        assert same(a["design"], b["design"]) and abs(a["total_s"] - b["total_s"]) < 1e-6
