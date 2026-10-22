import yaml
from pathlib import Path

from labloop.design.base import build_space
from labloop.execution.durations import DurationModel
from labloop.execution.plr_bridge import PLRBridge
from labloop.layout.layouts import build_layout
from labloop.loop import run_campaign
from labloop.types import PlateSpec


def test_protocol_matches_duration_model_and_volumes():
    space = build_space({"cell_lines": ["L1"], "n_drugs": 2, "dose": {"max": 10, "steps": 7, "fold": 3},
                         "timepoints_h": [48]})
    for name in ("sequential", "optimized"):
        pm = build_layout(name, PlateSpec(), {}, 0).place(space.all_conditions[:50])[0]
        rep = PLRBridge(2, space.ladders).run(pm)
        assert rep.ok
        assert rep.passes == DurationModel().cycles(pm)


def test_level0_runs_protocols_through_plr():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level0.yaml").read_text())
    run = run_campaign(dict(cfg, rounds=1), 0)
    protos = [e for e in run["events"] if e["kind"] == "protocol"]
    assert len(protos) == 1 and protos[0]["passes"] == protos[0]["model_cycles"] and not protos[0]["wells_bad"]
