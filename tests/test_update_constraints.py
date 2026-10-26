"""Constraints changed mid-run (Lab.update_constraints), as an approved review does."""

from pathlib import Path

import yaml

from labloop.constraints.audit import audit_versions
from labloop.constraints.compile import load_constraints
from labloop.loop import run_campaign

ROOT = Path(__file__).parents[1]
LEVEL2 = yaml.safe_load((ROOT / "configs" / "level2.yaml").read_text())
CFG = dict(LEVEL2, rounds=3, scheduler="repair", scheduler_cfg={"deterministic": True},
           truth=str(ROOT / "configs" / "truth.yaml"))
T_UPDATE = 5000  # minutes: round 2 has switched cell line (uncleaned), round 3 not yet seeded


def run_with_update(cfg, new_specs, scope="queued"):
    labs = []

    def on_lab(env, lab):
        labs.append(lab)

        def later():
            yield env.timeout(T_UPDATE)
            lab.update_constraints(load_constraints(new_specs), scope=scope, reason="test")
        env.process(later())
    return run_campaign(cfg, 0, on_lab=on_lab), labs[0]


def test_cleaning_added_mid_run_stops_carry_over_from_then_on():
    clean = [{"id": "lh_clean", "kind": "changeover", "minutes": 15, "hard": False, "value_assumed": True}]
    run, lab = run_with_update(CFG, clean)
    upd = [e for e in run["events"] if e["kind"] == "constraints_update"]
    assert len(upd) == 1 and upd[0]["added"] == ["lh_clean"] and upd[0]["t"] == T_UPDATE
    dirty = [e for e in run["events"] if e["kind"] == "truth" and e.get("limit") == "changeover"]
    cleans = [e for e in run["events"] if e["kind"] == "clean"]
    assert dirty and all(e["t"] < T_UPDATE for e in dirty)  # carry-over before the update only
    assert cleans and all(e["t"] >= T_UPDATE for e in cleans)
    assert any(v == 1 for v in run["plate_version"].values())
    later = audit_versions(run)[1]["lh_clean"]
    assert later["n"] > 0 and later["violated"] == 0


def test_relaxing_a_wait_rebuilds_the_queued_plates():
    rest = {"id": "rest", "kind": "min_between", "a": "incubate", "a_anchor": "end", "b": "read", "minutes": 60,
            "hard": False, "value_assumed": True}
    run, lab = run_with_update(dict(CFG, constraints=[rest]), [{**rest, "minutes": 30}])
    redone = [p for p, v in run["plate_version"].items() if v == 1]
    assert redone
    for pid in redone:  # exactly one constraint lag into the read, at the new value
        tagged = [l for l in lab.lags if l.after == f"{pid}.read" and l.tag == "rest"]
        assert [l.min_lag for l in tagged] == [30]
    a = audit_versions(run)[1]["rest"]
    assert a["n"] > 0 and a["violated"] == 0 and a["tight"] > 0  # kept, and now the 30 min is what binds


def test_constraint_history_and_slip_tags_are_exported():
    run, _ = run_with_update(CFG, [{"id": "lh_clean", "kind": "changeover", "minutes": 15}], scope="new")
    assert [h["version"] for h in run["constraint_history"]] == [0, 1]
    assert run["constraint_history"][1]["specs"][0]["id"] == "lh_clean"
