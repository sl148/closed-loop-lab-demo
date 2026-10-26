"""In-loop reviews: hooks, pause-and-resume replay, and the lab's own guardrails (review/reviewer.py)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from labloop.constraints.compile import load_constraints
from labloop.loop import run_campaign
from labloop.review.evidence import observable_view, strip_wall
from labloop.review.providers import NondeterminismError
from labloop.review.reviewer import apply_answer
from test_kb import make_kb

ROOT = Path(__file__).parents[1]
LEVEL2 = yaml.safe_load((ROOT / "configs" / "level2.yaml").read_text())
REST = {"id": "rest", "kind": "min_between", "a": "incubate", "a_anchor": "end", "b": "read", "minutes": 60,
        "hard": False, "value_assumed": True}
QUOTE = "Equilibrate the plate to room temperature for approximately 30 minutes before adding reagent"
CFG = dict(LEVEL2, rounds=3, scheduler="repair", scheduler_cfg={"deterministic": True},
           truth=str(ROOT / "configs" / "truth.yaml"), constraints=[REST])


def review_cfg(tmp_path, **kw):
    (tmp_path / "kb").mkdir()
    make_kb(tmp_path / "kb")
    return {"data": False, "sched": True, "provider": "replay", "log": str(tmp_path / "log.jsonl"),
            "kb": str(tmp_path / "kb"), "every_reads": 3, "sched_min_n": 3, **kw}


def test_review_that_never_fires_changes_nothing():
    base = run_campaign(dict(CFG, rounds=2), 0)
    quiet = run_campaign(dict(CFG, rounds=2, review={"data": False, "sched": False, "provider": "scripted"}), 0)
    assert strip_wall({**quiet, "reviews": None}) == strip_wall({**base, "reviews": None})


def test_pending_review_then_replay_applies_the_logged_answer(tmp_path):
    cfg = dict(CFG, review=review_cfg(tmp_path))
    first = run_campaign(cfg, 0)
    req = first["pending"]  # the costly 60 min wait stopped the run for an answer
    assert req["kind"] == "sched" and req["evidence"]["constraint_id"] == "rest"
    spec = {**REST, "minutes": 30, "value_assumed": False,
            "source": {"doc": "guide", "page": 2, "quote": QUOTE, "verified": True}}
    entry = {"key": req["key"], "prefix": req["prefix"], "kind": "sched", "index": 1, "t": req["t"],
             "answer": {"action": "replace", "constraint_id": "rest", "spec": spec}}
    Path(cfg["review"]["log"]).write_text(json.dumps(entry) + "\n")
    run = run_campaign(cfg, 0)
    assert "pending" not in run and run["reviews"][0]["applied"]
    upd = [e for e in run["events"] if e["kind"] == "constraints_update"]
    assert upd and upd[0]["changed"] == [{"id": "rest", "from": 60, "to": 30}]
    again = run_campaign(cfg, 0)  # replays bit-identically from the log
    assert strip_wall(again) == strip_wall(run)
    entry["prefix"] = "0" * 16  # a log from a different history
    Path(cfg["review"]["log"]).write_text(json.dumps(entry) + "\n")
    with pytest.raises(NondeterminismError):
        run_campaign(cfg, 0)


def test_the_lab_checks_every_decision(tmp_path):
    make_kb(tmp_path)
    from labloop.kb.archive import Archive

    lab = SimpleNamespace(cons=load_constraints([REST, {**REST, "id": "hard", "a": "dispense", "b": "to_incubator",
                                                        "minutes": 5, "hard": True, "value_assumed": False}]))
    arch = Archive(tmp_path)
    clean = {"id": "lh_clean", "kind": "changeover", "minutes": 15}
    assert apply_answer({"action": "add", "spec": {**clean, "hard": True, "evidence": "data"}}, lab, arch)[0] is None
    assert apply_answer({"action": "add", "spec": {**clean, "hard": False}}, lab, arch)[0] is None  # not marked as data
    assert apply_answer({"action": "add", "spec": {**clean, "hard": False, "evidence": "data"}}, lab, arch)[0]
    fake = {**REST, "minutes": 30, "source": {"doc": "guide", "page": 2, "quote": "about 30 minutes, trust me"}}
    assert apply_answer({"action": "replace", "constraint_id": "rest", "spec": fake}, lab, arch)[0] is None
    assert apply_answer({"action": "loosen", "constraint_id": "hard", "spec": {**REST, "id": "hard", "a": "dispense",
                         "b": "to_incubator", "minutes": 10, "hard": True}}, lab, arch)[0] is None
    assert apply_answer({"action": "escalate", "reason": "x"}, lab, arch)[0] is None


def test_reviews_never_see_hidden_limits():
    seen = []

    def on_lab(env, lab):
        def later():
            yield env.timeout(9000)
            seen.append(observable_view(lab))
        env.process(later())
    run_campaign(CFG, 0, on_lab=on_lab)
    view = seen[0]
    assert view["plates"] and not [e for e in view["events"] if e["kind"] in ("truth", "undetected_misplacement")]
    assert not any("corrupt_accepted" in e for e in view["events"])
