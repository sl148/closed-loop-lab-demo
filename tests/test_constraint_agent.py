"""The constraint agent's tools and guardrails, without API calls (agent/constraint_agent.py)."""

import json
from pathlib import Path

import yaml

from labloop.agent.constraint_agent import Toolbox
from labloop.kb.archive import Archive
from test_kb import make_kb

LEVEL1 = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
QUOTE = "Equilibrate the plate to room temperature for approximately 30 minutes before adding reagent"


def box(tmp_path):
    make_kb(tmp_path)
    b = Toolbox(Archive(tmp_path), dict(LEVEL1, rounds=1))
    b.slot = "read_equilibration"
    return b


SPEC = {"kind": "min_between", "a": "to_reader", "a_anchor": "start", "b": "read", "minutes": 30}


def test_propose_needs_a_verbatim_quote_with_the_number(tmp_path):
    b = box(tmp_path)
    assert json.loads(b.list_parameters())["slot"] == "read_equilibration"
    assert json.loads(b.search_corpus("equilibrate room temperature"))[0]["page"] == 2
    assert b.propose(SPEC, "guide", 2, "room temperature for about 30 min").startswith("rejected")  # not verbatim
    assert b.propose({**SPEC, "minutes": 45}, "guide", 2, QUOTE).startswith("rejected")  # number not in quote
    assert b.propose({**SPEC, "a": "read", "b": "to_reader"}, "guide", 2, QUOTE).startswith("rejected")  # wrong order
    assert b.propose({**SPEC, "kind": "make_it_faster"}, "guide", 2, QUOTE).startswith("rejected")  # outside the vocabulary
    assert b.propose({**SPEC, "a": "dispense"}, "guide", 2, QUOTE).startswith("rejected: always true")  # spans the incubation
    assert b.propose({**SPEC, "kind": "max_between", "a": "dispense"}, "guide", 2, QUOTE).startswith("rejected: impossible")
    assert b.propose(SPEC, "guide", 2, QUOTE).startswith("accepted")
    p = b.proposals["read_equilibration"]
    assert p["source"]["verified"] and p["minutes"] == 30 and p["id"] == "read_equilibration"
    assert b.abstain("changed my mind").startswith("slot") and not b.proposals
    assert [t["tool"] for t in b.trace] == ["list_parameters", "search_corpus"] + ["propose"] * 7 + ["abstain"]


def test_hours_in_the_quote_count(tmp_path):
    b = box(tmp_path)
    b.slot = "reagent_stability"
    quote = "Aqueous solutions should not be stored for more than one day."
    spec = {"kind": "deadline_from_release", "step": "dispense", "minutes": 1440}
    assert b.propose({**spec, "minutes": 720}, "sheet", 1, quote).startswith("rejected")  # half a day is not stated
    assert b.propose(spec, "sheet", 1, quote).startswith("accepted")  # "one day" = 1440 min


def test_assumed_value_becomes_soft(tmp_path):
    b = box(tmp_path)
    b.slot = "changeover"
    quote = "Work with one cell line at a time and clean the hood between lines."
    spec = {"kind": "changeover", "minutes": 15}
    assert b.propose(spec, "guide", 1, quote).startswith("rejected")  # 15 is not in the quote
    assert b.propose(spec, "guide", 1, quote, value_assumed=True).startswith("accepted")
    assert b.proposals["changeover"]["hard"] is False and "assumed" in b.proposals["changeover"]["rationale"]


def test_open_mode_plans_then_grounds_then_sweeps(tmp_path):
    b = box(tmp_path)
    b.slot = "plan"
    lab = json.loads(b.describe_lab())
    assert "equilibrat" not in json.dumps(lab).lower()  # the task names no limit
    assert b.hypothesize("plate_warmup", "Must the plate warm up before the read?", "min_between", ["to_reader", "read"],
                         "luminescence depends on temperature").startswith("recorded")
    assert b.hypothesize("plate_warmup", "again", "min_between", ["read"], "dup").startswith("rejected")
    assert b.hypothesize("Bad Id", "x", "min_between", ["read"], "x").startswith("rejected")
    assert b.hypothesize("odd_kind", "x", "make_it_faster", ["read"], "x").startswith("rejected")
    b.slot = "plate_warmup"
    assert "warm up" in json.loads(b.list_parameters())["question"]
    assert b.propose(SPEC, "guide", 2, QUOTE).startswith("accepted")
    assert b.proposals["plate_warmup"]["id"] == "plate_warmup"
    b.slot = "sweep"
    assert {d["doc"] for d in json.loads(b.list_documents())} == {"guide", "sheet"}
    assert b.propose(SPEC, "guide", 2, QUOTE, limit_id="read_rest").startswith("rejected")  # plate_warmup covers it
    assert b.propose(SPEC, "guide", 2, QUOTE).startswith("rejected")  # the sweep needs a new id
    spec = {"kind": "deadline_from_release", "step": "dispense", "minutes": 1440}
    assert b.propose(spec, "sheet", 1, "Aqueous solutions should not be stored for more than one day.",
                     limit_id="drug_shelf_life").startswith("accepted")
    assert b.hypotheses["drug_shelf_life"]["origin"] == "sweep" and b.hypotheses["plate_warmup"]["origin"] == "reasoning"


def test_simulate_reports_the_proposals_effect(tmp_path):
    b = box(tmp_path)
    b.propose(SPEC, "guide", 2, QUOTE)
    out = json.loads(b.simulate(1))
    assert out["ok"] and "read_equilibration" in out["risk"]
    assert out["makespan_h_p50"] >= out["makespan_h_p50_without"]
