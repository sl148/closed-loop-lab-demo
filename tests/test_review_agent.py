"""Review episode tools and their guardrails, without API calls (agent/review_agent.py)."""

from labloop.agent.review_agent import ReviewBox
from labloop.kb.archive import Archive
from test_kb import make_kb

CFG = {"campaigns": [{"space": {"timepoints_h": [24, 48]}}]}
QUOTE = "Equilibrate the plate to room temperature for approximately 30 minutes before adding reagent"
REST = {"id": "rest", "kind": "min_between", "a": "incubate", "a_anchor": "end", "b": "read", "minutes": 60,
        "hard": False, "value_assumed": True}
FLAG = {"hypothesis": "lh_changeover", "hypothesis_kind": "changeover", "factor": "lh_switch_uncleaned",
        "score": "over_frac", "effect": 0.006, "p": 0.003, "summary": "plates after an uncleaned switch read above controls"}


def box(tmp_path, kind, constraints=(), flag=None):
    make_kb(tmp_path)
    b = ReviewBox(Archive(tmp_path), CFG, request={"kind": kind, "constraints": list(constraints),
                                                    "evidence": {"flag": flag} if flag else {}})
    b.slot = "review_1"
    return b


def test_data_review_adds_a_soft_limit_backed_by_data(tmp_path):
    b = box(tmp_path, "data", [REST], FLAG)
    assert b.propose_data({"kind": "min_between", "a": "to_reader", "b": "read", "minutes": 30}, "x").startswith("rejected")
    assert b.propose_data({"kind": "changeover", "minutes": 15}, "carry-over", "guide", 1, "made up").startswith("rejected")
    assert b.propose_data({"kind": "changeover", "minutes": 15}, "cells carried over on the tips",
                          "guide", 1, "Work with one cell line at a time and clean the hood between lines.").startswith("accepted")
    c = b.answer["spec"]
    assert b.answer["action"] == "add" and c["id"] == "lh_changeover" and not c["hard"] and c["value_assumed"]
    assert c["evidence"] == "data" and c["data"]["factor"] == "lh_switch_uncleaned" and c["source"]["verified"]


def test_sched_review_replaces_an_assumed_value_only_with_a_quote(tmp_path):
    b = box(tmp_path, "sched", [REST])
    assert b.replace_value("rest", 30, "guide", 2, "room temperature for about 30 min").startswith("rejected")
    assert b.replace_value("rest", 45, "guide", 2, QUOTE).startswith("rejected")  # 45 is not in the quote
    assert b.replace_value("rest", 30, "guide", 2, QUOTE).startswith("accepted")
    assert b.answer["spec"]["minutes"] == 30 and not b.answer["spec"]["value_assumed"]


def test_hard_limits_need_a_person(tmp_path):
    hard = {**REST, "hard": True, "value_assumed": False}
    b = box(tmp_path, "sched", [hard])
    assert b.replace_value("rest", 30, "guide", 2, QUOTE).startswith("rejected")
    assert b.loosen("rest", 30, "x").startswith("rejected")
    assert b.escalate("check the source") == "sent to a person" and b.answer["action"] == "escalate"


def test_loosen_must_loosen(tmp_path):
    b = box(tmp_path, "sched", [REST])
    assert b.loosen("rest", 90, "x").startswith("rejected")  # a longer minimum wait is stricter
    assert b.loosen("rest", 40, "x").startswith("accepted")
