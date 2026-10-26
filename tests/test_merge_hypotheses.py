"""Several independent brainstorms: hypotheses about the same limit merge, distinct ones survive."""

from labloop.agent.constraint_agent import merge_hypotheses


def h(hid, kind, steps, plan):
    return {"id": hid, "concern": hid, "kind": kind, "steps": steps, "why": "", "plan": plan}


def test_merge_keeps_first_and_records_the_rest():
    hyps = {"plate_equilibration": h("plate_equilibration", "min_between", ["to_reader", "read"], 1),
            "drug_age": h("drug_age", "deadline_from_release", ["dispense"], 1),
            "reader_warmup": h("reader_warmup", "min_between", ["read", "to_reader"], 2),  # same limit, other order
            "lh_changeover": h("lh_changeover", "changeover", ["dispense"], 3),
            "odd_a": h("odd_a", "unsure", ["read"], 2), "odd_b": h("odd_b", "unsure", ["read"], 3)}
    m = merge_hypotheses(hyps)
    assert list(m) == ["plate_equilibration", "drug_age", "lh_changeover", "odd_a", "odd_b"]
    assert m["plate_equilibration"]["also"] == [{"id": "reader_warmup", "plan": 2}]
    assert "also" not in hyps["plate_equilibration"]  # inputs untouched
