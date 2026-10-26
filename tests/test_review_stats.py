"""Data-side review statistics on a synthetic run (review/stats.py)."""

import numpy as np

from labloop.review.stats import associate, check_hypotheses, hypothesis_test, plate_factors, plate_scores

LINES = ["L1", "L2", "L3", "L4"]
CHANGEOVER = {"id": "lh_changeover", "kind": "changeover", "steps": ["dispense"]}


def synthetic_run(n=32, clean=False, seed=0, contaminate=True):
    """Plates alternate cell lines on two liquid handlers; a switch without a clean puts 24 wells off by
    U(0.2, 1.8), as the lab's hidden changeover limit does."""
    rng = np.random.default_rng(seed)
    plates, schedule, events = {}, [], []
    last: dict[str, str] = {}
    for k in range(n):
        pid, line, lh = f"P{k:03d}", LINES[(k // 4) % 4], f"lh{k % 2 + 1}"  # each LH: A, A, B, B, ...
        t0 = 100 * k
        events.append({"t": t0, "kind": "submit", "plates": [pid]})
        prev, last[lh] = last.get(lh), line
        switch = prev is not None and prev != line
        if switch and clean:
            events.append({"t": t0 + 5, "kind": "clean", "plate": pid, "lh": lh})
        wells, reads = {}, {}
        names = [f"{chr(65 + r)}{c + 1}" for r in range(16) for c in range(24)]
        for w in names[:4]:
            wells[w] = "NEG"
        for w in names[4:300]:
            wells[w] = {"cell_line": line, "doses": [float(rng.integers(0, 4)) for _ in range(4)], "timepoint_h": 24}
        dirty = set(rng.choice(names[4:300], 24, replace=False)) if switch and not clean and contaminate else set()
        for w, c in wells.items():
            v = 1.0 if c == "NEG" else 0.9 - 0.15 * sum(d > 0 for d in c["doses"]) / 4
            v *= 1 + 0.05 * rng.normal()
            if w in dirty:
                v *= rng.uniform(0.2, 1.8)
            reads[w] = {"well": w, "value": v, "viability": v, "valid": True}
        plates[pid] = {"cell_line": line, "timepoint_h": 24, "rows": 16, "cols": 24, "wells": wells, "reads": reads}
        for op, a, b, res in (("dispense", 10, 30, lh), ("mv_lh_incubator", 30, 32, "arm1"), ("incubate", 32, 1472, "incubator1"),
                              ("mv_incubator_reader", 1472, 1474, "arm1"), ("read", 1474, 1490, "reader1")):
            schedule.append({"op": f"{pid}.{op}", "job": pid, "kind": op, "resource": res, "start": t0 + a, "end": t0 + b})
    return {"plates": plates, "schedule": schedule, "events": events}


def test_contaminated_plates_read_above_their_controls():
    run = synthetic_run()
    scores, factors = plate_scores(run), plate_factors(run)
    sw = [p for p in factors if factors[p]["lh_switch_uncleaned"]]
    rest = [p for p in factors if not factors[p]["lh_switch_uncleaned"]]
    assert sw and rest
    assert np.mean([scores[p]["over_frac"] for p in sw]) > 5 * max(1e-3, np.mean([scores[p]["over_frac"] for p in rest]))


def test_the_agents_changeover_hypothesis_is_confirmed_by_data():
    run = synthetic_run()
    res = check_hypotheses(plate_scores(run), plate_factors(run), [CHANGEOVER, {"id": "x", "kind": "read_window", "steps": []}])
    assert [r["hypothesis"] for r in res] == ["lh_changeover"]  # read_window leaves no mark these scores see
    assert res[0]["flag"] and res[0]["p_holm"] < 0.05 and res[0]["effect"] > 0


def test_no_flag_when_every_switch_was_cleaned():
    run = synthetic_run(clean=True)
    res = check_hypotheses(plate_scores(run), plate_factors(run), [CHANGEOVER])
    assert res and not res[0]["testable"] and not res[0]["flag"]
    assert not [t for t in associate(plate_scores(run), plate_factors(run), B=199) if t["flag"]]


def test_hypothesis_mapping():
    assert hypothesis_test(CHANGEOVER) == ("lh_switch_uncleaned", "over_frac", 1)
    assert hypothesis_test({"kind": "min_between", "steps": ["to_reader", "read"]})[0] == "eq_min"
    assert hypothesis_test({"kind": "deadline_from_release", "steps": ["dispense"]}) is None


def test_switches_without_carry_over_are_not_flagged():
    run = synthetic_run(contaminate=False)
    for method in ("history", "contrast"):
        res = check_hypotheses(plate_scores(run), plate_factors(run), [CHANGEOVER], method=method, alpha=0.05)
        assert res and res[0]["testable"] and not res[0]["flag"], method


def test_history_method_fires_after_a_few_exposed_plates():
    run = synthetic_run(n=10)
    res = check_hypotheses(plate_scores(run), plate_factors(run), [CHANGEOVER], alpha=0.01)
    assert res[0]["method"] == "history" and res[0]["flag"] and res[0]["arms"][0] < 10
