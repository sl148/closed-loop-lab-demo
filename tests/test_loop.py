import json
from pathlib import Path

import yaml

from labloop.loop import run_campaign

CFG = {n: yaml.safe_load((Path(__file__).parents[1] / "configs" / f"{n}.yaml").read_text()) for n in ("level0", "level1")}


def test_level0_golden_run():
    run = run_campaign(CFG["level0"], seed=0)
    c = run["campaigns"]["C0"]
    assert c["plates"] == CFG["level0"]["rounds"]
    # Noise-free grid covers the 8x8 matrix in round 0 and finds the planted optimum.
    first = c["progress"][0]
    assert first["best"] == c["optimum"]["condition"]
    assert abs(first["best_est"] - c["optimum"]["excess"]) < 1e-9
    # Sync rounds: each round is >= 48 h incubation plus handling.
    ts = [p["t"] for p in c["progress"]]
    assert all(48 * 60 <= b - a <= 50 * 60 for a, b in zip([0] + ts, ts))


def _strip_timing(x):
    if isinstance(x, dict):
        return {k: _strip_timing(v) for k, v in x.items() if not k.endswith("time_s")}
    if isinstance(x, list):
        return [_strip_timing(v) for v in x]
    return x


def test_deterministic_given_seed():
    cfg = dict(CFG["level1"], noise={"rel_sigma": 0.05, "edge_bias": 0.1})
    a, b = run_campaign(cfg, 3), run_campaign(cfg, 3)
    assert json.dumps(_strip_timing(a), sort_keys=True) == json.dumps(_strip_timing(b), sort_keys=True)


def test_executed_schedule_respects_capacity_and_lags():
    from labloop.schedule.model import Lag, Problem, Resource, Schedule
    from labloop.schedule.validator import violations

    run = run_campaign(CFG["level1"], 0)
    for r in run["reschedules"]:
        assert r["lag_slips"] == 0 and "->" not in r["solver"]
    assert not [e for e in run["events"] if e["kind"] == "lag_slip"]
    # Rebuild an interval check from the executed schedule.
    res = {k: Resource(k, v) for k, v in run["resources"].items()}
    prob = Problem(resources=res)
    from labloop.schedule.model import Op

    sched = Schedule()
    for row in run["schedule"]:
        prob.add_op(Op(row["op"], row["job"], row["kind"], (row["resource"],), row["end"] - row["start"]))
        sched.start[row["op"]], sched.end[row["op"]], sched.assigned[row["op"]] = row["start"], row["end"], row["resource"]
    assert violations(prob, sched) == []
