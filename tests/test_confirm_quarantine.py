"""Confirmation round and quarantine (loop.Campaign, evaluate.History, Lab.quarantine)."""

from pathlib import Path

import yaml

from labloop.evaluate import History
from labloop.loop import run_campaign
from labloop.types import Condition, WellRead

ROOT = Path(__file__).parents[1]
LEVEL2 = yaml.safe_load((ROOT / "configs" / "level2.yaml").read_text())
CFG = dict(LEVEL2, rounds=3, scheduler="repair", scheduler_cfg={"deterministic": True}, landscape_eval=False)


def _read(plate, c, v):
    return WellRead(plate, "A1", c, False, v, v, 0)


def test_history_drop_and_replicated_best():
    a, b = Condition("X", (1.0, 1.0), 24), Condition("X", (2.0, 2.0), 24)
    singles = [a.single(0), a.single(1), b.single(0), b.single(1)]
    h = History()
    h.add([_read("P1", s, 0.9) for s in singles] + [_read("P1", a, 0.2)])  # a: one lucky well, excess 0.61
    h.add([_read("P2", s, 0.9) for s in singles for _ in range(3)] + [_read("P2", b, 0.4) for _ in range(3)])
    assert h.best(min_n=3)[0] == b  # only b has replicates
    assert h.drop({"P1"}) == 5 and all(r.plate_id == "P2" for r in h.reads)


def test_confirmation_round_retests_the_best_with_replicates():
    run = run_campaign(dict(CFG, confirm={"plates": 1, "candidates": 4, "replicates": 4}), 0)
    conf = [e for e in run["events"] if e["kind"] == "confirm"]
    assert len(conf) == 1 and conf[0]["candidates"] >= 1
    camp = next(iter(run["campaigns"].values()))
    assert camp["plates"] == CFG["rounds"] * CFG["plates_per_round"]  # inside the budget
    fb = camp["final_best"]["best"]
    plate = run["plates"][conf[0]["plates"][0]]
    hits = [w for w in plate["wells"].values() if isinstance(w, dict) and w["doses"] == fb["doses"]
            and plate["cell_line"] == fb["cell_line"]]
    assert len(hits) == 4  # the answer is one of the confirmed candidates, read 4 times


def test_quarantine_drops_a_plate_and_reruns_it():
    seen = {}

    def hook(env, lab):
        def later():  # quarantine the first plate read, once there is one
            while not lab.plate_reads:
                yield env.timeout(60)
            done = list(lab.plate_reads)
            seen["q"] = lab.quarantine(done[:1], reason="test")
        env.process(later())

    run = run_campaign(CFG, 0, on_lab=hook)
    q = seen["q"]["plates"]
    assert len(q) == 1 and seen["q"]["reads_dropped"] > 0 and run["plates"][q[0]]["status"] == "quarantined"
    assert any(e["kind"] == "quarantine" for e in run["events"])
    wells = lambda p: {(w["cell_line"], tuple(w["doses"])) for w in run["plates"][p]["wells"].values() if isinstance(w, dict)}
    later = set().union(*(wells(p) for p in run["plates"] if p != q[0]))
    assert len(wells(q[0]) & later) > 0.5 * len(wells(q[0]))  # its conditions came back on later plates
