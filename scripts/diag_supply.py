"""Why a plate keeps waiting on supplies: run one seed, and when the livelock guard fires, print the plate's
ops, the batches it saw, and what each supply check said.   PYTHONPATH=src python scripts/diag_supply.py fifo 0"""
import sys

import yaml

from labloop.execution import lab as labmod
from labloop.loop import run_campaign

sched, seed = sys.argv[1], int(sys.argv[2])
seen = {"lab": None}
orig = labmod.Lab._supplies_ok
log = []


def wrapped(self, lh, job, now):
    seen["lab"] = self
    S, pm = self.supplies, self.plates[job]
    b = S.batch.get(lh)
    log.append((now, job, lh, b.made if b else None, S.drug_ok(lh, pm, now + 2), S.cells_ok(lh, pm),
                {k: (v, round(b.left.get(k, 0), 1)) for k, v in S.need(pm)["drug"].items()} if b else None))
    return orig(self, lh, job, now)


labmod.Lab._supplies_ok = wrapped
cfg = yaml.safe_load(open("configs/level4.yaml"))
sets = {"designer": "throughput-bo", "scheduler": sched, "scheduler_cfg": {"deterministic": True},
        "deck": "configs/deck_level4_gold.json", "sim3d": {"misplace_sigma_mm": 1.5, "perception": True},
        "truth": "configs/truth_prep.yaml", "plr": False, "constraints": "configs/constraints_gold.yaml", "prep": "configs/prep_gold.json"}
try:
    run_campaign({**cfg, **sets}, seed)
    print("finished")
except RuntimeError as e:
    print("ERROR", e)
    L = seen["lab"]
    job = str(e).split(": ")[1].split(" ")[0] if "livelock" in str(e) else None
    if job is None:  # stalled: show what each pending op waits on
        now = int(L.env.now)
        pend = sorted((L.plan.start.get(o, 10**9), o) for o, r in L.run.items() if r.status == "pending" and o in L.plan.start)[:12]
        for t, o in pend:
            preds = [(l.before, L.run[l.before].status) for l in L._preds(o)]
            print("  ", o, "plan", t, "preds", preds, "res", L.plan.assigned.get(o))
        print("  zones", {k: (v.first if v else None) for k, v in L.zone.items()}, "waits", list(L._supply_waits))
        for sp in [v for v in L.zone.values() if v]:
            zj = sp.first.split(".")[0]
            for o, r in L.run.items():
                if o.startswith(zj + "."):
                    print("   zone job", o, r.status, "plan", L.plan.start.get(o), "res", L.plan.assigned.get(o), "preds", [(l.before, L.run[l.before].status) for l in L._preds(o)])
            print("   span", sp.first, sp.link, sp.last, "inner", getattr(sp, "inner", None))
        print("  running", [o for o, r in L.run.items() if r.status == "running"][:10])
        o = pend[0][1]; res = L.plan.assigned.get(o); sp = L.spans.get(o)
        print("  first pending", o, "status", L.run[o].status, "plan", L.plan.start.get(o), "now", now, "res", res,
              "down", L._is_down(res, now), "occupancy", sorted(L.occupancy[res]), "cap", L.resources[res].capacity)
        if sp:
            print("  span first/link/last/inner", sp.first, sp.link, sp.last, sp.inner, "zone of link res", L.zone.get(L.plan.assigned.get(sp.link)),
                  "status first", L.run[sp.first].status if sp.first in L.run else None)
        print("  last replans", [(x["t"], x["reasons"], x["solver"]) for x in L.reschedules[-5:]])
        w = L._supply_waits
        for j in w:
            print("  waiting", j, L.plates[j].cell_line, "release", L.release.get(j), "load plan", L.plan.start.get(f"{j}.mv_hotel_lh"),
                  "status", getattr(L.run.get(f"{j}.mv_hotel_lh"), "status", None), "lh", L.plan.assigned.get(f"{j}.dispense"))
        print("  B004 line", L.plates["B004"].cell_line if "B004" in L.plates else None, "release", L.release.get("B004"))
        raise SystemExit
    S = L.supplies
    print("plan max_age_min", S.plan.max_age_min, "ul_per_well", S.plan.ul_per_well)
    print("checks for", job)
    for row in [r for r in log if r[1] == job][-10:]:
        print("  t", row[0], "lh", row[2], "batch made", row[3], "drug_ok", row[4], "cells_ok", row[5], "need/left", row[6])
    print("ops of", job)
    for oid, r in L.run.items():
        if oid.startswith(job + "."):
            print("  ", oid, r.status, "plan start", L.plan.start.get(oid), "start", r.start)
    print("prep jobs", [(k, v) for k, v in L.supply_job.items()])
    for oid, r in L.run.items():
        if oid.startswith("PREP") and r.status != "done":
            print("  pending prep", oid, r.status, "plan start", L.plan.start.get(oid))
    ev = [e for e in L.events if e["kind"] in ("cells_order", "cell_swap", "prep_order", "prep")][-14:]
    for e in ev:
        print("  ev", e["t"], e["kind"], {k: v for k, v in e.items() if k in ("job", "reason", "line", "prev", "made")})
    starts = sorted((r.start, oid) for oid, r in L.run.items() if r.start is not None and oid.endswith(".dispense"))[-6:]
    print("  last dispenses", starts)
    print("last reschedules", [(x["t"], x["reasons"], x["solver"]) for x in L.reschedules[-4:]])
