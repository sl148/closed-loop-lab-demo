"""Export one recorded run of the whole agentic loop for the web page (M10 B1; run on the server).

    PYTHONPATH=src python scripts/export_loop.py --variant agent_review --seed 0 [--open results/constraints/open_haiku]

Writes web/data/replay_console.json: the usual replay bundle of that run (scripts/export_web.py replay),
plus what the Console's steps column needs:
- the events the plain bundle drops (cleaning, lag slips, QC fails, constraint updates, reviews, and the
  hidden-limit "truth" events, marked hidden: the lab never saw them, the viewer may),
- failures with whether they asked for a replan, replans with their solver status,
- "loop": the task as given, the agent's search before launch (trace and verdicts), the constraints it
  launched with, the arm routes checked in MuJoCo, every in-loop review (evidence, the agent's trace,
  its decision, what the lab applied) with the constraint history, the deck design search
  (configs/deck_level4.json), the camera's check of the deck before launch (sim3d/deck_check.py),
  the camera's detection curve, and the run's results next to
  blind / agent-only / gold runs of the same seeds.
Also writes web/data/lab_data.xlsx: the run's records as linked tables (labloop/lims.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from export_agent import run_view, step  # noqa: E402
from export_web import OUT, RES, dump, replay  # noqa: E402

from labloop.lims import tables, write_xlsx  # noqa: E402
from labloop.metrics import summarize  # noqa: E402
from labloop.protocol.optimizer import rack_split  # noqa: E402
from labloop.sim3d.deck_check import expected_layout, run_check  # noqa: E402
from labloop.sim3d.perception import POS_TOL_MM, YAW_TOL_DEG  # noqa: E402
from labloop.review.providers import read_log  # noqa: E402

EXTRA = ("clean", "truth", "lag_slip", "qc_fail", "constraints_update", "review", "undetected_misplacement",
         "quarantine", "confirm")
COMPARE = {"base": [("blind_det", "No limits"), ("agent_open_det", "Agent, before launch"), (None, "Agent + reviews"),
                    ("gold_det", "All real limits")],
           # labware truth (configs/truth_labware.yaml): agent rows use the agent's labware, "no limits" the fastest deck
           "lw": [("blind_lw_det", "No limits"), ("agent_open_lw_det", "Agent, before launch"), (None, "Agent + reviews"),
                  ("lw_right_det", "All real limits")],
           # labware + liquid prep truth, liquids made by the lab, confirmation round (configs/level4.yaml)
           "v2": [("blind_v2", "No limits"), ("agent_open_v2", "Agent, before launch"), (None, "Agent + reviews"),
                  ("gold_v2", "All real limits")],
           # v2 truth plus solubility/carryover/DMSO limits, pipetting run through PLR (configs/truth_prep.yaml)
           "v3": [("blind_v3", "No limits"), ("agent_open_v3", "Agent, before launch"), (None, "Agent + reviews"),
                  ("gold_v3", "All real limits")]}
METRICS = ["corrupt_accepted", "invalid_plates", "makespan_h", "regret", "ttt_h", "hit"]


def results(variant: str, label_self: str = "Agent + reviews", compare: str = "base") -> list[dict]:
    out = []
    for v, label in COMPARE[compare]:
        if v is None:
            label = label_self
        p = RES / "m9/level4" / f"{v or variant}.json"
        if not p.exists():
            continue
        runs = json.loads(p.read_text())
        m = summarize(runs)
        truth = {k: sum(r["truth"].get(k, 0) for r in runs) / len(runs) for k in ("changeover", "read_equilibration", "qc_fail", "labware", "drug_rt", "prep")}
        out.append({"variant": v or variant, "label": label, "mean": {k: m.get(k) for k in METRICS},
                    "seeds": [{k: summarize([r]).get(k) for k in METRICS} for r in runs], "truth": truth})
    return out


def labware(run_dir: Path) -> dict | None:
    """The agent's labware phase (agent/labware.py): its rules, what it left at the default, and its tool calls."""
    p = run_dir / "labware.json"
    if not p.exists():
        return None
    lw = json.loads(p.read_text(encoding="utf-8"))
    tr = run_dir / "labware_trace.jsonl"
    trace = [json.loads(x) for x in tr.read_text(encoding="utf-8").splitlines() if x.strip()] if tr.exists() else []
    from labloop.protocol.planner import CELLS_UL, DRUG_UL

    return {"rules": lw["rules"], "deck_rules": lw["deck_rules"], "abstained": lw.get("abstained", {}),
            "usage": lw.get("usage"), "model": lw.get("model"), "volumes_ul": {"cells": CELLS_UL[384], "drug": DRUG_UL[384]},
            "trace": [{"tool": t["tool"], "args": t["args"], "out": t["out"][:300]} for t in trace]}


def prep(run_dir: Path) -> dict | None:
    """The agent's prep phase (agent/prep.py): its plan, the quotes it relied on, the numbers, its tool calls."""
    p = run_dir / "prep.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    tr = run_dir / "prep_trace.jsonl"
    trace = [json.loads(x) for x in tr.read_text(encoding="utf-8").splitlines() if x.strip()] if tr.exists() else []
    return {"plan": d["plan"], "sources": d["sources"], "checks": d["checks"], "usage": d.get("usage"), "model": d.get("model"),
            "trace": [{"tool": t["tool"], "args": {k: v for k, v in t["args"].items() if k != "plan"}, "out": t["out"][:300]} for t in trace]}


def deck_check(bundle: dict) -> dict:
    """The camera's check of the deck as set up for this run (racks split by tip use, as the page draws them)."""
    deck, tot = bundle["deck"], {}
    for p in bundle.get("protocols", {}).values():
        for t in p["passes"]:
            tt = (t[10] if len(t) > 10 else None) or (deck["cell_tip"] if t[0] == 0 else deck["drug_tip"])
            tot[tt] = tot.get(tt, 0) + 1
    racks = [t for t, n in rack_split(tot or {"300uL": 1}, deck.get("racks_on_deck", 5)).items() for _ in range(n)][:5]
    return run_check(expected_layout(deck, bundle.get("drugs", []), racks), bundle.get("drugs", []), seed=0)


def reviews(run: dict, log: dict, pages: dict) -> list[dict]:
    out = []
    for r in run.get("reviews", []):
        e = log.get(r["key"], {})
        out.append({**r, "trace": [step(t, pages) for t in e.get("trace", []) if t.get("tool")],
                    "usage": e.get("usage"), "model": e.get("model"),
                    "turns": [t["text"] for t in e.get("turns", []) if t.get("text", "").strip()][-2:]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="agent_review")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--open", default="results/constraints/open_haiku")
    ap.add_argument("--name", default="console", help="writes web/data/replay_<name>.json (live mode: combo_<EDC>[_rev])")
    ap.add_argument("--label", default="Agent + reviews", help="this run's name in the results comparison")
    ap.add_argument("--compare", choices=sorted(COMPARE), default="base", help="which runs the results compare against")
    ap.add_argument("--deck", default="configs/deck_level4.json", help="the deck search output this run used")
    args = ap.parse_args()
    runs = json.loads((RES / "m9/level4" / f"{args.variant}.json").read_text())
    run = next(r for r in runs if r["seed"] == args.seed)
    title = "Level 4, two campaigns, hidden lab limits, constraints from the agent, reviews in the loop"
    bundle = replay(run, title, args.seed)
    resched = {(e["t"], e["event"].get("plate"), e["event"]["type"]): e["decision"].get("reschedule")
               for e in run["events"] if e["kind"] == "failure"}
    for f in bundle["failures"]:
        f["replan"] = bool(resched.get((f["t"], f["plate"], f["type"])))
    for b, r in zip(bundle["replans"], run["reschedules"]):
        b["status"] = r.get("status")
    bundle["events"] = [{**e, "hidden": e["kind"] in ("truth", "undetected_misplacement")}
                        for e in run["events"] if e["kind"] in EXTRA]

    pages = json.loads(Path("data/kb/pages.json").read_text())
    gold = {c["id"]: c for c in yaml.safe_load(Path("configs/constraints_gold.yaml").read_text())["constraints"]}
    agent = run_view(Path(args.open), gold, pages)
    task = next((t["lab"] for t in agent["trace"] if t["tool"] == "describe_lab"), None)
    motion = RES / "sim3d/motion_costs.json"
    log = read_log(RES / "m9/reviews" / args.variant / "decisions.jsonl")
    bundle["loop"] = {
        "task": task,
        "agent": {k: agent[k] for k in ("model", "usage", "slots", "trace")},
        "launch": json.loads((Path(args.open) / "constraints.json").read_text())["constraints"],
        "motion": {r: {"s": c.get("simulated_s"), "collisions": len(c.get("collisions", []))}
                   for r, c in json.loads(motion.read_text())["checks"].items()} if motion.exists() else {},
        "reviews": reviews(run, log, pages),
        "constraint_history": run.get("constraint_history", []),
        "plate_version": run.get("plate_version", {}),
        "precheck": json.loads(pc.read_text()) if (pc := RES / "precheck/open_haiku/level4.json").exists() else None,
        "deck": json.loads(Path(args.deck).read_text()),
        "labware": labware(Path(args.open)),
        "prep": prep(Path(args.open)),
        "camera": {"curve": json.loads(cam.read_text()) if (cam := RES / "sim3d/perception.json").exists() else [],
                   "tol_mm": POS_TOL_MM, "tol_deg": YAW_TOL_DEG},
        "deck_check": deck_check(bundle),
        "results": results(args.variant, args.label, args.compare),
        "gold": list(gold.values()),
    }
    kept = {e["plate"]: e.get("corrupt_accepted", 0) for e in run["events"] if e["kind"] == "read"}
    for r in bundle["reads"]:
        r["corrupt"] = kept.get(r["plate"], 0)  # sim only: corrupted values that passed QC (Results tab, truth sheet)
    raw = {pid: {r["well"]: r["value"] for r in (p["reads"] or {}).values()} for pid, p in run["plates"].items()}
    bundle["supplies_ledger"] = (run.get("supplies") or {}).get("ledger", [])
    lims = tables(bundle, raw)
    bundle.pop("supplies_ledger")
    for pid, r in raw.items():
        if pid in bundle["plates"]:
            bundle["plates"][pid]["raw"] = r  # the reader's raw signal per well (Data tab, workbook)
    bundle["lims"] = {k: v for k, v in lims.items() if k != "wells"}  # the page builds wells from the plates it already has
    OUT.mkdir(parents=True, exist_ok=True)
    if args.name == "console":
        try:
            write_xlsx(lims, OUT / "lab_data.xlsx")
        except ImportError:
            print("openpyxl missing: no lab_data.xlsx")
    name = f"replay_{args.name}.json"
    (OUT / name).write_text(dump(bundle))
    print(f"{name} {(OUT / name).stat().st_size / 1e6:.2f} MB, "
          f"{len(bundle['loop']['reviews'])} reviews, results {[x['variant'] for x in bundle['loop']['results']]}")


if __name__ == "__main__":
    main()
