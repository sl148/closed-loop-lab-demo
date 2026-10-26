"""Export the constraint agent's run and its effect on the lab into web/data/agent.json (M9 view).

    uv run python scripts/export_agent.py [--slots results/constraints/pilot_haiku] [--open results/constraints/open_haiku]

Reads each agent run (trace.jsonl, summary.json, constraints.json): one with the questions given
(slots mode) and one where the agent chose them (open mode), plus any results/constraints/open_haiku*
runs for recall across runs. Also the archive (data/kb), the lab's hidden limits
(configs/truth.yaml), the gold constraints, and the Level 4 runs in results/m9/level4/. Missing
runs are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from labloop.agent.constraint_agent import SLOTS
from labloop.constraints.audit import audit
from labloop.constraints.compile import load_constraints
from labloop.kb.archive import Archive, snippet
from labloop.metrics import summarize

KB, RES, OUT = Path("data/kb"), Path("results"), Path("web/data")
SHORT = {"promega_tb288": "Promega TB288", "atcc_culture_guide": "ATCC culture guide",
         "cayman_sorafenib": "Cayman sorafenib", "cayman_mk4827": "Cayman MK-4827",
         "cayman_erlotinib": "Cayman erlotinib", "cayman_vinblastine": "Cayman vinblastine"}
HIT = re.compile(r'"doc": "([^"]+)", "title": "[^"]*", "page": (\d+), "score": ([\d.]+), "snippet": "((?:[^"\\]|\\.)*)')
METRICS = ["corrupt_accepted", "invalid_plates", "regret", "ttt_h", "hit", "makespan_h"]
# the reproducible (deterministic-scheduler) Level 4 runs, where they exist
RUN_FILE = {"blind": "blind_det", "agent": "agent", "agent_open": "agent_open_det", "gold": "gold_det"}


def run_file(v: str) -> Path:
    p = RES / "m9/level4" / f"{RUN_FILE.get(v, v)}.json"
    return p if p.exists() else RES / "m9/level4" / f"{v}.json"


def squash(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def window(text: str, query: str, n: int = 280) -> str:
    return snippet(text, query, n)


def step(t: dict, pages: dict) -> dict:
    """One tool call, trimmed to what the viewer draws."""
    a, out = t["args"], t["out"]
    s = {"slot": t["slot"], "tool": t["tool"], "t": t["t"]}
    if t["tool"] == "search_corpus":
        s["query"] = a["query"]
        try:
            hits = [(h["doc"], h["page"], h["score"], h["snippet"]) for h in json.loads(out)]
        except json.JSONDecodeError:  # the logged result was cut at 4000 chars
            hits = [(d, p, sc, sn.replace('\\"', '"')) for d, p, sc, sn in HIT.findall(out)]
        s["hits"] = [{"doc": d, "page": int(p), "score": float(sc),
                      "snippet": window(pages[d][int(p) - 1], a["query"]) if d in pages else squash(sn, 260)} for d, p, sc, sn in hits]
    elif t["tool"] == "read_pages":
        s.update(doc=a["doc"], first=a["first"], last=a["last"])
    elif t["tool"] == "propose":
        s.update(spec=a["spec"], doc=a["doc"], page=a["page"], quote=a["quote"], assumed=a.get("value_assumed", False),
                 ok=out.startswith("accepted"), result=squash(out, 240))
        if a.get("limit_id"):
            s["limit_id"] = a["limit_id"]
    elif t["tool"] == "hypothesize":
        s.update({k: a[k] for k in ("limit_id", "concern", "kind", "steps", "why")}, ok=out.startswith("recorded"),
                 result=squash(out, 160))
    elif t["tool"] == "describe_lab":
        lab = json.loads(out)
        s["lab"] = {k: lab[k] for k in ("goal", "workflow", "devices", "materials", "campaigns") if k in lab}
    elif t["tool"] == "abstain":
        s["reason"] = a["reason"]
    elif t["tool"] == "simulate":
        s["result"] = squash(out, 400)
    return s


def verdict(gold: dict | None, mine: dict | None) -> str:
    """How the agent's slot compares with what a perfect reader extracts."""
    if gold is None:
        return "abstain-ok" if mine is None else "extra"
    if mine is None:
        return "missed"
    same = mine["kind"] == gold["kind"] and mine["minutes"] == gold["minutes"]
    return ("match" if all(mine.get(k) == gold.get(k) for k in ("a", "a_anchor", "b", "step")) else "match-value") if same else "wrong-value"


def key(c: dict) -> tuple:
    """What a limit is about: its kind and the step it holds back (the later step of a gap)."""
    return c["kind"], c.get("b") or c.get("step") or ("read" if c["kind"] == "read_window" else "")


def covers(mine: dict, gold: dict) -> bool:
    """The agent's value is at least as strict as the gold one (a longer minimum, a shorter maximum)."""
    return mine["minutes"] >= gold["minutes"] if mine["kind"] in ("min_between", "changeover") else mine["minutes"] <= gold["minutes"]


def grade(mine: dict, gold: dict) -> str:
    if mine["minutes"] == gold["minutes"]:
        return "match" if all(mine.get(k) == gold.get(k) for k in ("a", "a_anchor", "b", "step")) else "match-value"
    return "covers" if covers(mine, gold) else "wrong-value"


def open_rows(summary: dict, mine: dict, gold: dict, notes: dict) -> list[dict]:
    """Open mode: one row per hypothesis (and sweep find), matched to the gold limits; a gold limit
    no hypothesis touched gets its own row ("never")."""
    hyps, taken, rows = summary.get("hypotheses", []), {}, {}
    for h in hyps:  # proposals first, so an abstained twin does not take the match
        p = mine.get(h["id"])
        g = next((k for k, c in gold.items() if k not in taken.values() and p and key(p) == key(c)), None)
        if g:
            taken[h["id"]] = g
    for h in hyps:
        if h["id"] not in taken and h["id"] not in mine:
            g = next((k for k, c in gold.items() if k not in taken.values() and h["kind"] == c["kind"]
                      and key(c)[1] in h["steps"]), None)
            if g:
                taken[h["id"]] = g
    for h in hyps:
        p, g = mine.get(h["id"]), gold.get(taken.get(h["id"]))
        v = (grade(p, g) if g else "extra") if p else ("missed" if g else "abstain-ok")
        rows[h["id"]] = {"id": h["id"], "question": h["concern"], "origin": h["origin"], "kind": h["kind"], "steps": h["steps"],
                         "why": h["why"], "verdict": v, "proposed": p, "gold": g, "abstain": summary["abstained"].get(h["id"]),
                         "note": notes.get(h["id"])}
    for k, c in gold.items():
        if k not in taken.values():
            rows[k] = {"id": k, "question": "", "origin": "gold", "kind": c["kind"], "steps": [], "why": "",
                       "verdict": "never", "proposed": None, "gold": c, "abstain": None, "note": None}
    return list(rows.values())


def run_view(run: Path, gold: dict, pages: dict) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    raw = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines() if line.strip()]
    trace = [step(t, pages) for t in raw]
    t0 = trace[0]["t"] if trace else 0
    for s in trace:
        s["t"] = round(s["t"] - t0, 1)
    mine = {c["id"]: c for c in json.loads((run / "constraints.json").read_text())["constraints"]}
    notes = {}
    for turn in summary.get("turns", []):
        if turn["text"].strip():
            notes[turn["slot"]] = squash(turn["text"], 600)
    mode = summary.get("mode", "slots")
    if mode == "open":
        slots = open_rows(summary, mine, gold, notes)
    else:
        slots = [{"id": k, "question": SLOTS[k], "origin": "given", "verdict": verdict(gold.get(k), mine.get(k)),
                  "proposed": mine.get(k), "gold": gold.get(k), "abstain": summary["abstained"].get(k),
                  "note": notes.get(k)} for k in dict.fromkeys(s["slot"] for s in trace)]
    defaults = next((json.loads(t["out"])["current_defaults"] for t in raw if t["tool"] in ("list_parameters", "describe_lab")), {})
    return {"mode": mode, "run": run.name, "model": summary["model"], "effort": summary.get("effort"),
            "usage": summary["usage"], "slots": slots, "trace": trace, "defaults": defaults}


def recall(gold: dict, pages: dict) -> list[dict]:
    """Per open-mode run: which gold limits it thought of and which it grounded, extras, cost."""
    out = []
    for run in sorted((RES / "constraints").glob("open_haiku*")):
        if not (run / "summary.json").exists():
            continue
        v = run_view(run, gold, pages)
        by_gold = {r["gold"]["id"]: r["verdict"] for r in v["slots"] if r["gold"]}
        out.append({"run": run.name, "usd": v["usage"]["usd"], "hypotheses": sum(r["origin"] != "gold" for r in v["slots"]),
                    "extras": sum(r["verdict"] == "extra" for r in v["slots"]), "gold": by_gold})
    return out


def usage(cons: str, variant: str) -> dict:
    """Is each constraint actually used? Audited on the Level 4 runs scheduled with it: plates (or
    cell-line switches) it covered, how many sat right at the limit (it binds), how many broke it;
    and the same limit on the blind runs, which never heard of it."""
    p, b = run_file(variant), run_file("blind")
    if not p.exists() or not Path(cons).exists():
        return {}
    specs = load_constraints(cons).specs
    out = {}
    for name, f in (("used", p), ("blind", b)):
        if not f.exists():
            continue
        for run in json.loads(f.read_text()):
            for cid, a in audit(specs, run).items():
                u = out.setdefault(cid, {}).setdefault(name, {"n": 0, "violated": 0, "tight": 0})
                for k in u:
                    u[k] += a[k]
    return out


def impact() -> dict:
    out = {}
    for v in ("blind", "agent", "agent_open", "gold"):
        p = run_file(v)
        if not p.exists():
            continue
        runs = json.loads(p.read_text())
        per = [summarize([r]) for r in runs]
        truth = {k: sum(r["truth"].get(k, 0) for r in runs) / len(runs) for k in ("changeover", "read_equilibration", "drug_rt", "qc_fail")}
        out[v] = {"mean": {k: summarize(runs).get(k) for k in METRICS}, "seeds": [{k: s.get(k) for k in METRICS} for s in per],
                  "truth": truth, "plates": len(runs[0]["plates"])}
    return out


def modules() -> dict:
    """What each module did with the constraints, from the Level 4 runs (seed 0 for timelines).

    sched: per variant, each plate's minutes from leaving the incubator to the read, and each cell-line
    switch's gap between dispenses on one liquid handler with whether the lab cleaned. bo: the throughput designer's picks with
    hours to result with and without the constraints. lab: one liquid handler's dispenses, cleanings
    and contaminations. pre: the pre-launch check's rollouts (scripts/precheck_rollouts.py)."""
    out: dict = {"sched": {}, "bo": {}, "lab": {}}
    for v in ("blind", "agent", "agent_open", "gold"):
        p = run_file(v)
        if not p.exists():
            continue
        runs = json.loads(p.read_text())
        eq, sw = [], []
        for run in runs:
            rows = {r["op"]: r for r in run["schedule"]}
            for pid in run["plates"]:
                a, b = rows.get(f"{pid}.mv_incubator_reader"), rows.get(f"{pid}.read")
                if a and b:
                    eq.append(b["start"] - a["start"])
            cleaned = {e["plate"] for e in run["events"] if e["kind"] == "clean"}
            disp: dict[str, list] = {}
            for pid, plate in run["plates"].items():
                r = rows.get(f"{pid}.dispense")
                if r:
                    disp.setdefault(r["resource"], []).append((r["start"], r["end"], plate["cell_line"], pid))
            for seq in disp.values():
                seq.sort()
                sw += [[s2 - e1, p2 in cleaned] for (s1, e1, l1, _), (s2, e2, l2, p2) in zip(seq, seq[1:]) if l1 != l2]
        out["sched"][v] = {"eq_gap": eq, "switch_gap": sw}
        dec = [d for c in runs[0]["campaigns"].values() for d in c.get("group_decisions", [])]
        out["bo"][v] = [{"t": d["t"], "chosen": f"{d['chosen'][0]}@{d['chosen'][1]}h", "scores": d["scores"]} for d in dec]
        run = runs[0]
        rows = {r["op"]: r for r in run["schedule"]}
        seq = sorted((rows[f"{pid}.dispense"]["start"], rows[f"{pid}.dispense"]["end"], plate["cell_line"], pid)
                     for pid, plate in run["plates"].items()
                     if f"{pid}.dispense" in rows and rows[f"{pid}.dispense"]["resource"] == "lh1")
        out["lab"][v] = {"lh": "lh1", "dispense": seq,
                         "clean": [[e["t"], e["plate"], e.get("delay", 0)] for e in run["events"] if e["kind"] == "clean" and e["lh"] == "lh1"],
                         "dirty": [[e["t"], e["plate"], e["wells"]] for e in run["events"]
                                   if e["kind"] == "truth" and e.get("limit") == "changeover" and e.get("lh") == "lh1"]}
    roll = RES / "m9/precheck_rollouts.json"
    if roll.exists():
        out["pre"] = json.loads(roll.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="results/constraints/pilot_haiku")
    ap.add_argument("--open", default="results/constraints/open_haiku")
    args = ap.parse_args()
    pages = json.loads((KB / "pages.json").read_text())
    gold = {c["id"]: c for c in yaml.safe_load(Path("configs/constraints_gold.yaml").read_text())["constraints"]}
    runs = {m: run_view(Path(p), gold, pages) for m, p in (("open", args.open), ("slots", args.slots))
            if (Path(p) / "summary.json").exists()}
    gold_file = "configs/constraints_gold.yaml"
    for m, p, v in (("open", args.open, "agent_open"), ("slots", args.slots, "agent")):
        if m in runs:
            runs[m]["use"] = usage(str(Path(p) / "constraints.json"), v)  # the agent's own constraints
            runs[m]["gold_use"] = usage(gold_file, v)  # the real limits, on the same runs
    manifest = yaml.safe_load((KB / "manifest.yaml").read_text())["docs"]
    truth = yaml.safe_load(Path("configs/truth.yaml").read_text())
    archive = Archive(KB)
    gold_pages = {f"{d}:{p}": k for k, c in gold.items() for d in pages for p in range(1, len(pages[d]) + 1)
                  if archive.verify_quote(d, p, c["source"]["quote"])}  # every page a perfect reader could cite
    data = {
        "schema": "labloop-agent/2",
        "runs": runs, "recall": recall(gold, pages),
        "docs": [{"id": d["id"], "short": SHORT.get(d["id"], d["id"]), "title": d["title"], "url": d.get("url"),
                  "pages": len(pages.get(d["id"], []))} for d in manifest],
        "gold_pages": gold_pages,
        "truth": {"limits": {k: truth[k] for k in ("read_equilibration_min", "drug_rt_stable_min", "changeover_wells")},
                  "sources": truth.get("sources", {})},
        "impact": impact(),
        "modules": modules(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agent.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"agent.json runs {list(runs)}, {(OUT / 'agent.json').stat().st_size / 1e3:.0f} kB, recall {len(data['recall'])} runs")


if __name__ == "__main__":
    main()
