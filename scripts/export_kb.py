"""The document archive and the agent's tool rules, for the live agent in the web page (M10 C).

    PYTHONPATH=src python scripts/export_kb.py

Writes
- web/data/kb.json: the archive's pages and search chunks, so the page can search and quote it.
- web/data/agent_tool_cases.json and tests/fixtures/agent_tool_cases.json: search hits, quote checks,
  numbers read from quotes, and accept/reject verdicts for proposals, all computed by the Python tools.
  The page's JavaScript copies of these tools must reproduce every case (its selfTest), and
  tests/test_agent_tool_parity.py keeps the Python side from drifting.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from labloop.agent.constraint_agent import Toolbox, numbers_in
from labloop.kb.archive import Archive, snippet

KB = Path("data/kb")
QUERIES = ["equilibrate plate room temperature minutes before reading", "aqueous solution stability one day",
           "cross-contamination cell line", "luminescent signal half-life", "incubate 37 C CO2 hours",
           "clean between cell lines liquid handler", "°C storage −20 µM", "plate reader 10 minutes stabilize"]
QUOTES = [("promega_tb288", 7, "Equilibrate the plate and its contents at room temperature for approximately 30 minutes."),
          ("promega_tb288", 7, "equilibrate   the PLATE and its contents at room temperature"),
          ("promega_tb288", 6, "Equilibrate the plate and its contents at room temperature for approximately 30 minutes."),
          ("cayman_sorafenib", 1, "We do not recommend storing the aqueous solution for more than one day."),
          ("atcc_culture_guide", 26, "short"), ("nope", 1, "anything at all here")]
EQ = "Equilibrate the plate and its contents at room temperature for approximately 30 minutes."
AQ = "We do not recommend storing the aqueous solution for more than one day."
PROPOSALS = [
    ({"kind": "min_between", "a": "to_reader", "a_anchor": "start", "b": "read", "minutes": 30}, "promega_tb288", 7, EQ, False),
    ({"kind": "min_between", "a": "to_reader", "a_anchor": "start", "b": "read", "minutes": 45}, "promega_tb288", 7, EQ, False),
    ({"kind": "min_between", "a": "read", "b": "to_reader", "minutes": 30}, "promega_tb288", 7, EQ, False),
    ({"kind": "min_between", "a": "dispense", "b": "read", "minutes": 30}, "promega_tb288", 7, EQ, False),
    ({"kind": "max_between", "a": "dispense", "b": "read", "minutes": 30}, "promega_tb288", 7, EQ, False),
    ({"kind": "deadline_from_release", "step": "dispense", "minutes": 1440}, "cayman_sorafenib", 1, AQ, False),
    ({"kind": "deadline_from_release", "step": "dispense", "minutes": 720}, "cayman_sorafenib", 1, AQ, False),
    ({"kind": "changeover", "minutes": 15}, "promega_tb288", 7, EQ, False),
    ({"kind": "changeover", "minutes": 15}, "promega_tb288", 7, EQ, True),
    ({"kind": "make_it_faster", "minutes": 5}, "promega_tb288", 7, EQ, False),
    ({"kind": "min_between", "a": "to_reader", "b": "read", "minutes": 30}, "promega_tb288", 7, "room temperature for about 30 min", False),
]


def cases() -> dict:
    archive = Archive(KB)
    cfg = yaml.safe_load(Path("configs/level4.yaml").read_text())
    out = {"search": [{"query": q, "hits": [[h["doc"], h["page"], h["score"]] for h in archive.search(q, 5)],
                       "snippet": snippet(archive.search(q, 1)[0]["text"], q)} for q in QUERIES],
           "verify": [{"doc": d, "page": p, "quote": q, "ok": archive.verify_quote(d, p, q)} for d, p, q in QUOTES],
           "numbers": [{"text": t, "numbers": sorted(numbers_in(t))} for t in (EQ, AQ, "for 2 to 3.5 hours, twelve wells", "none")],
           "propose": []}
    for spec, doc, page, quote, assumed in PROPOSALS:
        box = Toolbox(archive, cfg)
        box.slot = "case"
        res = box.propose(dict(spec), doc, page, quote, assumed)
        out["propose"].append({"spec": spec, "doc": doc, "page": page, "quote": quote, "value_assumed": assumed,
                               "accepted": res.startswith("accepted"), "hard": (box.proposals.get("case") or {}).get("hard")})
    return out


def main() -> None:
    pages = json.loads((KB / "pages.json").read_text())
    chunks = [json.loads(line) for line in (KB / "chunks.jsonl").read_text().splitlines() if line.strip()]
    docs = yaml.safe_load((KB / "manifest.yaml").read_text())["docs"]
    kb = {"docs": [{"id": d["id"], "title": d["title"]} for d in docs], "pages": pages,
          "chunks": [[c["doc"], c["page"], c["text"]] for c in chunks]}
    Path("web/data").mkdir(parents=True, exist_ok=True)
    Path("web/data/kb.json").write_text(json.dumps(kb, separators=(",", ":")))
    text = json.dumps(cases(), indent=1, ensure_ascii=False)
    for p in (Path("web/data/agent_tool_cases.json"), Path("tests/fixtures/agent_tool_cases.json")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(f"kb.json {Path('web/data/kb.json').stat().st_size / 1e3:.0f} kB, {len(chunks)} chunks")


if __name__ == "__main__":
    main()
