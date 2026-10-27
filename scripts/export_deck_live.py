"""The deck design search, for the web page's live mode (M11): the sample plates it plans, and
reference scores the page's JavaScript port must reproduce.

    PYTHONPATH=src python scripts/export_deck_live.py --config configs/level4.yaml

Writes web/data/deck_live.json (sample plates, the agent's labware rules) and tests/fixtures/deck_cases.json
(Python scores of a spread of decks, the best deck, the baseline, the candidate count, and the best deck and
count under the agent's rules). tests/test_deck_live_parity.py keeps
the Python side from drifting; the page's selfTest checks its port against the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from optimize_deck import sample_plates  # noqa: E402

from labloop.agent.labware import deck_rules  # noqa: E402
from labloop.protocol.deck import BASELINE, candidate_designs  # noqa: E402
from labloop.protocol.optimizer import allowed, optimize_deck, score  # noqa: E402
from labloop.protocol.planner import plan_plate  # noqa: E402
from labloop.types import Condition  # noqa: E402


def plates_json(sample) -> list[dict]:
    out = []
    for pm in sample:
        wells = []
        for (r, c), w in sorted(pm.wells.items()):
            wells.append([r, c, list(w.doses) if isinstance(w, Condition) else w])
        out.append({"id": pm.plate_id, "rows": pm.spec.rows, "cols": pm.spec.cols, "wells": wells})
    return out


LABWARE = Path("results/constraints/open_haiku/labware.json")  # the agent's rules (configs/deck_level4_agentlw2.json)


def rules() -> dict:
    lw = json.loads(LABWARE.read_text())
    return lw.get("deck_rules") or deck_rules(lw["rules"])


def cases(sample, rules: dict | None = None) -> dict:
    from dataclasses import replace

    cands = [replace(d, tip_reuse=True) for d in candidate_designs()]
    picks = cands[:: max(1, len(cands) // 12)][:12]
    best, top, base = optimize_deck(sample)
    row = lambda s: {"design": s.design.to_dict(), "total_s": round(s.total_s, 4), "protocol_s": round(s.protocol_s, 4),
                     "swaps": round(s.swaps, 6)}
    out = {"n_candidates": len(cands), "best": row(best), "baseline": row(base),
           "designs": [row(score(d, [plan_plate(pm, d) for pm in sample])) for d in picks]}
    if rules:
        rbest, _, _ = optimize_deck(sample, rules=rules)
        out["ruled"] = {"rules": rules, "n_candidates": sum(allowed(d, rules) for d in cands), "best": row(rbest)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/level4.yaml")
    args = ap.parse_args()
    sample = sample_plates(yaml.safe_load(Path(args.config).read_text()))
    r = rules()
    Path("web/data/deck_live.json").write_text(json.dumps({"config": args.config, "plates": plates_json(sample),
                                                           "baseline": BASELINE.to_dict(), "rules": r}, separators=(",", ":")))
    c = cases(sample, r)
    Path("tests/fixtures/deck_cases.json").write_text(json.dumps(c, indent=1))
    print(f"{len(sample)} plates, {c['n_candidates']} decks, best {c['best']['total_s'] / 60:.2f} min/plate; "
          f"agent rules: {c['ruled']['n_candidates']} decks, best {c['ruled']['best']['total_s'] / 60:.2f} min/plate")


if __name__ == "__main__":
    main()
