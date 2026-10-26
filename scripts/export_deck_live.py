"""The deck design search, for the web page's live mode (M11): the sample plates it plans, and
reference scores the page's JavaScript port must reproduce.

    PYTHONPATH=src python scripts/export_deck_live.py --config configs/level4.yaml

Writes web/data/deck_live.json (sample plates) and tests/fixtures/deck_cases.json (Python scores of a
spread of decks, the best deck, the baseline, the candidate count). tests/test_deck_live_parity.py keeps
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

from labloop.protocol.deck import BASELINE, candidate_designs  # noqa: E402
from labloop.protocol.optimizer import optimize_deck, score  # noqa: E402
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


def cases(sample) -> dict:
    from dataclasses import replace

    cands = [replace(d, tip_reuse=True) for d in candidate_designs()]
    picks = cands[:: max(1, len(cands) // 12)][:12]
    best, top, base = optimize_deck(sample)
    row = lambda s: {"design": s.design.to_dict(), "total_s": round(s.total_s, 4), "protocol_s": round(s.protocol_s, 4),
                     "swaps": round(s.swaps, 6)}
    return {"n_candidates": len(cands), "best": row(best), "baseline": row(base),
            "designs": [row(score(d, [plan_plate(pm, d) for pm in sample])) for d in picks]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/level4.yaml")
    args = ap.parse_args()
    sample = sample_plates(yaml.safe_load(Path(args.config).read_text()))
    Path("web/data/deck_live.json").write_text(json.dumps({"config": args.config, "plates": plates_json(sample),
                                                           "baseline": BASELINE.to_dict()}, separators=(",", ":")))
    c = cases(sample)
    Path("tests/fixtures/deck_cases.json").write_text(json.dumps(c, indent=1))
    print(f"{len(sample)} plates, {c['n_candidates']} decks, best {c['best']['total_s'] / 60:.2f} min/plate")


if __name__ == "__main__":
    main()
