"""Optimize the liquid-handler deck once per campaign config and save it.

    uv run python scripts/optimize_deck.py --config configs/level2.yaml

Samples plates the campaign would plausibly run (random combinations per group, the config's
layout), searches every valid deck (carrier rails, tip types, racks, cell source) for the
lowest liquid-handler minutes per plate including tip-rack swaps, and writes
configs/deck_<name>.json with the best design, the top 5, and the fixed-deck/fresh-tip baseline.

With --labware RUN/labware.json (agent/labware.py) the search keeps the agent's labware rules;
the output then also records the fastest deck without them, so the rules' time cost shows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from labloop.design.base import build_space
from labloop.layout.layouts import build_layout
from labloop.protocol.optimizer import optimize_deck
from labloop.types import PlateSpec


def sample_plates(cfg: dict, n: int = 4) -> list:
    """Plates the campaigns would plausibly run: random combinations per group, the config's layout."""
    camps = cfg.get("campaigns") or [cfg]
    rng = np.random.default_rng(0)
    spec = PlateSpec(**cfg.get("plate", {}))
    layout = build_layout(cfg.get("layout", "sequential"), spec, cfg.get("layout_cfg"), 0)
    sample = []
    for i in range(n):
        space = build_space(camps[i % len(camps)]["space"])
        group = (space.cell_lines[i % len(space.cell_lines)], space.timepoints_h[i % len(space.timepoints_h)])
        pool = [c for c in space.combos if c.group() == group]  # a plate holds one group
        pick = [pool[k] for k in rng.choice(len(pool), min(layout.capacity, len(pool)), replace=False)]
        sample.append(layout.place(pick)[0])
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plates", type=int, default=4)
    ap.add_argument("--labware", help="labware.json from the agent: search only within its rules")
    ap.add_argument("--out", help="output path (default configs/deck_<name>.json)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample = sample_plates(cfg, args.plates)
    every: list = []
    rules = json.loads(Path(args.labware).read_text())["deck_rules"] if args.labware else None
    best, top, base = optimize_deck(sample, all_scores=every, rules=rules)
    mins = np.array([s.total_s / 60 for s in every])
    counts, edges = np.histogram(mins, bins=24)
    out = {"config": args.config, "sample_plates": args.plates, "best": best.to_dict(),
           "top": [s.to_dict() for s in top], "baseline": base.to_dict(),
           "candidates": {"n": len(every), "lh_min_hist": {"edges": [round(float(e), 3) for e in edges],
                                                        "counts": counts.tolist()}}}
    if rules:
        out["labware"] = {"rules": rules, "source": args.labware, "fastest_without_rules": optimize_deck(sample)[0].to_dict()}
    path = Path(args.out) if args.out else Path("configs") / f"deck_{cfg['name']}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"baseline {base.total_s / 60:.1f} min/plate -> best {best.total_s / 60:.1f} min/plate")
    print("best:", best.design.to_dict())
    print("wrote", path)


if __name__ == "__main__":
    main()
