"""Optimize the liquid-handler deck once per campaign config and save it.

    uv run python scripts/optimize_deck.py --config configs/level2.yaml

Samples plates the campaign would plausibly run (random combinations per group, the config's
layout), searches every valid deck (carrier rails, tip types, racks, cell source) for the
lowest liquid-handler minutes per plate including tip-rack swaps, and writes
configs/deck_<name>.json with the best design, the top 5, and the fixed-deck/fresh-tip baseline.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plates", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    camps = cfg.get("campaigns") or [cfg]
    rng = np.random.default_rng(0)
    spec = PlateSpec(**cfg.get("plate", {}))
    layout = build_layout(cfg.get("layout", "sequential"), spec, cfg.get("layout_cfg"), 0)
    sample = []
    for i in range(args.plates):
        space = build_space(camps[i % len(camps)]["space"])
        group = (space.cell_lines[i % len(space.cell_lines)], space.timepoints_h[i % len(space.timepoints_h)])
        pool = [c for c in space.combos if c.group() == group]  # a plate holds one group
        pick = [pool[k] for k in rng.choice(len(pool), min(layout.capacity, len(pool)), replace=False)]
        sample.append(layout.place(pick)[0])
    best, top, base = optimize_deck(sample)
    out = {"config": args.config, "sample_plates": args.plates, "best": best.to_dict(),
           "top": [s.to_dict() for s in top], "baseline": base.to_dict()}
    path = Path("configs") / f"deck_{cfg['name']}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"baseline {base.total_s / 60:.1f} min/plate -> best {best.total_s / 60:.1f} min/plate")
    print("best:", best.design.to_dict())
    print("wrote", path)


if __name__ == "__main__":
    main()
