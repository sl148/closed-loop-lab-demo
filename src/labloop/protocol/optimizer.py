"""Deck design optimizer (per campaign).

Objective = expected liquid-handler time per plate over a sample of the campaign's plates:
protocol time on the deck + tip-rack resupply time. Racks on the tip carrier are shared
between tip types in proportion to use; when a type runs out, the arm swaps an empty rack
for a full one from the rack hotel (two arm transfers, with the head idle meanwhile).

Designs are grouped by what changes the *plan* (tip types, reuse); each group is planned once
per sample plate and then timed on every carrier placement in the group.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace

from labloop.protocol.deck import BASELINE, DeckDesign, candidate_designs
from labloop.protocol.planner import plan_plate, protocol_time
from labloop.types import PlateMap

RACK_SWAP_S = 2 * (7.0 + 3.0) + 6.0  # arm: empty rack to hotel, full rack to deck (+ re-teach check)


@dataclass
class DeckScore:
    design: DeckDesign
    protocol_s: float  # mean per plate
    swaps: float  # expected rack swaps per plate
    total_s: float

    def to_dict(self) -> dict:
        return {"design": self.design.to_dict(), "protocol_min": round(self.protocol_s / 60, 2),
                "swaps_per_plate": round(self.swaps, 2), "lh_min_per_plate": round(self.total_s / 60, 2)}


def rack_split(tips: dict[str, int], racks: int) -> dict[str, int]:
    """Racks per tip type, at least one each, the rest by demand."""
    types = sorted(tips)
    if len(types) == 1:
        return {types[0]: racks}
    split = {t: 1 for t in types}
    for _ in range(racks - len(types)):
        t = max(types, key=lambda t: tips[t] / (96 * split[t]))
        split[t] += 1
    return split


def expected_swaps(tips: dict[str, int], racks: int) -> float:
    split = rack_split(tips, racks)
    return sum(max(0.0, n / (96 * split[t]) - 0) for t, n in tips.items())  # fractional swaps per plate


def score(design: DeckDesign, protocols) -> DeckScore:
    prot = sum(protocol_time(p, design) for p in protocols) / len(protocols)
    tips: dict[str, float] = defaultdict(float)
    for p in protocols:
        for t, n in p.tips_used.items():
            tips[t] += n / len(protocols)
    swaps = expected_swaps(dict(tips), design.racks_on_deck)
    return DeckScore(design, prot, swaps, prot + swaps * RACK_SWAP_S)


def optimize_deck(sample: list[PlateMap], reuse: bool = True, top: int = 5) -> tuple[DeckScore, list[DeckScore], DeckScore]:
    """Returns (best, top-k, baseline score)."""
    groups: dict[tuple, list[DeckDesign]] = defaultdict(list)
    for d in candidate_designs():
        d = replace(d, tip_reuse=reuse)
        groups[(d.cell_tip, d.drug_tip, d.cell_source, d.tip_reuse)].append(d)
    scored: list[DeckScore] = []
    for key, designs in groups.items():
        protocols = [plan_plate(pm, designs[0]) for pm in sample]
        # Rail placement only moves zones: time the same plans on every placement.
        for d in designs:
            scored.append(score(d, protocols))
    scored.sort(key=lambda s: s.total_s)
    base = score(BASELINE, [plan_plate(pm, BASELINE) for pm in sample])
    return scored[0], scored[:top], base


def lh_minutes(pm: PlateMap, design: DeckDesign) -> tuple[int, dict[str, int]]:
    """Dispense duration (min) and tips used for one plate on a fixed deck."""
    p = plan_plate(pm, design)
    return math.ceil(protocol_time(p) / 60), p.tips_used
