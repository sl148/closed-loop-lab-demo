"""Nominal op durations (minutes). Arm moves and pipetting passes default to hand-set
constants; `labloop.sim3d` (M7) measures them from the MuJoCo cell and writes
configs/motion_costs.json, which `build_durations` picks up when present."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from labloop.types import PlateMap

MOTION_COSTS = Path(__file__).parents[3] / "configs" / "motion_costs.json"


@dataclass(frozen=True)
class DurationModel:
    lh_overhead_min: float = 3.0  # deck setup, tip loading
    lh_cycle_min: float = 0.5  # one 8-channel aspirate/dispense pass
    read_min_96: int = 6
    read_min_384: int = 10
    arm_move_min: dict[str, float] = field(default_factory=lambda: {
        "hotel>lh": 1.0, "lh>incubator": 1.2, "incubator>reader": 1.0, "reader>hotel": 1.0,
    })
    transfer_max_lag_min: int = 10  # a plate may wait this long between stations
    incubation_tolerance_min: int = 120  # read window after the nominal timepoint
    reagent_expiry_min: int | None = None  # drug stocks must be dispensed within this of release
    # Liquid-handler deck design (protocol/deck.py). When set, dispense time and tip use come
    # from the protocol planner on that deck instead of the fixed per-pass constant.
    deck: dict | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _plan(self, plate: PlateMap):
        key = plate.plate_id
        if key not in self._cache:
            from labloop.protocol.deck import DeckDesign
            from labloop.protocol.planner import plan_plate, protocol_time

            p = plan_plate(plate, DeckDesign(**self.deck))
            self._cache[key] = (math.ceil(protocol_time(p) / 60 + 1.0), dict(p.tips_used))
        return self._cache[key]

    def tips(self, plate: PlateMap) -> dict[str, int]:
        return self._plan(plate)[1] if self.deck else {}

    def cycles(self, plate: PlateMap) -> int:
        """Aspirate/dispense passes: per channel group, one cell-seeding pass plus one pass
        per distinct nonzero dose of each drug (different doses come from different sources)."""
        groups: dict[tuple[int, int], list] = {}
        for (r, c), content in plate.wells.items():
            groups.setdefault(plate.spec.channel_group(r, c), []).append(content)
        n = 0
        for contents in groups.values():
            n += 1
            conds = [x for x in contents if not isinstance(x, str)]
            if conds:
                for i in range(len(conds[0].doses)):
                    n += len({c.doses[i] for c in conds if c.doses[i] > 0})
        return n

    def dispense(self, plate: PlateMap) -> int:
        if self.deck:
            return self._plan(plate)[0]
        return math.ceil(self.lh_overhead_min + self.cycles(plate) * self.lh_cycle_min)

    def read(self, plate: PlateMap) -> int:
        return self.read_min_384 if plate.spec.n_wells > 96 else self.read_min_96

    def arm(self, route: str) -> int:
        return max(1, math.ceil(self.arm_move_min.get(route, 1.0)))

    def incubation_window(self, plate: PlateMap) -> tuple[int, int]:
        t = plate.timepoint_h * 60
        return t, t + self.incubation_tolerance_min


def build_durations(cfg: dict | None, use_motion_costs: bool = True) -> DurationModel:
    cfg = dict(cfg or {})
    if use_motion_costs and MOTION_COSTS.exists() and cfg.pop("motion_costs", True):
        measured = json.loads(MOTION_COSTS.read_text())
        cfg.setdefault("arm_move_min", measured["arm_move_min"])
        cfg.setdefault("lh_cycle_min", measured["lh_cycle_min"])
    cfg.pop("motion_costs", None)
    return DurationModel(**cfg)
