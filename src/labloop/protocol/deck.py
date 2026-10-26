"""Liquid-handler deck design: which labware sits where on a Hamilton STARlet, and which tips.

Geometry follows PyLabRobot's STARlet: 32 rails at 22.5 mm pitch starting at x = 100 mm; a
carrier is 6 rails wide with 5 sites at 96 mm pitch in y. The waste block runs front to back
at the right end (rail 31, 445 mm long), with the tip-eject slot along it, so carriers must end
before rail 31.
The plate arm can only reach carriers inside `ARM_REACH_RAILS` (the deck's right half, next
to the arm), which constrains where the assay plate and swappable tip racks may go.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

RAIL_PITCH = 22.5  # mm
RAIL0_X = 100.0
N_RAILS = 32
CARRIER_RAILS = 6
SITE_Y0, SITE_PITCH = 63.0 + 48.0, 96.0  # y of site 0 center, site pitch (mm)
WASTE_RAIL = 31  # PLR STARLetDeck waste block
WASTE_X, WASTE_Y = 800.0, (190.6, 431.8)  # tip-eject slot: x, front..back y (PLR trash)
HEAD_SPAN_MM = 63.0  # channel 1 to channel 8
ARM_REACH_RAILS = (12, 27)  # carrier start rails the plate arm can service


@dataclass(frozen=True)
class TipType:
    name: str
    capacity_ul: float
    dead_ul: float  # kept in the tip, never dispensed
    min_ul: float = 0.0  # smallest volume the lab has validated for this tip (data/kb lab_catalog)


TIPS = {"50uL": TipType("50uL", 50.0, 5.0, 0.5), "300uL": TipType("300uL", 300.0, 20.0, 10.0),
        "1000uL": TipType("1000uL", 1000.0, 50.0, 50.0)}
# Assay plates the lab stocks (384-well, tissue-culture treated); "clear" is the stockroom default.
ASSAY_PLATES = ("clear", "white", "black")


@dataclass(frozen=True)
class DeckDesign:
    tip_rail: int = 25  # tip carrier start rail (defaults = the optimized Level 2 deck)
    source_rail: int = 13  # cell reservoir + drug dilution plates
    dest_rail: int = 19  # assay plate carrier (arm-accessible)
    cell_tip: str = "1000uL"
    drug_tip: str = "50uL"
    racks_on_deck: int = 5  # tip racks on the tip carrier (1-5)
    cell_source: str = "trough"  # "trough" (one big reservoir) or "plate" (96-well reservoir)
    tip_reuse: bool = True  # aspirate once per reagent, dispense to many wells without contact
    assay_plate: str = "clear"  # plate wall (ASSAY_PLATES); timing does not depend on it
    # Pipetting per reagent (protocol/pipetting.py): "jet" drops from above the well, "surface" touches
    # it; reuse = one tip serves many wells of the same reagent (None: follow tip_reuse). drugs_first adds
    # the drugs to the empty plate before the cells. The defaults are the original protocol.
    cell_mode: str = "jet"
    drug_mode: str = "jet"
    cell_reuse: bool | None = None
    drug_reuse: bool | None = None
    drugs_first: bool = False
    meta: dict = field(default_factory=dict, hash=False, compare=False)

    def reuse(self, cells: bool) -> bool:
        r = self.cell_reuse if cells else self.drug_reuse
        return self.tip_reuse if r is None else r

    def mode(self, cells: bool) -> str:
        return self.cell_mode if cells else self.drug_mode

    def x_of(self, rail: int, col_offset_mm: float = 0.0) -> float:
        return RAIL0_X + (rail - 1) * RAIL_PITCH + col_offset_mm

    def valid(self) -> bool:
        spans = sorted([(self.tip_rail, "tip"), (self.source_rail, "src"), (self.dest_rail, "dst")])
        for (a, _), (b, _) in zip(spans, spans[1:]):
            if b - a < CARRIER_RAILS:
                return False
        if any(r < 1 or r + CARRIER_RAILS > WASTE_RAIL for r, _ in spans):
            return False
        lo, hi = ARM_REACH_RAILS
        # The arm loads the assay plate and swaps tip racks.
        return lo <= self.dest_rail <= hi and lo <= self.tip_rail <= hi and 1 <= self.racks_on_deck <= 5

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("meta")
        return d


def candidate_designs() -> list[DeckDesign]:
    """Every valid combination of carrier rails (step 1), tip types, racks and options."""
    out = []
    for tip_r, src_r, dst_r in itertools.product(range(1, N_RAILS - 4), repeat=3):
        for cell_tip, drug_tip in (("300uL", "50uL"), ("1000uL", "50uL"), ("300uL", "300uL"), ("50uL", "50uL")):
            for racks in (2, 3, 4, 5):
                for cell_source in ("trough", "plate"):
                    d = DeckDesign(tip_r, src_r, dst_r, cell_tip, drug_tip, racks, cell_source)
                    if d.valid():
                        out.append(d)
    return out


BASELINE = DeckDesign(tip_rail=15, source_rail=8, dest_rail=21, cell_tip="300uL", drug_tip="300uL",
                      racks_on_deck=5, cell_source="plate", tip_reuse=False)
