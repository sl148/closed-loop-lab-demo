"""Per-plate pipetting protocol planner for an 8-channel head.

A protocol is a list of passes. A pass serves one reagent (cells, or one drug at one dose)
for one row group: pick tips -> aspirate -> dispense -> eject.
- Channel k addresses row k of a 96-well plate, or row 2k + parity of a 384-well plate
  (channels are 9 mm apart, 384-well rows are 4.5 mm apart).
- With tip reuse (non-contact dispensing), a channel aspirates once for several wells of the
  same reagent and dispenses column by column; it re-aspirates when the tip's usable volume
  runs out. Without reuse, a pass dispenses one column, then ejects (the PLR bridge baseline).

The planner emits waypoints with zones (tips / source / dest / waste) and offsets; the deck
design only sets where the zones are. So one plan is timed on many decks cheaply
(`protocol_time`), which is what the deck optimizer needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from labloop.protocol.deck import HEAD_SPAN_MM, SITE_PITCH, SITE_Y0, TIPS, WASTE_X, WASTE_Y, DeckDesign
from labloop.sim3d.motion import trapezoid_time
from labloop.types import Condition, PlateMap

# Head kinematics (MuJoCo gantry limits, m/s and m/s^2) and per-action dwell (s).
VX, AX, VY, AY, VZ, AZ = 0.8, 2.0, 0.5, 2.0, 0.3, 2.0
Z_DIP_M = 0.10  # traverse height to labware
DWELL = {"pick": 1.5, "aspirate": 1.8, "dispense_jet": 0.5, "dispense_contact": 1.5, "eject": 1.2}
WELL_PITCH = {96: 9.0, 384: 4.5}
CELLS_UL = {96: 40.0, 384: 16.0}
DRUG_UL = {96: 5.0, 384: 2.0}


@dataclass
class Waypoint:
    zone: str  # tips | source | dest | waste
    dx_mm: float  # offset inside the zone's carrier
    dy_mm: float
    action: str
    dwell_s: float
    dip: bool = True


@dataclass
class Pass:
    reagent: tuple  # ("cells",) or ("drug", i, dose)
    channels: dict[int, list[int]]  # channel -> destination well indices, in dispense order
    vol_ul: float
    tip_type: str
    aspirations: int
    source_col: int
    waypoints: list[Waypoint] = field(default_factory=list)
    mode: str = "jet"  # protocol/pipetting.py
    reuse: bool = True


@dataclass
class Protocol:
    plate_id: str
    passes: list[Pass]
    tips_used: dict[str, int]
    design: DeckDesign

    @property
    def n_passes(self) -> int:
        return len(self.passes)


def _lanes(pm: PlateMap) -> dict[tuple[int, int], list[int]]:
    """(parity, channel) -> rows; 96-well: parity 0 only."""
    if pm.spec.rows <= 8:
        return {(0, r): [r] for r in range(pm.spec.rows)}
    return {(p, k): [2 * k + p] for p in (0, 1) for k in range(pm.spec.rows // 2)}


def plan_plate(pm: PlateMap, design: DeckDesign, ladders: tuple[tuple[float, ...], ...] | None = None) -> Protocol:
    n_wells = pm.spec.n_wells
    size = 384 if n_wells > 96 else 96
    pitch = WELL_PITCH[size]
    rows, cols = pm.spec.rows, pm.spec.cols
    conds = pm.conditions()
    n_drugs = len(conds[0].doses) if conds else 0
    ladders = ladders or tuple((0.0, *sorted({c.doses[i] for c in conds if c.doses[i] > 0})) for i in range(n_drugs))
    # reagent -> well indices
    reagents: dict[tuple, list[int]] = {("cells",): []}
    for (r, c), content in sorted(pm.wells.items()):
        if content == "EMPTY":
            continue
        idx = r * cols + c
        reagents[("cells",)].append(idx)
        if isinstance(content, Condition):
            for i, d in enumerate(content.doses):
                if d > 0:
                    reagents.setdefault(("drug", i, d), []).append(idx)
    passes: list[Pass] = []
    tips_used: dict[str, int] = {}
    lanes = _lanes(pm)
    parities = sorted({p for p, _ in lanes})
    order = list(reagents.items())
    if design.drugs_first:  # drugs into the empty plate, cells on top
        order = order[1:] + order[:1]
    for reagent, wells in order:
        is_cells = reagent[0] == "cells"
        vol = CELLS_UL[size] if is_cells else DRUG_UL[size]
        tip = TIPS[design.cell_tip if is_cells else design.drug_tip]
        reuse, mode = design.reuse(is_cells), design.mode(is_cells)
        per_asp = max(1, int((tip.capacity_ul - tip.dead_ul) // vol)) if reuse else 1
        src_col = 1 if is_cells else ladders[reagent[1]].index(reagent[2])
        for parity in parities:
            lane_wells: dict[int, list[int]] = {}
            for (p, k), lane_rows in lanes.items():
                if p != parity:
                    continue
                ws = sorted((w for w in wells if w // cols in lane_rows), key=lambda w: w % cols)
                if ws:
                    lane_wells[k] = ws
            if not lane_wells:
                continue
            if reuse:
                chunks = [lane_wells]
            else:  # one column per pass, fresh tips each time
                columns = sorted({w % cols for ws in lane_wells.values() for w in ws})
                chunks = [{k: [w for w in ws if w % cols == c] for k, ws in lane_wells.items()
                           if any(w % cols == c for w in ws)} for c in columns]
            for chunk in chunks:
                n_disp = max(len(ws) for ws in chunk.values())
                asps = math.ceil(n_disp / per_asp)
                ps = Pass(reagent, chunk, vol, tip.name, asps, src_col, mode=mode, reuse=reuse)
                tips_used[tip.name] = tips_used.get(tip.name, 0) + len(chunk)
                _waypoints(ps, design, pitch, cols, rows, per_asp, parity, size)
                passes.append(ps)
    return Protocol(pm.plate_id, passes, tips_used, design)


def _waypoints(ps: Pass, design: DeckDesign, pitch: float, cols: int, rows: int, per_asp: int,
               parity: int, size: int) -> None:
    wp = ps.waypoints
    wp.append(Waypoint("tips", 0.0, 0.0, "pick", DWELL["pick"]))
    src_dx = (ps.source_col - 1) * 9.0
    jet = ps.mode == "jet"
    # Column visit order: all channels dispense at a column together (aligned rows).
    columns = sorted({w % cols for ws in ps.channels.values() for w in ws})
    done = 0
    for a in range(ps.aspirations):
        wp.append(Waypoint("source", src_dx, 0.0, "aspirate", DWELL["aspirate"]))
        batch = columns[a * per_asp:(a + 1) * per_asp] if ps.reuse else columns
        for c in batch:
            wp.append(Waypoint("dest", c * pitch, parity * pitch, "dispense",
                               DWELL["dispense_jet"] if jet else DWELL["dispense_contact"], dip=not jet))
        done += len(batch)
    wp.append(Waypoint("waste", 0.0, 0.0, "eject", DWELL["eject"]))


def waste_xy(y_from: float) -> tuple[float, float]:
    """Eject point on the waste slot: the slot runs front to back, so the head ejects at the
    point nearest to where it comes from (channel 1 y; the head spans 63 mm)."""
    return WASTE_X, min(max(y_from, WASTE_Y[0]), WASTE_Y[1] - HEAD_SPAN_MM)


def zone_xy(design: DeckDesign, zone: str, rack: int = 0) -> tuple[float, float]:
    """Deck coordinates (mm) of a zone's reference point (the waste: its slot center)."""
    if zone == "waste":
        return waste_xy(sum(WASTE_Y) / 2)
    rail = {"tips": design.tip_rail, "source": design.source_rail, "dest": design.dest_rail}[zone]
    site = {"tips": rack, "source": 0, "dest": 0}[zone]
    return design.x_of(rail, 20.0), SITE_Y0 + site * SITE_PITCH


def move_s(p, q) -> float:
    dx, dy = abs(q[0] - p[0]) / 1000, abs(q[1] - p[1]) / 1000
    return max(trapezoid_time(dx, VX, AX), trapezoid_time(dy, VY, AY))


DIP_S = 2 * trapezoid_time(Z_DIP_M, VZ, AZ)


def protocol_time(proto: Protocol, design: DeckDesign | None = None) -> float:
    """Seconds to run the protocol on `design` (defaults to the one it was planned for)."""
    design = design or proto.design
    t, pos = 0.0, zone_xy(design, "waste")
    tip_count = 0
    for ps in proto.passes:
        rack = min(4, tip_count // 96 % max(1, design.racks_on_deck))
        for w in ps.waypoints:
            x0, y0 = waste_xy(pos[1]) if w.zone == "waste" else zone_xy(design, w.zone, rack)
            q = (x0 + w.dx_mm, y0 + w.dy_mm)
            t += move_s(pos, q) + w.dwell_s + (DIP_S if w.dip else DIP_S * 0.35)
            pos = q
        tip_count += len(ps.channels)
    return t
