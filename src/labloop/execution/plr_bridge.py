"""PyLabRobot bridge: executes a plate map as a real pipetting protocol on a simulated
Hamilton STARlet (chatterbox backend) with tip and volume tracking.

PLR is the reference for deck state and commands: this verifies that a layout is executable
(every well gets cells + the right drug volumes, nothing over- or under-drawn), counts the
8-channel passes the protocol actually needs (cross-check for `DurationModel.cycles`), and
exposes the deck geometry the 3D layer (sim3d) is built from.

Deck (rails): tip carrier @1 (5 racks), source carrier @8 (cells reservoir plate + one
dilution plate per drug, dose k of drug i in column k so the 8 channels draw 8 rows), destination
carrier @15 (assay plate).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.hamilton.STAR_chatterbox import STARChatterboxBackend
from pylabrobot.resources import (PLT_CAR_L5AC_A00, STARLetDeck, TIP_CAR_480_A00, Greiner_384_wellplate_28ul_Fb,
                                  hamilton_1_trough_200mL_Vb,
                                  cor_96_wellplate_360uL_Fb, hamilton_96_tiprack_50uL, set_tip_tracking,
                                  set_volume_tracking)

from labloop.types import PlateMap

logging.getLogger("pylabrobot").setLevel(logging.ERROR)

VOLUMES = {96: (40.0, 5.0), 384: (16.0, 2.0)}  # (cells uL, per-drug uL)
TROUGH_UL = 150_000.0  # cell suspension loaded into the 200 mL trough


@dataclass
class ProtocolReport:
    plate_id: str
    passes: int
    tips_used: int
    wells_ok: int
    wells_bad: list[str]
    refills: int = 0
    rack_swaps: int = 0
    # One record per 8-channel pass when tracing (for the web liquid-handler view):
    # [what (0 = cells, i + 1 = drug i), dose uM, source column, destination well indices,
    #  uL per channel, first tip index on the carrier, rack swaps so far,
    #  mean uL left in the source wells after aspirating (from the PLR volume tracker)]
    trace: list | None = None

    @property
    def ok(self) -> bool:
        return not self.wells_bad


class PLRBridge:
    def __init__(self, n_drugs: int, ladders: tuple[tuple[float, ...], ...] | None = None, trace: bool = False):
        """`ladders` fixes the dose -> source column map; without it the map is derived per
        plate from the doses it uses."""
        self.n_drugs, self.ladders, self.trace = n_drugs, ladders, trace

    def _build(self, n_wells: int):
        set_tip_tracking(True)
        set_volume_tracking(True)
        deck = STARLetDeck()
        lh = LiquidHandler(backend=STARChatterboxBackend(), deck=deck)
        tip_car = TIP_CAR_480_A00(name="tip_carrier")
        for i in range(5):
            tip_car[i] = hamilton_96_tiprack_50uL(name=f"tips_{i}")
        deck.assign_child_resource(tip_car, rails=1)
        src_car = PLT_CAR_L5AC_A00(name="source_carrier")
        src_car[0] = cells = cor_96_wellplate_360uL_Fb(name="cells")
        sources = []
        for i in range(self.n_drugs):
            src_car[i + 1] = p = cor_96_wellplate_360uL_Fb(name=f"drug{i}")
            sources.append(p)
        deck.assign_child_resource(src_car, rails=8)
        dst_car = PLT_CAR_L5AC_A00(name="dest_carrier")
        dst_car[0] = dst = (Greiner_384_wellplate_28ul_Fb if n_wells > 96 else cor_96_wellplate_360uL_Fb)(name="assay")
        deck.assign_child_resource(dst_car, rails=15)
        for w in cells.get_all_items():
            w.tracker.set_volume(300.0)
        for p in sources:
            for w in p.get_all_items():
                w.tracker.set_volume(300.0)
        return lh, deck, cells, sources, dst

    async def _run(self, pm: PlateMap) -> ProtocolReport:
        lh, deck, cells, sources, dst = self._build(pm.spec.n_wells)
        ladders = self.ladders or tuple(
            (0.0, *sorted({c.doses[i] for c in pm.conditions() if c.doses[i] > 0})) for i in range(self.n_drugs))
        await lh.setup()
        cells_ul, drug_ul = VOLUMES[384 if pm.spec.n_wells > 96 else 96]
        tips = [t for i in range(5) for t in deck.get_resource(f"tips_{i}").get_all_items()]
        tip_i = 0
        tips_total = 0
        passes = 0
        refills = 0
        swaps = 0
        cell_col = 1

        def cell_sources(n: int, vol: float):
            """Rotate through reservoir columns; refill the plate when it runs dry."""
            nonlocal cell_col, refills
            for _ in range(13):
                ws = [cells.get_well(f"{chr(65 + k)}{cell_col}") for k in range(n)]
                if all(w.tracker.get_used_volume() >= vol for w in ws):
                    return ws
                cell_col = cell_col % 12 + 1
                if cell_col == 1:
                    refills += 1
                    for w in cells.get_all_items():
                        w.tracker.set_volume(300.0)
            raise RuntimeError("cell reservoir cannot supply this pass")

        trace: list = []
        cols = pm.spec.cols

        async def transfer(sources_wells, targets, vol, what=0, dose=0.0, col=0, dst_idx=()):
            nonlocal tip_i, passes, swaps, tips_total
            n = len(targets)
            if tip_i + n > len(tips):  # carrier exhausted: swap in fresh racks
                for i in range(5):
                    deck.get_resource(f"tips_{i}").fill()
                tip_i, swaps = 0, swaps + 1
            use = tips[tip_i: tip_i + n]
            tip_i += n
            tips_total += n
            ch = list(range(n))
            await lh.pick_up_tips(use, use_channels=ch)
            await lh.aspirate(sources_wells, vols=[vol] * n, use_channels=ch)
            if self.trace:
                left = sum(w.tracker.get_used_volume() for w in sources_wells) / n
                trace.append([what, dose, col, list(dst_idx), vol, tip_i - n, swaps, round(left, 2)])
            await lh.dispense(targets, vols=[vol] * n, use_channels=ch)
            await lh.discard_tips(use_channels=ch)
            passes += 1

        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for (r, c), content in pm.wells.items():
            if content == "EMPTY":
                continue
            groups.setdefault(pm.spec.channel_group(r, c), []).append((r, c))
        for key in sorted(groups):
            wells = sorted(groups[key])[:8]
            targets = [dst.get_well(pm.spec.well_name(r, c)) for r, c in wells]
            srcs = cell_sources(len(wells), cells_ul)
            await transfer(srcs, targets, cells_ul, 0, 0.0, cell_col, [r * cols + c for r, c in wells])
            for i in range(self.n_drugs):
                by_dose: dict[float, list] = {}
                for r, c in wells:
                    content = pm.wells[(r, c)]
                    if not isinstance(content, str) and content.doses[i] > 0:
                        by_dose.setdefault(content.doses[i], []).append((r, c))
                for dose, ws in sorted(by_dose.items()):
                    col = ladders[i].index(dose)
                    srcs = [sources[i].get_well(f"{chr(65 + k)}{col}") for k in range(len(ws))]
                    await transfer(srcs, [dst.get_well(pm.spec.well_name(r, c)) for r, c in ws], drug_ul,
                                   i + 1, dose, col, [r * cols + c for r, c in ws])

        bad = []
        for (r, c), content in pm.wells.items():
            if content == "EMPTY":
                continue
            n_drugs = 0 if isinstance(content, str) else len(content.active)
            expected = cells_ul + n_drugs * drug_ul
            got = dst.get_well(pm.spec.well_name(r, c)).tracker.get_used_volume()
            if abs(got - expected) > 1e-6:
                bad.append(pm.spec.well_name(r, c))
        await lh.stop()
        n_ok = sum(1 for v in pm.wells.values() if v != "EMPTY") - len(bad)
        return ProtocolReport(pm.plate_id, passes, tips_total, n_ok, bad, refills, swaps,
                              trace if self.trace else None)

    def run(self, pm: PlateMap) -> ProtocolReport:
        return asyncio.run(self._run(pm))

    # ------------------------------------------------------------------ planned protocols
    def run_protocol(self, pm: PlateMap, proto) -> ProtocolReport:
        """Execute a planned protocol (protocol/planner.py: tip reuse, multi-dispense, deck
        design) with tip and volume tracking."""
        return asyncio.run(self._run_protocol(pm, proto))

    async def _run_protocol(self, pm: PlateMap, proto) -> ProtocolReport:
        from pylabrobot.resources import hamilton_96_tiprack_1000uL, hamilton_96_tiprack_300uL

        from labloop.protocol.optimizer import rack_split

        design = proto.design
        set_tip_tracking(True)
        set_volume_tracking(True)
        deck = STARLetDeck()
        lh = LiquidHandler(backend=STARChatterboxBackend(), deck=deck)
        make = {"50uL": hamilton_96_tiprack_50uL, "300uL": hamilton_96_tiprack_300uL,
                "1000uL": hamilton_96_tiprack_1000uL}
        split = rack_split(proto.tips_used, design.racks_on_deck)
        tip_car = TIP_CAR_480_A00(name="tip_carrier")
        racks: dict[str, list] = {}
        slot = 0
        for t, n in split.items():
            for _ in range(n):
                tip_car[slot] = make[t](name=f"tips_{t}_{slot}")
                racks.setdefault(t, []).append(tip_car[slot].resource)
                slot += 1
        deck.assign_child_resource(tip_car, rails=design.tip_rail)
        src_car = PLT_CAR_L5AC_A00(name="source_carrier")
        trough = design.cell_source == "trough"  # one 200 mL reservoir (118 mm long): every channel draws from it
        src_car[0] = cells = (hamilton_1_trough_200mL_Vb if trough else cor_96_wellplate_360uL_Fb)(name="cells")
        sources = []
        for i in range(self.n_drugs):
            src_car[i + 1] = p = cor_96_wellplate_360uL_Fb(name=f"drug{i}")
            sources.append(p)
        deck.assign_child_resource(src_car, rails=design.source_rail)
        dst_car = PLT_CAR_L5AC_A00(name="dest_carrier")
        dst_car[0] = dst = (Greiner_384_wellplate_28ul_Fb if pm.spec.n_wells > 96 else cor_96_wellplate_360uL_Fb)(name="assay")
        deck.assign_child_resource(dst_car, rails=design.dest_rail)
        for p in [cells, *sources]:
            for w in ([p] if p is cells and trough else p.get_all_items()):
                w.tracker.set_volume(TROUGH_UL if p is cells and trough else 350.0)
        await lh.setup()
        from labloop.protocol.deck import TIPS

        tip_idx = {t: 0 for t in racks}
        swaps = refills = 0
        trace, cols = [], pm.spec.cols
        names = lambda ws: [dst.get_well(pm.spec.well_name(w // cols, w % cols)) for w in ws]

        def next_tips(t: str, n: int):
            nonlocal swaps
            pool = [tip for r in racks[t] for tip in r.get_all_items()]
            if tip_idx[t] + n > len(pool):  # the arm swaps in full racks from the rack hotel
                for r in racks[t]:
                    r.fill()
                tip_idx[t], swaps = 0, swaps + 1
            out = pool[tip_idx[t]: tip_idx[t] + n]
            start = tip_idx[t]
            tip_idx[t] += n
            return out, start

        for ps in proto.passes:
            what = 0 if ps.reagent[0] == "cells" else ps.reagent[1] + 1
            chans = sorted(ps.channels)
            n = len(chans)
            tips, start = next_tips(ps.tip_type, n)
            await lh.pick_up_tips(tips, use_channels=list(range(n)))
            src_plate = cells if what == 0 else sources[what - 1]
            col = ps.source_col
            per_asp = max(1, int((TIPS[ps.tip_type].capacity_ul - TIPS[ps.tip_type].dead_ul) // ps.vol_ul)) \
                if proto.design.tip_reuse else 1
            columns = sorted({w % cols for ws in ps.channels.values() for w in ws})
            left = None
            for a in range(ps.aspirations):
                batch = columns[a * per_asp:(a + 1) * per_asp] if proto.design.tip_reuse else columns
                vols = [ps.vol_ul * sum(1 for w in ps.channels[k] if w % cols in batch) for k in chans]
                if what == 0 and trough:
                    well = cells
                    if well.tracker.get_used_volume() < sum(vols) + 1:
                        refills += 1
                        well.tracker.set_volume(TROUGH_UL)
                    await lh.aspirate([well] * n, vols=vols, use_channels=list(range(n)), spread="tight")
                    if left is None:  # as a 350 uL-well equivalent, like the plate reservoir
                        left = 350.0 * well.tracker.get_used_volume() / TROUGH_UL
                elif what == 0:  # cells: rotate reservoir columns, refill when dry
                    while any(src_plate.get_well(f"{chr(65 + i)}{col}").tracker.get_used_volume() < v + 1
                              for i, v in enumerate(vols)):
                        col = col % 12 + 1
                        if col == 1:
                            refills += 1
                            for w in src_plate.get_all_items():
                                w.tracker.set_volume(350.0)
                if not (what == 0 and trough):
                    srcs = [src_plate.get_well(f"{chr(65 + i)}{col}") for i in range(n)]
                    await lh.aspirate(srcs, vols=vols, use_channels=list(range(n)))
                    if left is None:
                        left = sum(w.tracker.get_used_volume() for w in srcs) / n
                for c in batch:
                    use = [i for i, k in enumerate(chans) if any(w % cols == c for w in ps.channels[k])]
                    wells = [next(w for w in ps.channels[chans[i]] if w % cols == c) for i in use]
                    await lh.dispense(names(wells), vols=[ps.vol_ul] * len(use), use_channels=use)
            await lh.discard_tips(use_channels=list(range(n)))
            trace.append([what, ps.reagent[2] if what else 0.0, col, sorted(w for ws in ps.channels.values() for w in ws),
                          ps.vol_ul, start, swaps, round(left or 0.0, 2), ps.aspirations,
                          [[k, ps.channels[k]] for k in chans], ps.tip_type])
        bad = []
        for (r, c), content in pm.wells.items():
            if content == "EMPTY":
                continue
            n_dr = 0 if isinstance(content, str) else len(content.active)
            expected = VOLUMES_PLAN[384 if pm.spec.n_wells > 96 else 96][0] + n_dr * VOLUMES_PLAN[384 if pm.spec.n_wells > 96 else 96][1]
            if abs(dst.get_well(pm.spec.well_name(r, c)).tracker.get_used_volume() - expected) > 1e-6:
                bad.append(pm.spec.well_name(r, c))
        await lh.stop()
        n_ok = sum(1 for v in pm.wells.values() if v != "EMPTY") - len(bad)
        return ProtocolReport(pm.plate_id, len(proto.passes), sum(proto.tips_used.values()), n_ok, bad, refills, swaps, trace)


VOLUMES_PLAN = {96: (40.0, 5.0), 384: (16.0, 2.0)}  # planner volumes (protocol/planner.py)


def deck_geometry() -> dict:
    """Absolute positions (mm) and sizes of the bridge deck's resources, for sim3d."""
    bridge = PLRBridge(2, ((0.0, 1.0), (0.0, 1.0)))
    _, deck, *_ = bridge._build(96)
    out = {"deck": {"size": [deck.get_size_x(), deck.get_size_y(), deck.get_size_z()]}, "resources": {}}
    for res in deck.get_all_children():
        if res.category in ("plate", "tip_rack", "plate_carrier", "tip_carrier", "trash"):
            loc = res.get_absolute_location()
            out["resources"][res.name] = {"category": res.category, "origin": [loc.x, loc.y, loc.z],
                                          "size": [res.get_size_x(), res.get_size_y(), res.get_size_z()]}
    return out
