import json
from pathlib import Path

import yaml

from labloop.design.base import build_space
from labloop.execution.plr_bridge import PLRBridge
from labloop.layout.layouts import build_layout
from labloop.loop import run_campaign
from labloop.protocol.deck import BASELINE, DeckDesign
from labloop.protocol.planner import plan_plate, protocol_time
from labloop.types import PlateSpec

SPACE = build_space({"cell_lines": ["L1"], "n_drugs": 2, "dose": {"max": 10, "steps": 7, "fold": 3}, "timepoints_h": [48]})


def plate():
    return build_layout("optimized", PlateSpec(), {}, 0).place(SPACE.all_conditions[:50])[0]


def test_reuse_cuts_passes_and_time():
    pm = plate()
    fresh, reuse = plan_plate(pm, BASELINE), plan_plate(pm, DeckDesign())
    assert reuse.n_passes < fresh.n_passes
    assert protocol_time(reuse) < protocol_time(fresh)
    # Every destination well receives cells exactly once in both plans.
    for p in (fresh, reuse):
        cells = [w for ps in p.passes if ps.reagent == ("cells",) for ws in ps.channels.values() for w in ws]
        assert len(cells) == len(set(cells)) == sum(1 for v in pm.wells.values() if v != "EMPTY")


def test_planned_protocol_volumes_verified_by_plr():
    pm = plate()
    rep = PLRBridge(2).run_protocol(pm, plan_plate(pm, DeckDesign()))
    assert rep.ok and rep.passes == len(rep.trace)


def test_deck_design_validity():
    assert DeckDesign().valid()
    assert not DeckDesign(tip_rail=1, source_rail=3, dest_rail=15).valid()  # carriers overlap
    assert not DeckDesign(dest_rail=2).valid()  # assay plate out of the arm's reach
    assert not DeckDesign(tip_rail=27, source_rail=13, dest_rail=19).valid()  # tip carrier over the waste block


def test_tips_are_consumed_and_resupplied_by_the_arm():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
    deck = dict(DeckDesign().to_dict(), racks_on_deck=2)
    run = run_campaign(dict(cfg, rounds=2, deck=deck), 0)
    kinds = [e["kind"] for e in run["events"]]
    assert "tip_order" in kinds and "tip_refill" in kinds
    assert any(r["kind"] == "tip_swap" for r in run["schedule"])


def test_arm_and_gantry_never_share_the_deck_zone():
    """Load -> dispense -> unload holds the LH zone; no other plate or tip-rack swap enters it."""
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
    deck = dict(DeckDesign().to_dict(), racks_on_deck=2)
    run = run_campaign(dict(cfg, rounds=2, deck=deck), 0)
    rows = {r["op"]: r for r in run["schedule"]}
    zones = {}
    for oid, r in rows.items():
        if r["kind"] not in ("dispense", "tip_swap"):
            continue
        if r["kind"] == "tip_swap" and not oid.endswith(".tip_swap"):
            continue  # CO-RE box moves run inside their job's zone hold
        job = r["job"]
        first, last = (f"{job}.mv_hotel_lh", f"{job}.mv_lh_incubator") if r["kind"] == "dispense" else \
            (min((x for x in rows.values() if x["job"] == job), key=lambda x: x["start"])["op"], oid)
        zones.setdefault(r["resource"], []).append((rows[first]["start"], rows[last]["end"], job))
    assert zones
    for spans in zones.values():
        spans.sort()
        for (_, e, a), (s, _, b) in zip(spans, spans[1:]):
            assert s >= e, (a, b)


def test_tip_boxes_are_swapped_at_their_own_sites():
    """Each used box leaves from its own tip-carrier site and the new one goes back there: the arm
    directly for the front sites, through the LH CO-RE gripper and the handoff site for deep ones."""
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
    run = run_campaign(dict(cfg, rounds=3, deck=DeckDesign().to_dict()), 0)
    rows = [r for r in run["schedule"] if r["job"].startswith("TIPS")]
    assert rows
    expect = {True: ["mv_lh_rackwaste", "mv_rackhotel_lh"],
              False: ["core_out", "mv_handoff_rackwaste", "mv_rackhotel_handoff", "core_in"]}
    sites_seen = set()
    for job in {r["job"] for r in rows}:
        seq = [r["op"].split(".")[1] for r in sorted((r for r in rows if r["job"] == job), key=lambda r: r["start"])]
        assert seq[-1] == "tip_swap"
        by_site = {}
        for name in seq[:-1]:
            base, site = name.rsplit("_s", 1)
            by_site.setdefault(int(site), []).append(base)
        for site, moves in by_site.items():
            assert moves == expect[site < 2], (site, moves)
        sites_seen |= set(by_site)
    assert len(sites_seen) > 1  # boxes are replaced where they are, not at one location


def test_full_plate_draws_cells_from_the_trough_with_all_channels():
    """Every channel draws cells from one trough at once; it must fit the 8-channel head."""
    pm = build_layout("optimized", PlateSpec(), {}, 0).place(SPACE.all_conditions)[0]
    rep = PLRBridge(2).run_protocol(pm, plan_plate(pm, DeckDesign()))
    assert rep.ok and rep.passes == len(rep.trace)
