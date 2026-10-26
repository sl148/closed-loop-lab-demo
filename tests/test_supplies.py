"""Liquid supplies (execution/supplies.py): prep arithmetic, hidden prep limits, and runs that make their own liquids."""

from pathlib import Path

import yaml

from labloop.execution.supplies import DRUGS, PrepPlan
from labloop.execution.truth import Truth
from labloop.loop import run_campaign

ROOT = Path(__file__).parents[1]
LEVEL2 = yaml.safe_load((ROOT / "configs" / "level2.yaml").read_text())
DRUG3 = LEVEL2["oracle"]["drugs"]
TOPS = {"MK-4827": 10.0, "SORAFENIB": 20.0, "VINBLASTINE": 0.04}
CFG = dict(LEVEL2, rounds=3, scheduler="repair", scheduler_cfg={"deterministic": True}, landscape_eval=False,
           deck=str(ROOT / "configs" / "deck_level4.json"), inventory=str(ROOT / "configs" / "inventory_level4.yaml"))


def test_prep_arithmetic():
    p = PrepPlan.default(DRUG3)
    assert abs(p.stock_mM("SORAFENIB") - 10.0) < 0.05  # 1 mg in DMSO to 10 mM
    assert abs(p.top_working_uM(20.0) - 240.0) < 1e-9  # 2 ul into a 24 ul well
    assert 0 < p.dmso_final_pct(TOPS) < 0.5
    assert abs(p.stock_ul_per_batch("SORAFENIB", 20.0) - 8 * 1.5 * 300 * 240 / (p.stock_mM("SORAFENIB") * 1e3)) < 1e-6  # 8 rows


def test_prep_effects_are_hidden_limits():
    t = Truth(stock_solubility=True, dmso_max_pct=2.0)
    good = PrepPlan.default(DRUG3)
    assert t.prep_effects(good, DRUG3, TOPS) == {}
    over = PrepPlan({**good.stocks, "VINBLASTINE": {"mg": 1.0, "dmso_ul": 20}}, good.series_solvent)  # 50 mg/ml
    fx = t.prep_effects(over, DRUG3, TOPS)
    assert abs(fx["drug_scale"][2] - DRUGS["VINBLASTINE"][1] / 50.0) < 1e-9 and fx["drug_scale"][0] == 1.0
    dilute = PrepPlan({d: {"mg": 0.01, "dmso_ul": 1000} for d in DRUG3}, {d: "dmso" for d in DRUG3})  # ~0.02 mM stocks
    assert "dmso" in t.prep_effects(dilute, DRUG3, TOPS)


def test_lab_makes_its_own_liquids_and_stale_batches_lose_potency():
    truth = {"drug_rt_stable_min": 1440, "drug_halflife_min": 720}
    stale = run_campaign(dict(CFG, prep="default", truth=truth), 0)  # one batch, never remade
    fresh = run_campaign(dict(CFG, prep={**PrepPlan.default(DRUG3).to_dict(), "max_age_min": 1200}, truth=truth), 0)
    for r in (stale, fresh):
        assert "pending" not in r and len([e for e in r["events"] if e["kind"] == "read"]) == len(r["plates"])
        kinds = {e["kind"] for e in r["events"]}
        assert {"prep", "cell_swap"} <= kinds
        assert r["supplies"]["inventory_left"]["SRC-DIL plates"] < 60
    assert stale["supplies"]["batches"] == 1 and fresh["supplies"]["batches"] > 1
    assert stale["truth"].get("drug_rt", 0) > 0 and fresh["truth"].get("drug_rt", 0) == 0
    lines = [e["line"] for e in stale["events"] if e["kind"] == "cell_swap"]
    assert len(set(lines)) == len(LEVEL2["space"]["cell_lines"])  # one reservoir per cell line, swapped in


def test_batch_scales_up_for_a_plate_the_plan_cannot_serve():
    from types import SimpleNamespace

    from labloop.execution.supplies import DEAD_UL, DRUG_UL, Supplies

    plan = PrepPlan.from_dict({**PrepPlan.default(DRUG3).to_dict(), "ul_per_well": 60.0})
    S = Supplies(plan, DRUG3, [[1.0, 2.0], [1.0, 2.0], [0.01, 0.02]], {})
    dose = SimpleNamespace(doses=(2.0, 0.0, 0.0))  # one dose of drug 0 in all 48 wells of channel 0
    pm = SimpleNamespace(wells={(r, c): dose for r in (0, 1) for c in range(24)}, spec=SimpleNamespace(rows=16),
                         cell_line="X")
    assert S.need(pm)["drug"][(0, 2.0)] == 48 * DRUG_UL > plan.ul_per_well - DEAD_UL
    S.make_batch("lh1", 0, TOPS, "PREP001")
    assert not S.drug_ok("lh1", pm, 1)  # the plan's batch can never serve it: the lab used to reorder forever
    S.make_batch("lh1", 10, TOPS, "PREP002", S.batch_ul(pm))
    assert S.drug_ok("lh1", pm, 11)
