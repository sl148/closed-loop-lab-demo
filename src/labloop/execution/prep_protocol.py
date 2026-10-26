"""The liquid handler's dilution protocol for one batch, as transfers (shared by timing, checks and PyLabRobot).

Per drug, from its DMSO stock to a dilution plate column per dose (smallest dose in column 1):
- optional intermediate dilutions in the prep plate, each `factor`-fold: medium first, then the source;
- the top column, row by row (8 channels): medium, then the last intermediate (or the stock);
- the 3-fold series down the plate: medium in every lower column, then 1/3 carried column to column, mixed.

Every column ends with `ul_per_well` (the smallest keeps the last carry). Each transfer names its role
(fill, stock, intermediate, top, series), liquid and the tip and mode the plan gives that role, so
`check` can hold it to the Hamilton liquid class range (protocol/pipetting.py) and to well capacities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from labloop.protocol.pipetting import calibrated_range

WELL_MAX = {"dilution": 1200.0, "prep": 1200.0}  # SRC-DIL deep well; prep plate deep well
DEAD = 20.0  # left in a well the tip cannot reach
S_PER_TRANSFER = {1: 12, 8: 16}  # one or eight channels: pick, aspirate, dispense, eject
S_MIX = 6
ROLES = ("fill", "stock", "intermediate", "top", "series")
DEFAULT_PIPETTING = {r: {"tip": "300uL", "mode": "surface"} for r in ROLES}


@dataclass
class Transfer:
    drug: str
    role: str
    src: str  # "medium", "stock", "int1", ... or "col3"
    dst: str  # "int1", ..., "col7"
    ul: float  # per channel
    channels: int  # 8: one row each, in parallel; 1: a single prep-plate well
    liquid: str  # "dmso" | "medium" (what is in the tip)
    tip: str
    mode: str
    mix: bool = False
    repeats: int = 1  # a volume bigger than the tip holds goes in this many equal parts (ul is one part)


def _split(t: Transfer) -> Transfer:
    from labloop.protocol.deck import TIPS

    cap = TIPS[t.tip].capacity_ul - TIPS[t.tip].dead_ul
    n = max(1, math.ceil(t.ul / cap - 1e-9))
    t.repeats, t.ul = n, t.ul / n
    return t


def build(plan, drugs: list[str], tops: dict[str, float], n_doses: int) -> list[Transfer]:
    """Transfers for one batch of `plan` (execution/supplies.PrepPlan)."""
    out: list[Transfer] = []
    pip = {**DEFAULT_PIPETTING, **(getattr(plan, "pipetting", None) or {})}
    V = plan.ul_per_well
    carry = V / 2  # 3-fold: carry 1 part into 2 parts of medium
    for d in drugs:
        P = lambda role: (pip[role]["tip"], pip[role]["mode"])
        solvent = "dmso" if plan.series_solvent.get(d, "medium") == "dmso" else "medium"
        top = plan.top_working_uM(tops[d])
        stock = plan.stock_mM(d) * 1e3
        steps = list((getattr(plan, "dilution", None) or {}).get(d, []))  # intermediate factors, in order
        final = stock / top / math.prod(steps) if steps else stock / top  # last step into the top column
        top_need = V + carry  # per row
        x_top = top_need / final  # per row, from the last intermediate (or the stock)
        # intermediates, backwards: each must hold what the next step draws plus dead volume
        need = x_top * 8 + DEAD
        vols = []
        for f in reversed(steps):
            vols.append(need)
            need = need / f + DEAD
        vols.reverse()
        src = "stock"
        for j, (f, vol) in enumerate(zip(steps, vols), 1):
            out.append(Transfer(d, "fill", "medium", f"int{j}", vol - vol / f, 1, "medium", *P("fill")))
            out.append(Transfer(d, "stock" if src == "stock" else "intermediate", src, f"int{j}", vol / f, 1,
                                "dmso" if src == "stock" else solvent, *P("stock" if src == "stock" else "intermediate"), mix=True))
            src = f"int{j}"
        out.append(Transfer(d, "fill", "medium", f"col{n_doses}", top_need - x_top, 8, "medium", *P("fill")))
        out.append(Transfer(d, "stock" if src == "stock" else "top", src, f"col{n_doses}", x_top, 8,
                            "dmso" if src == "stock" else solvent, *P("stock" if src == "stock" else "top"), mix=True))
        for c in range(n_doses - 1, 0, -1):
            out.append(Transfer(d, "fill", "medium", f"col{c}", 2 * carry, 8, "medium", *P("fill")))
        for c in range(n_doses, 1, -1):
            out.append(Transfer(d, "series", f"col{c}", f"col{c - 1}", carry, 8, solvent, *P("series"), mix=True))
    return [_split(t) for t in out]


def check(transfers: list[Transfer], plan) -> list[str]:
    """What the lab knows is wrong with a protocol: volumes outside a liquid class, wells overfilled."""
    bad = []
    for t in transfers:
        r = calibrated_range(t.tip, t.liquid, t.mode)
        if r is None:
            bad.append(f"{t.drug} {t.role} {t.src}->{t.dst}: no liquid class for a {t.tip} tip, {t.liquid}, {t.mode}")
        elif not r[0] - 1e-9 <= t.ul <= r[1] + 1e-9:
            bad.append(f"{t.drug} {t.role} {t.src}->{t.dst}: {t.ul:.2f} ul outside {t.tip} {t.liquid} {t.mode} ({r[0]:g}-{r[1]:g} ul)")
    top = plan.ul_per_well * 1.5
    if top > WELL_MAX["dilution"]:
        bad.append(f"top column holds {top:g} ul, a dilution well holds {WELL_MAX['dilution']:g}")
    for t in transfers:
        if t.dst.startswith("int") and t.role == "fill":
            vol = t.ul * t.repeats + next(u.ul * u.repeats for u in transfers if u.drug == t.drug and u.dst == t.dst and u.role != "fill")
            if vol > WELL_MAX["prep"]:
                bad.append(f"{t.drug} {t.dst} holds {vol:.0f} ul, a prep well holds {WELL_MAX['prep']:g}")
    return bad


def out_of_range(transfers: list[Transfer]) -> set[str]:
    """Drugs whose dilution has a transfer outside its liquid class (the emulator's dose error)."""
    out = set()
    for t in transfers:
        r = calibrated_range(t.tip, t.liquid, t.mode)
        if r is None or not r[0] - 1e-9 <= t.ul <= r[1] + 1e-9:
            if t.role in ("stock", "intermediate", "top", "series"):
                out.add(t.drug)
    return out


def minutes(transfers: list[Transfer]) -> int:
    s = sum((S_PER_TRANSFER[t.channels] + (S_MIX if t.mix else 0)) * t.repeats for t in transfers)
    return max(1, math.ceil(s / 60))


def tips(transfers: list[Transfer]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in transfers:
        out[t.tip] = out.get(t.tip, 0) + t.channels * t.repeats
    return out
