"""Which volumes a tip can pipette, from the Hamilton STAR liquid classes that PyLabRobot ships.

A liquid class is Hamilton's calibration for one tip size, liquid and dispense mode; its correction
curve is calibrated over a volume range, and outside it the head delivers less, and less evenly, than
asked. `calibrated_range` reads that range from PyLabRobot 0.2.2
(pylabrobot.liquid_handling.liquid_classes.hamilton.star.star_mapping): standard volume, filtered
tips on the 8-channel head (not CO-RE 96), the widest range over the blow-out variants.

    jet      non-contact: the drop leaves the tip above the well; one aspiration can serve many wells
    surface  the tip touches the well (or the liquid in it) to release the drop; small volumes
"""

from __future__ import annotations

from functools import lru_cache

LIQUIDS = ("medium", "dmso")  # culture medium and buffers pipette like water
MODES = ("jet", "surface")


@lru_cache(maxsize=None)
def _table() -> dict[tuple[int, str, str], tuple[float, float]]:
    from pylabrobot.liquid_handling.liquid_classes.hamilton.star import star_mapping
    from pylabrobot.resources.liquid import Liquid

    liq = {Liquid.WATER: "medium", Liquid.DMSO: "dmso"}
    out: dict[tuple[int, str, str], tuple[float, float]] = {}
    for (tip_ul, core, is_tip, has_filter, liquid, jet, _blow), lc in star_mapping.items():
        if core or not is_tip or not has_filter or liquid not in liq:
            continue
        pts = sorted(v for v in lc.curve if v > 0)
        if not pts:
            continue
        key = (int(tip_ul), liq[liquid], "jet" if jet else "surface")
        lo, hi = out.get(key, (pts[0], min(pts[-1], tip_ul)))
        out[key] = (min(lo, pts[0]), max(hi, min(pts[-1], tip_ul)))
    return out


def calibrated_range(tip: str, liquid: str, mode: str) -> tuple[float, float] | None:
    """(min, max) ul for a tip type such as "50uL"; None if Hamilton has no liquid class for it."""
    return _table().get((int(tip.rstrip("uL")), liquid, mode))


def in_range(vol_ul: float, tip: str, liquid: str, mode: str) -> bool:
    r = calibrated_range(tip, liquid, mode)
    return r is not None and r[0] - 1e-9 <= vol_ul <= r[1] + 1e-9


def table_text(tips=("50uL", "300uL", "1000uL")) -> list[str]:
    """Lines for the lab catalog: each tip's calibrated range per liquid and mode."""
    lines = []
    for t in tips:
        parts = []
        for liq in LIQUIDS:
            for mode in MODES:
                r = calibrated_range(t, liq, mode)
                if r:
                    parts.append(f"{liq} {mode} {r[0]:g} to {r[1]:g} ul")
        lines.append(f"{t.replace('uL', ' ul')} tip: " + "; ".join(parts) + ".")
    return lines
