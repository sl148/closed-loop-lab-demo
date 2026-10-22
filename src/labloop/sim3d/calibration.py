"""Feed the MuJoCo calibration (scripts/sim3d_calibrate.py -> results/sim3d/) into the
lab: arm and pipetting durations from simulated motion, tip-pickup failure probability from
the contact model at the configured positioning error, and the camera's misplacement
detection curve.

    sim3d: {dir: results/sim3d, positioning_sigma_mm: 0.4, misplace_sigma_mm: 1.5, perception: true}
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from labloop.execution.durations import DurationModel


def apply_sim3d(cfg: dict, durations: DurationModel, failures):
    d = Path(cfg.get("dir", "results/sim3d"))
    motion = json.loads((d / "motion_costs.json").read_text())
    durations = dataclasses.replace(durations, arm_move_min=motion["arm_move_min"], lh_cycle_min=motion["lh_cycle_min"])
    tips = json.loads((d / "tips.json").read_text())["p_fail"]
    sig = [r["sigma_mm"] for r in tips]
    p_tip = float(np.interp(cfg.get("positioning_sigma_mm", 0.4), sig, [r["p_fail_sim"] for r in tips]))
    curve = ()
    if cfg.get("perception", True):
        rows = [r for r in json.loads((d / "perception.json").read_text()) if r["yaw_deg"] == 0]
        curve = tuple((r["offset_mm"], r["detect_rate"]) for r in sorted(rows, key=lambda r: r["offset_mm"]))
    else:
        curve = ((0.0, 1.0), (100.0, 1.0))
    failures = dataclasses.replace(failures, p_tip_pickup=p_tip, perception_curve=curve,
                                   misplace_sigma_mm=cfg.get("misplace_sigma_mm", failures.misplace_sigma_mm))
    return durations, failures
