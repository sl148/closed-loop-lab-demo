"""Measure tier-1 recovery of a misplaced plate in the MuJoCo cell (M9).

    MUJOCO_GL=egl uv run python scripts/sim3d_replace.py

For each station and misplacement (offset x direction): the plate sits off its nest; the deck
camera estimates its pose (at the LH; elsewhere the true pose stands in, with the camera's
measured error added); the arm re-grips it at the estimate and sets it on the nominal center
(motion.replace_plan). Records time, final placement error and collisions. Writes
results/sim3d/replace.json, which `apply_sim3d` turns into the re-placement time and failure
rate the lab uses instead of a hand-set constant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from labloop.sim3d.cell import load
from labloop.sim3d.motion import joint_q, playback, plan_duration, replace_plan, set_joints
from labloop.sim3d.perception import POS_TOL_MM, DeckCamera

OUT = Path("results/sim3d")
OFFSETS_MM = (1.5, 2.0, 3.0, 5.0)
N_DIRS = 8
OVERHEAD_S = {"lh": 0.0, "reader": 12.0}  # reader drawer opens and closes again


def trial(station: str, off_mm: float, k: int, rng: np.random.Generator, est_err_mm: float) -> dict:
    spec, m, d = load(plates={"plate0": station})
    center = spec.stations[station].center
    plan0 = replace_plan(spec, "plate0", station, center)
    set_joints(m, d, plan0[0].target)  # the arm has just left the station, folded
    for _ in range(300):
        mujoco.mj_step(m, d)
    adr = m.jnt_qposadr[m.body("plate0").jntadr[0]]
    a = 2 * math.pi * k / N_DIRS
    d.qpos[adr] = center[0] + off_mm / 1000 * math.cos(a)
    d.qpos[adr + 1] = center[1] + off_mm / 1000 * math.sin(a)
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    true = (float(d.qpos[adr]), float(d.qpos[adr + 1]))
    est = DeckCamera(m).estimate(d, "plate0_geom") if station == "lh" else None
    if est is None:  # no camera over this station in the model: true pose + the deck camera's error
        e = rng.normal(0, est_err_mm / 1000, 2)
        est = (true[0] + e[0], true[1] + e[1], 0.0)
    plan = replace_plan(spec, "plate0", station, est[:2])
    planned = plan_duration(plan, joint_q(m, d))
    pb = playback(m, d, plan, record=False)
    p = d.xpos[m.body("plate0").id]
    err = 1000 * math.hypot(p[0] - center[0], p[1] - center[1])
    return {"station": station, "offset_mm": off_mm, "dir": k, "est_err_mm": round(1000 * math.hypot(est[0] - true[0], est[1] - true[1]), 3),
            "planned_s": round(planned, 2), "simulated_s": round(pb.duration + OVERHEAD_S[station], 2),
            "final_err_mm": round(err, 3), "ok": bool(err <= POS_TOL_MM and not pb.collisions), "collisions": pb.collisions}


def main() -> None:
    rng = np.random.default_rng(0)
    perc = json.loads((OUT / "perception.json").read_text())
    est_err = float(np.nanmean([r["est_err_mm"] for r in perc if r["offset_mm"] > 0]))
    rows = [trial(st, off, k, rng, est_err) for st in OVERHEAD_S for off in OFFSETS_MM for k in range(N_DIRS)]
    summary = {}
    for st in OVERHEAD_S:
        r = [x for x in rows if x["station"] == st]
        summary[st] = {"trials": len(r), "mean_s": round(float(np.mean([x["simulated_s"] for x in r])), 2),
                       "success_rate": round(float(np.mean([x["ok"] for x in r])), 3),
                       "mean_final_err_mm": round(float(np.mean([x["final_err_mm"] for x in r])), 3),
                       "collisions": sorted({c for x in r for c in x["collisions"]})}
    out = {"summary": summary, "trials": rows, "pos_tol_mm": POS_TOL_MM, "camera_err_mm": round(est_err, 3)}
    (OUT / "replace.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
