"""One re-placement of a misplaced plate, simulated in the MuJoCo cell: the lab's re-grasp, event by event.

The plate lies (dx, dy) mm off its nest at `station`. The deck camera estimates where it is (at the liquid
handler; elsewhere the true pose plus the camera's measured error stands in), the arm re-grips it there and sets
it on the nest (motion.replace_plan), and the cell's contacts decide where it ends up. scripts/sim3d_replace.py
measures the same thing over a grid of offsets; this runs it once, for the offset the lab drew.
"""

from __future__ import annotations

import math

import mujoco

from labloop.sim3d.cell import load
from labloop.sim3d.motion import playback, replace_plan, set_joints
from labloop.sim3d.perception import POS_TOL_MM, DeckCamera

OVERHEAD_S = {"reader": 12.0}  # the reader drawer opens and closes again


def regrasp(station: str, dx_mm: float, dy_mm: float, rng, est_err_mm: float) -> dict:
    """Simulated seconds, where the plate ends up relative to its nest (mm), and whether it is seated
    (within the camera tolerance, no collisions)."""
    spec, m, d = load(plates={"plate0": station})
    center = spec.stations[station].center
    set_joints(m, d, replace_plan(spec, "plate0", station, center)[0].target)  # the arm has just left, folded
    for _ in range(300):
        mujoco.mj_step(m, d)
    adr = m.jnt_qposadr[m.body("plate0").jntadr[0]]
    d.qpos[adr] = center[0] + dx_mm / 1000
    d.qpos[adr + 1] = center[1] + dy_mm / 1000
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    true = (float(d.qpos[adr]), float(d.qpos[adr + 1]))
    est = None
    if station == "lh":
        cam = DeckCamera(m)
        est = cam.estimate(d, "plate0_geom")
        cam.renderer.close()
    if est is None:
        e = rng.normal(0, est_err_mm / 1000, 2)
        est = (true[0] + float(e[0]), true[1] + float(e[1]))
    pb = playback(m, d, replace_plan(spec, "plate0", station, est[:2]), record=False)
    p = d.xpos[m.body("plate0").id]
    fx, fy = 1000 * (float(p[0]) - center[0]), 1000 * (float(p[1]) - center[1])
    err = math.hypot(fx, fy)
    return {"s": round(pb.duration + OVERHEAD_S.get(station, 0.0), 2), "est_err_mm": round(1000 * math.hypot(est[0] - true[0], est[1] - true[1]), 2),
            "final_dx_mm": round(fx, 2), "final_dy_mm": round(fy, 2), "final_mm": round(err, 2),
            "ok": bool(err <= POS_TOL_MM and not pb.collisions), "collisions": list(pb.collisions)}
