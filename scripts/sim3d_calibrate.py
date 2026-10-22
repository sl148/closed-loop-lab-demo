"""Calibrate the scheduler's duration and failure models from the MuJoCo cell (M7).

    uv run python scripts/sim3d_calibrate.py [--install]

Writes results/sim3d/:
  motion_costs.json   arm route minutes (MuJoCo motion + device overheads) and LH pass minutes
  tips.json           tip pickup: offset sweep + p_fail(sigma), simulated vs analytic
  perception.json     camera misplacement detection rate vs offset
  cell_geometry.json  static + moving geoms for the web 3D viewer
  episode.json        body poses of one plate's full journey (hotel -> LH -> incubator ->
                      reader -> hotel, with 3 pipetting passes) for the web viewer
  cell.gif            the same episode rendered from the overview camera
--install copies motion_costs.json to configs/, where DurationModel picks it up.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from labloop.sim3d.cell import load, scara_ik
from labloop.sim3d.motion import (SAFE_Z, joint_q, lh_pass_plan, playback, plan_duration, set_joints,
                                  transfer_plan)
from labloop.sim3d.perception import detection_curve
from labloop.sim3d.tips import offset_sweep, p_fail

OUT = Path("results/sim3d")
# Device handling around a transfer (s): incubator door + shelf shuttle, reader drawer, hotel.
OVERHEAD_S = {"hotel": 3.0, "lh": 0.0, "incubator": 40.0, "reader": 12.0, "rackhotel": 3.0, "tiprack": 0.0,
              "rackwaste": 0.0}
# Scheduler route name -> (MuJoCo source station, target station). Tip boxes move like plates.
ROUTES = {"hotel>lh": ("hotel", "lh"), "lh>incubator": ("lh", "incubator"), "incubator>reader": ("incubator", "reader"),
          "reader>hotel": ("reader", "hotel"), "lh>rackwaste": ("tiprack", "rackwaste"),
          "rackhotel>lh": ("rackhotel", "tiprack")}
MOVING = ("plate0", "lh_gantry", "arm_carriage", "arm_link1", "arm_link2", "arm_wrist")


def geometry(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    """Geoms with their pose in the body frame (for moving bodies) and in the world at rest."""
    kinds = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
             mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_PLANE: "plane"}
    geoms = []
    for i in range(m.ngeom):
        kind = kinds.get(int(m.geom_type[i]))
        if kind is None:
            continue
        geoms.append({"name": m.geom(i).name, "type": kind, "body": m.body(m.geom_bodyid[i]).name,
                      "size": [round(float(x), 5) for x in m.geom_size[i]],
                      "pos": [round(float(x), 5) for x in m.geom_pos[i]],
                      "quat": [round(float(x), 5) for x in m.geom_quat[i]],
                      "rgba": [round(float(x), 3) for x in m.geom_rgba[i]],
                      "xpos": [round(float(x), 5) for x in d.geom_xpos[i]],
                      "xmat": [round(float(x), 5) for x in d.geom_xmat[i]]})
    return {"geoms": geoms, "moving": list(MOVING)}


def episode(spec, m, d) -> tuple[list[dict], dict]:
    set_joints(m, d, {**scara_ik(spec.stations["hotel"].center, 0.4), "arm_z": SAFE_Z,
                      "lh_x": spec.lh_sites["tips"][0], "lh_y": spec.lh_sites["tips"][1], "lh_z": 0.0})
    for _ in range(300):
        mujoco.mj_step(m, d)
    frames, t0, timing, collisions = [], 0.0, {}, []

    def run(segs, label):
        nonlocal t0
        pb = playback(m, d, segs, fps=12, bodies=MOVING)
        for f in pb.frames:
            frames.append({"t": round(t0 + f["t"], 3), "phase": label, "q": f["q"], "bodies": f["bodies"]})
        t0 += pb.duration
        timing[label] = round(pb.duration, 2)
        collisions.extend(pb.collisions)

    run(transfer_plan(spec, "plate0", "hotel", "lh"), "hotel>lh")
    for k in range(3):
        run(lh_pass_plan(spec), f"pipetting pass {k + 1}")
    for src, dst in (("lh", "incubator"), ("incubator", "reader"), ("reader", "hotel")):
        run(transfer_plan(spec, "plate0", src, dst), f"{src}>{dst}")
    return frames, {"timing_s": timing, "collisions": collisions}


def render_gif(m, frames: list[dict], path: Path, stride: int = 2) -> None:
    from PIL import Image

    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 360, 480)
    images = []
    for f in frames[::stride]:
        adr = m.jnt_qposadr[m.body("plate0").jntadr[0]]
        d.qpos[adr:adr + 7] = f["bodies"]["plate0"]
        for j, v in f["q"].items():
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="overview")
        images.append(Image.fromarray(r.render()))
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // 12 * stride, loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--tip-samples", type=int, default=300)
    ap.add_argument("--episode-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.episode_only:
        calibrate(args)
    spec3, m3, d3 = load(plates={"plate0": "hotel"})
    mujoco.mj_forward(m3, d3)
    (OUT / "cell_geometry.json").write_text(json.dumps(geometry(m3, d3)))
    frames, meta = episode(spec3, m3, d3)
    (OUT / "episode.json").write_text(json.dumps({"meta": meta, "frames": frames}))
    print("episode:", meta, len(frames), "frames")
    render_gif(m3, frames, OUT / "cell.gif")
    print("wrote", OUT / "cell.gif")
    if args.install:
        shutil.copy(OUT / "motion_costs.json", Path("configs") / "motion_costs.json")
        print("installed configs/motion_costs.json")


def calibrate(args) -> None:

    spec, m, d = load(plates={"plate0": "hotel"})
    set_joints(m, d, {**scara_ik(spec.stations["hotel"].center, 0.4), "arm_z": SAFE_Z})
    for _ in range(300):
        mujoco.mj_step(m, d)
    routes, checks = {}, {}
    at = "hotel"
    for route, (src, dst) in ROUTES.items():
        if src != at:  # tip boxes: start a fresh cell with the box (plate-sized body) at its source
            spec, m, d = load(plates={"plate0": src})
            set_joints(m, d, {**scara_ik(spec.stations[src].center, 0.4), "arm_z": SAFE_Z})
            for _ in range(300):
                mujoco.mj_step(m, d)
        at = dst
        plan = transfer_plan(spec, "plate0", src, dst)
        planned = plan_duration(plan, joint_q(m, d))
        pb = playback(m, d, plan, record=False)
        p = d.xpos[m.body("plate0").id]
        err = 1000 * float(np.hypot(p[0] - spec.stations[dst].center[0], p[1] - spec.stations[dst].center[1]))
        total = pb.duration + OVERHEAD_S[src] + OVERHEAD_S[dst]
        routes[route] = round(total / 60, 3)
        checks[route] = {"planned_s": round(planned, 2), "simulated_s": round(pb.duration, 2),
                                  "overhead_s": OVERHEAD_S[src] + OVERHEAD_S[dst], "placement_err_mm": round(err, 2),
                                  "collisions": pb.collisions}
    lh_pass_s = plan_duration(lh_pass_plan(spec), joint_q(m, d))
    costs = {"arm_move_min": routes, "lh_cycle_min": round(lh_pass_s / 60, 3), "checks": checks,
             "lh_pass_s": round(lh_pass_s, 2), "overhead_s": OVERHEAD_S}
    (OUT / "motion_costs.json").write_text(json.dumps(costs, indent=1))
    print("motion costs:", json.dumps({k: costs[k] for k in ("arm_move_min", "lh_cycle_min")}))

    tips = {"offset_sweep": offset_sweep(),
            "p_fail": [p_fail(s, n=args.tip_samples) for s in (0.25, 0.3, 0.4, 0.5)]}
    (OUT / "tips.json").write_text(json.dumps(tips, indent=1))
    print("tips:", tips["p_fail"])

    spec2, m2, d2 = load(plates={"plate0": "lh"})
    for _ in range(300):
        mujoco.mj_step(m2, d2)
    curve = detection_curve(m2, d2, "plate0", spec2.stations["lh"].center,
                            offsets_mm=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0), n_dirs=16)
    curve += detection_curve(m2, d2, "plate0", spec2.stations["lh"].center, offsets_mm=(0.0,), n_dirs=8,
                             yaw_deg=2.0)
    (OUT / "perception.json").write_text(json.dumps(curve, indent=1))
    print("perception:", [(c["offset_mm"], c["yaw_deg"], c["detect_rate"]) for c in curve])



if __name__ == "__main__":
    main()
