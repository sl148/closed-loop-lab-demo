"""Render one re-placement of a misplaced plate in the MuJoCo cell (the trial scripts/sim3d_replace.py measures).

    MUJOCO_GL=egl PYTHONPATH=src python scripts/sim3d_replace_video.py [--offset-mm 5] [--station lh]

The plate sits off its nest at the liquid handler; the deck camera estimates where it is; the arm re-grips it
there and sets it on the nest (motion.replace_plan), simulated with the cell's contacts. Two views side by side,
an angled close-up and a top-down camera over the nest, with the step and the time. Writes
results/sim3d/replace.gif (and replace.mp4 when imageio-ffmpeg is installed).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from labloop.sim3d.cell import load
from labloop.sim3d.motion import joint_q, move_time, playback, replace_plan, set_joints
from labloop.sim3d.perception import DeckCamera

OUT = Path("results/sim3d")
W, H, FPS = 480, 360, 15
STEP = {"to": "re-entering the station", "approach": "over the plate, where the camera saw it", "descend": "down to grip height",
        "grip": "gripping the plate where it lies", "lift": "lifting it just clear of the nest", "center": "sliding it over the nest",
        "place": "setting it down", "release": "releasing", "retract": "leaving"}


def segment_starts(m, d, plan) -> list[tuple[float, str]]:
    """When each step starts, timed as motion.playback runs them."""
    q, t, out = joint_q(m, d), 0.0, []
    for seg in plan:
        out.append((t, seg.label.split()[0]))
        t += move_time(q, seg.target) + seg.dwell
        q.update(seg.target)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset-mm", type=float, default=5.0)
    ap.add_argument("--dir-deg", type=float, default=35.0)
    ap.add_argument("--station", default="lh")
    args = ap.parse_args()
    spec, m, d = load(plates={"plate0": args.station})
    center = spec.stations[args.station].center
    plan0 = replace_plan(spec, "plate0", args.station, center)
    set_joints(m, d, plan0[0].target)
    for _ in range(300):
        mujoco.mj_step(m, d)
    adr = m.jnt_qposadr[m.body("plate0").jntadr[0]]
    a = math.radians(args.dir_deg)
    d.qpos[adr] = center[0] + args.offset_mm / 1000 * math.cos(a)
    d.qpos[adr + 1] = center[1] + args.offset_mm / 1000 * math.sin(a)
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    est = DeckCamera(m).estimate(d, "plate0_geom")
    plan = replace_plan(spec, "plate0", args.station, est[:2])
    starts = segment_starts(m, d, plan)
    start_q = joint_q(m, d)
    start_plate = d.qpos[adr:adr + 7].copy()
    pb = playback(m, d, plan, fps=FPS, record=True, bodies=("plate0",))
    p = d.xpos[m.body("plate0").id]
    final_mm = 1000 * math.hypot(p[0] - center[0], p[1] - center[1])
    est_err = 1000 * math.hypot(est[0] - start_plate[0], est[1] - start_plate[1])

    # replay the recorded frames through two cameras
    rd, r = mujoco.MjData(m), mujoco.Renderer(m, H, W)
    z = spec.stations[args.station].plate_z
    near = mujoco.MjvCamera()
    near.type, near.lookat[:], near.distance, near.azimuth, near.elevation = mujoco.mjtCamera.mjCAMERA_FREE, [center[0], center[1], z], 0.8, 50, -38
    top = mujoco.MjvCamera()
    top.type, top.lookat[:], top.distance, top.azimuth, top.elevation = mujoco.mjtCamera.mjCAMERA_FREE, [center[0], center[1], z], 0.62, 90, -89.5
    from labloop.sim3d.cell import PLATE

    def nest(scene):  # the nest's outline (where the plate belongs), drawn into the scene only
        hx, hy, zz = PLATE[0] / 2, PLATE[1] / 2, z + PLATE[2] + 0.002
        for (x, y, sx, sy) in ((center[0], center[1] + hy, hx, 0.0008), (center[0], center[1] - hy, hx, 0.0008),
                               (center[0] + hx, center[1], 0.0008, hy), (center[0] - hx, center[1], 0.0008, hy)):
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([sx, sy, 0.0006]), np.array([x, y, zz]), np.eye(3).flatten(),
                                np.array([0.0, 0.8, 0.85, 1.0], dtype=np.float32))
            scene.ngeom += 1
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()

    def shot(q, plate, label, t):
        for j, v in q.items():
            rd.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        rd.qpos[adr:adr + 7] = plate
        mujoco.mj_forward(m, rd)
        views = []
        for cam in (near, top):
            r.update_scene(rd, camera=cam)
            nest(r.scene)
            views.append(r.render())
        off = 1000 * math.hypot(plate[0] - center[0], plate[1] - center[1])
        img = Image.fromarray(np.concatenate(views, axis=1))
        g = ImageDraw.Draw(img)
        g.rectangle([0, 0, 2 * W, 26], fill=(255, 255, 255))
        g.text((10, 5), label, fill=(20, 20, 20), font=font)
        g.text((2 * W - 70, 5), f"{t:4.1f} s", fill=(90, 90, 90), font=font)
        g.rectangle([0, H - 22, 2 * W, H], fill=(255, 255, 255))
        g.text((10, H - 19), f"MuJoCo cell · plate {args.offset_mm:g} mm off its nest · camera estimate {est_err:.1f} mm off", fill=(70, 70, 70), font=small)
        g.text((W + 10, H - 19), "top-down · cyan: the nest", fill=(70, 70, 70), font=small)
        g.rectangle([2 * W - 230, 32, 2 * W - 8, 56], fill=(255, 255, 255))
        g.text((2 * W - 222, 36), f"plate off the nest: {off:4.1f} mm", fill=(200, 40, 40) if off > 1.5 else (20, 140, 60), font=font)
        return img

    frames = [shot(start_q, start_plate, f"camera: plate {args.offset_mm:g} mm off its nest, flagged", 0.0)] * (FPS + FPS // 2)
    for f in pb.frames:
        step = [s for s in starts if s[0] <= f["t"]][-1][1]
        frames.append(shot(f["q"], f["bodies"]["plate0"], STEP.get(step, step), f["t"]))
    last = pb.frames[-1]
    frames += [shot(last["q"], last["bodies"]["plate0"], f"seated: {final_mm:.1f} mm from the nest center (tolerance 1.5 mm)", last["t"])] * (2 * FPS)
    OUT.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT / "replace.gif", save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0)
    print(f"wrote {OUT / 'replace.gif'}: {len(frames)} frames, motion {pb.duration:.1f} s, final error {final_mm:.2f} mm, "
          f"collisions {pb.collisions or 'none'}")
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(OUT / "replace.mp4", fps=FPS) as w:
            for im in frames:
                w.append_data(np.asarray(im))
        print(f"wrote {OUT / 'replace.mp4'}")
    except Exception as e:  # imageio-ffmpeg not installed: the GIF is enough
        print("no mp4:", type(e).__name__)


if __name__ == "__main__":
    main()
