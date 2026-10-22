"""Camera perception: an overhead deck camera renders a segmentation image; each plate's
pixel mask gives its centroid and orientation (second moments), back-projected to deck
coordinates with the pinhole model at the known plate-top height. Comparing against the
planned layout flags misplaced plates; `detection_curve` measures how detection depends on
the misplacement size.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from labloop.sim3d.cell import PLATE

W, H = 1280, 960
POS_TOL_MM, YAW_TOL_DEG = 1.5, 1.0


class DeckCamera:
    def __init__(self, m: mujoco.MjModel, camera: str = "deckcam"):
        self.m, self.cam = m, m.camera(camera)
        self.renderer = mujoco.Renderer(m, H, W)
        self.renderer.enable_segmentation_rendering()
        self.f = (H / 2) / math.tan(math.radians(m.cam_fovy[self.cam.id]) / 2)

    def estimate(self, d: mujoco.MjData, plate_geom: str) -> tuple[float, float, float] | None:
        """Plate (x, y, yaw) in world coordinates, or None if not visible."""
        self.renderer.update_scene(d, camera=self.cam.id)
        seg = self.renderer.render()
        gid = self.m.geom(plate_geom).id
        mask = (seg[..., 0] == gid) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        if mask.sum() < 50:
            return None
        v, u = np.nonzero(mask)
        uc, vc = u.mean(), v.mean()
        cov = np.cov(np.stack([u - uc, v - vc]))
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, np.argmax(evals)]
        # Image u -> +x, v -> -y for this downward camera; yaw of the long axis mod pi.
        yaw = math.atan2(-major[1], major[0])
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        cam_pos = d.cam_xpos[self.cam.id]
        top_z = d.geom_xpos[gid][2] + PLATE[2] / 2
        depth = cam_pos[2] - top_z
        x = cam_pos[0] + (uc - (W - 1) / 2) * depth / self.f
        y = cam_pos[1] - (vc - (H - 1) / 2) * depth / self.f
        return float(x), float(y), float(yaw)


def misplaced(est, planned_xy, planned_yaw=0.0) -> tuple[bool, float, float]:
    if est is None:
        return True, float("inf"), float("inf")
    err = 1000 * math.hypot(est[0] - planned_xy[0], est[1] - planned_xy[1])
    yaw_err = abs(math.degrees((est[2] - planned_yaw + math.pi / 2) % math.pi - math.pi / 2))
    return err > POS_TOL_MM or yaw_err > YAW_TOL_DEG, err, yaw_err


def detection_curve(m, d, plate_body: str, planned_xy, offsets_mm=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
                    n_dirs: int = 8, yaw_deg: float = 0.0) -> list[dict]:
    """Shift the plate by each offset in `n_dirs` directions (plus optional yaw), render, and
    record detection rate and estimation error."""
    cam = DeckCamera(m)
    jid = m.body(plate_body).jntadr[0]
    adr = m.jnt_qposadr[jid]
    base = d.qpos[adr:adr + 7].copy()
    out = []
    for off in offsets_mm:
        hits, errs = 0, []
        for k in range(n_dirs):
            a = 2 * math.pi * k / n_dirs
            d.qpos[adr:adr + 7] = base
            d.qpos[adr] = planned_xy[0] + off / 1000 * math.cos(a)
            d.qpos[adr + 1] = planned_xy[1] + off / 1000 * math.sin(a)
            th = math.radians(yaw_deg) * (1 if k % 2 == 0 else -1)
            d.qpos[adr + 3:adr + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]
            mujoco.mj_forward(m, d)
            est = cam.estimate(d, f"{plate_body}_geom")
            flag, err, _ = misplaced(est, planned_xy)
            hits += flag
            true = (d.qpos[adr], d.qpos[adr + 1])
            errs.append(1000 * math.hypot(est[0] - true[0], est[1] - true[1]) if est else float("nan"))
        out.append({"offset_mm": off, "yaw_deg": yaw_deg, "detect_rate": hits / n_dirs,
                    "est_err_mm": round(float(np.nanmean(errs)), 3)})
    d.qpos[adr:adr + 7] = base
    mujoco.mj_forward(m, d)
    return out
