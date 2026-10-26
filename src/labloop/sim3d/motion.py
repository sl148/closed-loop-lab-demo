"""Motion planning and playback in the MuJoCo cell.

Moves are synchronized joint-space trapezoids through teach points. Every transfer starts and
ends with the arm folded at transit height and rotates only while folded (link 1 sweeps 0.3 m
around the base, clear of the STARlet housing); stations inside the housing are entered and left
along a straight line through the open front, via a staging point in front of the header. Their durations are what the scheduler consumes; playback in MuJoCo checks
that the plan is physically consistent (plate ends on the target nest, no arm or carried-plate
collisions) and records poses for the web viewer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from labloop.sim3d import cell
from labloop.sim3d.cell import INSIDE, PLATE, CellSpec, _wrap, scara_ik

# (vmax, amax) per joint, PF400 / STARlet class
LIMITS = {"arm_z": (0.5, 1.5), "arm_shoulder": (math.radians(200), math.radians(500)),
          "arm_elbow": (math.radians(200), math.radians(500)), "arm_wrist": (math.radians(360), math.radians(900)),
          "lh_x": (0.8, 2.0), "lh_y": (0.5, 2.0), "lh_z": (0.3, 2.0)}
SAFE_Z = 0.72  # link plane height for transits: wrist bottom clears every device
APPROACH = 0.05
GRIP_S, RELEASE_S = 0.6, 0.4
STAGE_Y = -0.13  # gripper y in front of the housing header
FOLD = 2.2  # elbow angle (rad) while rotating: gripper 0.27 m from the base axis
CART_STEP = 0.08  # m between IK points on straight in/out moves
TIP_PICK_S, ASPIRATE_S, DISPENSE_S, EJECT_S = 1.2, 2.0, 2.0, 1.0


def trapezoid_time(dist: float, vmax: float, amax: float) -> float:
    dist = abs(dist)
    if dist < 1e-9:
        return 0.0
    if dist < vmax * vmax / amax:
        return 2 * math.sqrt(dist / amax)
    return dist / vmax + vmax / amax


def trapezoid_s(t: float, T: float) -> float:
    """Normalized progress 0..1 of a symmetric trapezoid (1/4 accel, 1/2 cruise, 1/4 decel)."""
    if T <= 0:
        return 1.0
    u = min(max(t / T, 0.0), 1.0)
    a, vmax = 1 / (0.25 * 0.75), 1 / 0.75  # area under the velocity profile = 1
    if u < 0.25:
        return 0.5 * a * u * u
    if u < 0.75:
        return 0.5 * a * 0.0625 + vmax * (u - 0.25)
    w = 1 - u
    return 1 - 0.5 * a * w * w


@dataclass
class Segment:
    target: dict[str, float]  # joint -> position
    action: str | None = None  # grasp:<plate> | release:<plate> | dwell
    dwell: float = 0.0
    label: str = ""


@dataclass
class Playback:
    duration: float
    collisions: list[str] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    final_plate_error_mm: float | None = None


def move_time(q0: dict[str, float], q1: dict[str, float]) -> float:
    return max((trapezoid_time(q1[j] - q0.get(j, q1[j]), *LIMITS[j]) for j in q1), default=0.0)


def transfer_plan(spec: CellSpec, plate: str, src: str, dst: str) -> list[Segment]:
    s, t = spec.stations[src], spec.stations[dst]
    grip_z_s = s.plate_z + PLATE[2] / 2
    grip_z_t = t.plate_z + PLATE[2] / 2
    sign = cell.ELBOW

    def at(st, z, xy=None):
        return scara_ik(xy or st.center, z, st.yaw, elbow=sign)

    def high(st, xy=None):
        return {**at(st, 0.0, xy), "arm_z": SAFE_Z}

    def folded(q, yaw):
        e = math.copysign(FOLD, q["arm_elbow"] or 1.0)
        return {"arm_z": SAFE_Z, "arm_shoulder": q["arm_shoulder"], "arm_elbow": e,
                "arm_wrist": _wrap(yaw - q["arm_shoulder"] - e)}

    def line(st, a, b, label):  # straight move at transit height between two gripper xy points
        n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / CART_STEP))
        return [Segment(high(st, (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)), label=label)
                for k in range(1, n + 1)]

    def stage(st):
        return (st.center[0], STAGE_Y)

    entry_s = high(s, stage(s)) if src in INSIDE else high(s)
    entry_t = high(t, stage(t)) if dst in INSIDE else high(t)
    segs = [Segment(folded(entry_s, s.yaw), label=f"to {src}"), Segment(entry_s, label=f"to {src}")]
    if src in INSIDE:
        segs += line(s, stage(s), s.center, f"to {src}")
    segs += [Segment(at(s, grip_z_s + APPROACH), label="approach"),
             Segment(at(s, grip_z_s), label="descend"),
             Segment(at(s, grip_z_s), action=f"grasp:{plate}", dwell=GRIP_S, label="grip"),
             Segment(high(s), label="lift")]
    if src in INSIDE:
        segs += line(s, s.center, stage(s), f"to {dst}")
    segs += [Segment(folded(entry_s, s.yaw), label=f"to {dst}"), Segment(folded(entry_t, t.yaw), label=f"to {dst}"),
             Segment(entry_t, label=f"to {dst}")]
    if dst in INSIDE:
        segs += line(t, stage(t), t.center, f"to {dst}")
    segs += [Segment(at(t, grip_z_t + APPROACH), label="approach"),
             Segment(at(t, grip_z_t + 0.001), label="place"),
             Segment(at(t, grip_z_t + 0.001), action=f"release:{plate}", dwell=RELEASE_S, label="release"),
             Segment(high(t), label="retract")]
    if dst in INSIDE:
        segs += line(t, t.center, stage(t), "retract")
    segs.append(Segment(folded(entry_t, t.yaw), label="retract"))
    # Take the short way around for the wrist (the plate only needs its yaw).
    w = segs[0].target["arm_wrist"]
    for sg in segs:
        sg.target = {**sg.target, "arm_wrist": w + _wrap(sg.target["arm_wrist"] - w)}
        w = sg.target["arm_wrist"]
    return segs


def replace_plan(spec: CellSpec, plate: str, station: str, est_xy: tuple[float, float]) -> list[Segment]:
    """Recover a misplaced plate: re-enter the station, grip it where the camera says it is,
    lift just clear of the nest, slide over the nominal center, set it down, and leave."""
    st = spec.stations[station]
    grip_z = st.plate_z + PLATE[2] / 2
    sign = cell.ELBOW

    def at(z, xy=None):
        return scara_ik(xy or st.center, z, st.yaw, elbow=sign)

    def high(xy=None):
        return {**at(0.0, xy), "arm_z": SAFE_Z}

    def folded(q):
        e = math.copysign(FOLD, q["arm_elbow"] or 1.0)
        return {"arm_z": SAFE_Z, "arm_shoulder": q["arm_shoulder"], "arm_elbow": e,
                "arm_wrist": _wrap(st.yaw - q["arm_shoulder"] - e)}

    inside = station in INSIDE
    stage = (st.center[0], STAGE_Y)
    entry = high(stage) if inside else high()
    segs = [Segment(folded(entry), label=f"to {station}"), Segment(entry, label=f"to {station}")]
    if inside:
        n = max(1, math.ceil(math.hypot(st.center[0] - stage[0], st.center[1] - stage[1]) / CART_STEP))
        segs += [Segment(high((stage[0] + (est_xy[0] - stage[0]) * k / n, stage[1] + (est_xy[1] - stage[1]) * k / n)),
                         label=f"to {station}") for k in range(1, n + 1)]
    segs += [Segment(at(grip_z + APPROACH, est_xy), label="approach"),
             Segment(at(grip_z, est_xy), label="descend"),
             Segment(at(grip_z, est_xy), action=f"grasp:{plate}", dwell=GRIP_S, label="grip"),
             Segment(at(grip_z + 0.006, est_xy), label="lift"),
             Segment(at(grip_z + 0.006), label="center"),
             Segment(at(grip_z + 0.001), label="place"),
             Segment(at(grip_z + 0.001), action=f"release:{plate}", dwell=RELEASE_S, label="release"),
             Segment(high(), label="retract")]
    if inside:
        segs.append(Segment(high(stage), label="retract"))
    segs.append(Segment(folded(entry), label="retract"))
    w = segs[0].target["arm_wrist"]
    for sg in segs:
        sg.target = {**sg.target, "arm_wrist": w + _wrap(sg.target["arm_wrist"] - w)}
        w = sg.target["arm_wrist"]
    return segs


def lh_pass_plan(spec: CellSpec) -> list[Segment]:
    """One 8-channel pass: pick tips, aspirate from a source column, dispense into the assay
    plate, eject tips. Positions from the PLR deck."""
    head_z = 0.42

    def over(site, depth):
        x, y, z = spec.lh_sites[site]
        return {"lh_x": x, "lh_y": y, "lh_z": (z + depth) - (head_z - 0.11)}

    seg = []
    for site, dwell, depth in (("tips", TIP_PICK_S, 0.0), ("source", ASPIRATE_S, -0.008),
                               ("dest", DISPENSE_S, -0.006), ("trash", EJECT_S, 0.05)):
        top = {**over(site, depth), "lh_z": 0.0}
        seg += [Segment(top, label=f"over {site}"), Segment(over(site, depth), label=f"down {site}"),
                Segment(over(site, depth), action="dwell", dwell=dwell, label=site), Segment(top, label="up")]
    return seg


def plan_duration(segments: list[Segment], q0: dict[str, float]) -> float:
    q, total = dict(q0), 0.0
    for seg in segments:
        total += move_time(q, seg.target) + seg.dwell
        q.update(seg.target)
    return total


def _set_weld(m, d, eq_name: str, active: bool) -> None:
    eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    if active:
        b1, b2 = m.eq_obj1id[eid], m.eq_obj2id[eid]
        # Anchor the weld at the current relative pose so the plate does not jump.
        p1, q1 = d.xpos[b1], d.xquat[b1]
        p2, q2 = d.xpos[b2], d.xquat[b2]
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, q1)
        rel_p = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_p, p2 - p1, q1inv)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, q1inv, q2)
        m.eq_data[eid, 0:3] = 0.0
        m.eq_data[eid, 3:6] = rel_p
        m.eq_data[eid, 6:10] = rel_q
        m.eq_data[eid, 10] = 1.0
    d.eq_active[eid] = 1 if active else 0


def joint_q(m, d) -> dict[str, float]:
    return {m.joint(i).name: float(d.qpos[m.jnt_qposadr[i]]) for i in range(m.njnt)
            if m.jnt_type[i] in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)}


def set_joints(m, d, q: dict[str, float]) -> None:
    for j, v in q.items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        d.ctrl[m.actuator(f"act_{j}").id] = v
    mujoco.mj_forward(m, d)


COLLIDERS = ("link1", "link2", "wrist")
OBSTACLE_PREFIX = ("dev_", "plr_", "obstacle", "lh_", "frame_")  # nests are expected contacts at pick/place


def _collisions(m, d, carried: set[str]) -> list[str]:
    """Arm links touching anything, or a carried plate touching a device in transit."""
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
        for a, b in ((n1, n2), (n2, n1)):
            if a in COLLIDERS and b not in COLLIDERS:
                out.append(f"{a}x{b}")
            if a.removesuffix("_geom") in carried and b.startswith(OBSTACLE_PREFIX):
                out.append(f"{a}x{b}")
    return out


def playback(m, d, segments: list[Segment], fps: float = 15.0, record: bool = True,
             bodies: tuple[str, ...] = ()) -> Playback:
    """Simulate the plan with position actuators following the trapezoid references."""
    dt = m.opt.timestep
    q = joint_q(m, d)
    carried: set[str] = set()
    pb = Playback(0.0)
    next_frame, t_total = 0.0, 0.0
    names = bodies or tuple(m.body(i).name for i in range(1, m.nbody))

    def frame():
        pb.frames.append({"t": round(t_total, 3), "q": {k: round(v, 5) for k, v in joint_q(m, d).items()},
                          "bodies": {n: [round(float(x), 4) for x in (*d.xpos[m.body(n).id], *d.xquat[m.body(n).id])]
                                     for n in names}})

    for seg in segments:
        T = move_time(q, seg.target)
        start = dict(q)
        steps = int(math.ceil((T + seg.dwell) / dt)) if (T + seg.dwell) > 0 else 0
        if seg.action and seg.action.startswith("grasp:"):
            plate = seg.action.split(":")[1]
            _set_weld(m, d, f"carry_{plate}", True)
            carried.add(plate)
        for k in range(steps):
            s = trapezoid_s(k * dt, T)
            for j, v in seg.target.items():
                d.ctrl[m.actuator(f"act_{j}").id] = start.get(j, v) + (v - start.get(j, v)) * s
            mujoco.mj_step(m, d)
            t_total += dt
            # Carried plates touch their supports at pick/place; check them only in transit.
            for c in _collisions(m, d, carried if seg.label.startswith("to ") else set()):
                if c not in pb.collisions:
                    pb.collisions.append(c)
            if record and t_total >= next_frame:
                frame()
                next_frame += 1 / fps
        if seg.action and seg.action.startswith("release:"):
            plate = seg.action.split(":")[1]
            _set_weld(m, d, f"carry_{plate}", False)
            carried.discard(plate)
        q.update(seg.target)
    # Settle and record the final state.
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
        t_total += dt
    if record:
        frame()
    pb.duration = t_total
    return pb
