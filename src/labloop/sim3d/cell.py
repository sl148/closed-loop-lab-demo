"""MuJoCo workcell: Hamilton STARlet deck (PLR coordinates), an 8-channel pipetting gantry,
a PF400-style 4-axis SCARA plate mover, the off-deck stations (hotel, incubator, reader, tip-box
hotel and tip-box waste bin) and the STARlet housing (side walls, back, roof, front header; the
front below the header is open for the arm). Fidelity stops at device motion: no liquids, no
torque-level control.

Units: meters (PLR mm / 1000). Deck origin = PLR deck origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from labloop.execution.plr_bridge import deck_geometry
from labloop.protocol.deck import CARRIER_RAILS, RAIL_PITCH, SITE_PITCH, SITE_Y0, DeckDesign

PLATE = (0.12776, 0.08548, 0.0142)  # SBS footprint
L1, L2 = 0.302, 0.289  # SCARA link lengths (PF400 class reach ~0.59 m)
ARM_BASE = np.array([0.62, -0.30, 0.0])
GRIP_DROP = 0.16  # gripper fingertip below the link plane
CHANNEL_PITCH = 0.009
ELBOW = 1  # preferred elbow branch (sign of the elbow angle)
CARRIER_TOP = 0.183  # PLR plate-site height on a carrier
# STARlet housing (center, size), matching the web viewer. The arm enters through the open front
# below the header, whose underside sits above the arm's transit plane.
FRAME = {"left": ((-0.02, 0.31, 0.50), (0.04, 0.72, 0.80)), "right": ((1.02, 0.31, 0.50), (0.04, 0.72, 0.80)),
         "back": ((0.50, 0.67, 0.50), (1.06, 0.03, 0.80)), "roof": ((0.50, 0.31, 0.915), (1.06, 0.72, 0.03)),
         "header": ((0.50, -0.04, 0.87), (1.04, 0.03, 0.10)),
         "wasteblock": ((0.790, 0.3376, 0.15), (0.030, 0.4452, 0.10))}  # PLR waste block, rail 31
RACK_HOTEL, RACK_WASTE, TIP_BIN = (0.50, -0.62), (0.99, -0.25), (0.80, -0.17)
INSIDE = ("lh", "tiprack", "tiprack1", "handoff")  # inside the housing: reached straight through the open front


@dataclass(frozen=True)
class Station:
    name: str
    center: tuple[float, float]  # plate center in world xy
    plate_z: float  # plate bottom height
    yaw: float = 0.0  # plate long axis angle


@dataclass
class CellSpec:
    stations: dict[str, Station] = field(default_factory=dict)
    deck: dict = field(default_factory=dict)
    lh_sites: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # name -> top center


def cell_spec(design: DeckDesign | None = None) -> CellSpec:
    """`design` places the tip carrier whose front rack site the arm swaps boxes at."""
    design = design or DeckDesign()
    g = deck_geometry()
    res = {k: v for k, v in g["resources"].items()}
    a = res["assay"]
    lh_center = (a["origin"][0] / 1000 + PLATE[0] / 2, a["origin"][1] / 1000 + PLATE[1] / 2)
    spec = CellSpec(deck=g)
    spec.stations = {
        "lh": Station("lh", lh_center, a["origin"][2] / 1000),
        "hotel": Station("hotel", (1.05, -0.45), 0.25),
        "incubator": Station("incubator", (0.25, -0.55), 0.30),
        "reader": Station("reader", (0.80, -0.735), 0.20),
        "rackhotel": Station("rackhotel", RACK_HOTEL, 0.31),
        "tiprack": Station("tiprack", (design.x_of(design.tip_rail) / 1000 + CARRIER_RAILS * RAIL_PITCH / 2000,
                                       SITE_Y0 / 1000), CARRIER_TOP),
        "rackwaste": Station("rackwaste", RACK_WASTE, 0.30),  # released above the open bin
        # site 1 of the tip carrier (the arm reaches sites 0-1), and the handoff site: site 1 of the
        # assay-plate carrier, where the LH CO-RE gripper leaves and takes boxes from deep sites
        "tiprack1": Station("tiprack1", (design.x_of(design.tip_rail) / 1000 + CARRIER_RAILS * RAIL_PITCH / 2000,
                                         (SITE_Y0 + SITE_PITCH) / 1000), CARRIER_TOP),
        "handoff": Station("handoff", (design.x_of(design.dest_rail) / 1000 + CARRIER_RAILS * RAIL_PITCH / 2000,
                                       (SITE_Y0 + SITE_PITCH) / 1000), CARRIER_TOP),
    }
    def top(name):
        r = res[name]
        return (r["origin"][0] / 1000 + r["size"][0] / 2000, r["origin"][1] / 1000 + r["size"][1] / 2000,
                (r["origin"][2] + r["size"][2]) / 1000)
    spec.lh_sites = {"tips": top("tips_2"), "source": top("drug0"), "dest": top("assay"),
                     "trash": (0.80, 0.31, 0.137)}
    return spec


def _box(name, center, size, rgba, contype=1, conaffinity=1) -> str:
    c = " ".join(f"{x:.5f}" for x in center)
    s = " ".join(f"{x / 2:.5f}" for x in size)
    return (f'<geom name="{name}" type="box" pos="{c}" size="{s}" rgba="{rgba}" '
            f'contype="{contype}" conaffinity="{conaffinity}"/>')


def build_mjcf(spec: CellSpec, plates: dict[str, str] | None = None, obstacles: list[tuple] = ()) -> str:
    """plates: plate body name -> station it starts at."""
    plates = plates or {"plate0": "hotel"}
    g = spec.deck
    parts = ['<mujoco model="labloop_cell"><compiler angle="radian"/>',
             '<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>',
             '<visual><global offwidth="1280" offheight="960"/></visual>',
             '<default><geom friction="1 0.01 0.001"/><joint damping="20" armature="0.2"/></default>',
             '<asset><texture name="grid" type="2d" builtin="checker" width="256" height="256" '
             'rgb1=".92 .92 .92" rgb2=".86 .86 .86"/><material name="floor" texture="grid" texrepeat="8 8"/></asset>',
             '<worldbody><light pos="0.5 0 2.5" dir="0 0 -1" diffuse=".9 .9 .9"/>',
             '<geom name="floor" type="plane" size="3 3 .1" material="floor" contype="0" conaffinity="0"/>',
             '<camera name="overview" pos="1.9 -1.9 1.6" xyaxes="0.7 0.7 0 -0.35 0.35 0.87"/>',
             '<camera name="deckcam" pos="0.42 0.25 0.95" xyaxes="1 0 0 0 1 0" fovy="45"/>']
    # Deck table and PLR resources (static).
    dx, dy = g["deck"]["size"][0] / 1000, g["deck"]["size"][1] / 1000
    parts.append(_box("deck", (dx / 2, dy / 2, 0.05), (dx, dy, 0.1), ".55 .58 .62 1"))
    colors = {"tip_carrier": ".35 .45 .7 1", "plate_carrier": ".45 .5 .55 1", "tip_rack": ".2 .3 .8 1",
              "plate": ".95 .95 .9 1"}
    for name, r in g["resources"].items():
        if r["category"] not in colors or min(r["size"]) <= 0:
            continue
        o = np.array(r["origin"]) / 1000
        s = np.array(r["size"]) / 1000
        if r["category"] in ("plate_carrier", "tip_carrier"):
            s[2] = 0.083  # carrier top = PLR plate site height (0.183 m)
        if name == "assay":
            continue  # the assay plate site holds a movable plate instead
        parts.append(_box(f"plr_{name}", o + s / 2, s, colors[r["category"]], contype=2, conaffinity=3))
    # Off-deck devices (static bodies), each with a nest the plate sits on.
    dev = {"hotel": ((1.05, -0.58, 0.20), (0.18, 0.12, 0.40), ".6 .6 .65 1"),
           "incubator": ((0.25, -0.72, 0.25), (0.30, 0.22, 0.50), ".85 .85 .9 1"),
           "reader": ((0.80, -0.90, 0.10), (0.30, 0.22, 0.20), ".3 .3 .35 1"),
           "rackhotel": ((*RACK_HOTEL, 0.15), (0.16, 0.12, 0.30), ".75 .85 .9 1")}
    for name, (c, s, rgba) in dev.items():
        parts.append(_box(f"dev_{name}", c, s, rgba))
    for name, (cx, cy), (w, d, h) in (("rackwaste", RACK_WASTE, (0.20, 0.16, 0.22)), ("tipbin", TIP_BIN, (0.16, 0.14, 0.13))):
        for k, (c, s) in enumerate((((cx, cy - d / 2, h / 2), (w, 0.006, h)), ((cx, cy + d / 2, h / 2), (w, 0.006, h)),
                                    ((cx - w / 2, cy, h / 2), (0.006, d, h)), ((cx + w / 2, cy, h / 2), (0.006, d, h)))):
            parts.append(_box(f"dev_{name}{k}", c, s, ".85 .65 .2 1"))  # open-top bins
    tr = spec.stations["tiprack"]  # the design's tip carrier (the bridge deck keeps its own at rail 1)
    parts.append(_box("plr_tipcarrier_design", (tr.center[0], 0.3115, (0.10 + CARRIER_TOP) / 2),
                      (CARRIER_RAILS * RAIL_PITCH / 1000, 0.497, CARRIER_TOP - 0.10), ".35 .45 .7 1", contype=2, conaffinity=3))
    ho = spec.stations["handoff"]  # the design's assay-plate carrier (holds the handoff site)
    parts.append(_box("plr_destcarrier_design", (ho.center[0], 0.3115, (0.10 + CARRIER_TOP) / 2),
                      (CARRIER_RAILS * RAIL_PITCH / 1000, 0.497, CARRIER_TOP - 0.10), ".45 .5 .55 1", contype=2, conaffinity=3))
    for name, (c, s) in FRAME.items():  # housing: hits the arm and carried plates, not the gantry
        parts.append(_box(f"frame_{name}", c, s, ".82 .85 .87 1", contype=32, conaffinity=9))
    for name, st in spec.stations.items():
        if name in ("lh", "tiprack", "tiprack1", "handoff", "rackwaste", "rackhotel"):
            continue
        parts.append(_box(f"nest_{name}", (st.center[0], st.center[1], st.plate_z - 0.01), (0.14, 0.10, 0.02),
                          ".2 .2 .2 1"))
    for i, (c, s) in enumerate(obstacles):
        parts.append(_box(f"obstacle{i}", c, s, ".9 .2 .2 1"))
    # Pipetting gantry: the X-arm beam spans the deck front to back and slides in x; the Y head
    # rides on it; the channels move in z. The beam and head collide with the plate arm (group 16
    # vs the arm's 8) and with a carried plate, not with deck labware.
    head_z = 0.42
    parts.append('<body name="lh_xarm" pos="0 0 0"><joint name="lh_x" type="slide" axis="1 0 0" range="0 1.0"/>'
                 f'<geom name="lh_beam" type="box" pos="0 .33 {head_z + 0.08}" size=".03 .33 .025" rgba=".36 .39 .42 1" '
                 'mass="6" contype="16" conaffinity="9"/>'
                 '<body name="lh_ycar" pos="0 0 0"><joint name="lh_y" type="slide" axis="0 1 0" range="0 0.65"/>'
                 f'<geom name="lh_yhead" type="box" pos="0 0 {head_z + 0.02}" size=".03 .05 .05" rgba=".23 .26 .28 1" '
                 'mass="2" contype="16" conaffinity="9"/>'
                 f'<body name="lh_gantry" pos="0 0 {head_z}">'
                 '<joint name="lh_z" type="slide" axis="0 0 1" range="-0.25 0"/>'
                 '<geom type="box" size=".02 .045 .03" rgba=".9 .6 .1 1" mass="4" contype="0" conaffinity="0"/>')
    for k in range(8):
        y = (k - 3.5) * CHANNEL_PITCH
        parts.append(f'<geom name="channel{k}" type="cylinder" pos="0 {y:.4f} -.07" size=".0035 .04" '
                     f'rgba=".8 .8 .8 1" contype="4" conaffinity="0"/>')
    parts.append('</body></body></body>')
    # SCARA arm: z column, shoulder, elbow, wrist, gripper fingers.
    bx, by, bz = ARM_BASE
    parts.append(f'<body name="arm_base" pos="{bx} {by} {bz}">'
                 '<geom type="cylinder" size=".07 .4" pos="0 0 .4" rgba=".25 .25 .28 1" contype="0" conaffinity="0"/>'
                 '<body name="arm_carriage" pos="0 0 0">'
                 '<joint name="arm_z" type="slide" axis="0 0 1" range="0.2 0.75"/>'
                 '<geom type="box" size=".08 .08 .03" rgba=".3 .3 .35 1" contype="0" conaffinity="0"/>'
                 '<body name="arm_link1" pos="0 0 0">'
                 '<joint name="arm_shoulder" type="hinge" axis="0 0 1" range="-3.14 3.14"/>'
                 f'<geom name="link1" type="capsule" fromto="0 0 0 {L1} 0 0" size=".035" rgba=".85 .85 .88 1" mass="3" '
                 'contype="8" conaffinity="3"/>'
                 f'<body name="arm_link2" pos="{L1} 0 0">'
                 '<joint name="arm_elbow" type="hinge" axis="0 0 1" range="-2.8 2.8"/>'
                 f'<geom name="link2" type="capsule" fromto="0 0 0 {L2} 0 0" size=".03" rgba=".85 .85 .88 1" mass="2" '
                 'contype="8" conaffinity="3"/>'
                 f'<body name="arm_wrist" pos="{L2} 0 0">'
                 '<joint name="arm_wrist" type="hinge" axis="0 0 1" range="-6.3 6.3"/>'
                 f'<geom name="wrist" type="cylinder" fromto="0 0 0 0 0 -{GRIP_DROP - 0.02}" size=".02" '
                 'rgba=".4 .4 .45 1" mass="1" contype="8" conaffinity="3"/>'
                 f'<site name="grip" pos="0 0 -{GRIP_DROP}" size=".005"/>'
                 f'<geom name="finger_a" type="box" pos=".07 0 -{GRIP_DROP - 0.01}" size=".005 .03 .015" '
                 'rgba=".2 .2 .2 1" contype="0" conaffinity="0"/>'
                 f'<geom name="finger_b" type="box" pos="-.07 0 -{GRIP_DROP - 0.01}" size=".005 .03 .015" '
                 'rgba=".2 .2 .2 1" contype="0" conaffinity="0"/>'
                 '</body></body></body></body></body>')
    # Movable plates.
    for pname, st_name in plates.items():
        st = spec.stations[st_name]
        z = st.plate_z + PLATE[2] / 2
        parts.append(f'<body name="{pname}" pos="{st.center[0]} {st.center[1]} {z}"><freejoint/>'
                     f'<geom name="{pname}_geom" type="box" size="{PLATE[0] / 2} {PLATE[1] / 2} {PLATE[2] / 2}" '
                     'rgba=".95 .95 .85 1" mass=".08" contype="1" conaffinity="1"/></body>')
    parts.append('</worldbody>')
    # Weld that carries a plate with the gripper (activated on grasp).
    parts.append('<equality>' + "".join(
        f'<weld name="carry_{p}" body1="arm_wrist" body2="{p}" active="false"/>' for p in plates) + '</equality>')
    # Position actuators follow planned joint trajectories.
    acts = [("arm_z", 40000, 1500), ("arm_shoulder", 4000, 150), ("arm_elbow", 2500, 80), ("arm_wrist", 400, 15),
            ("lh_x", 8000, 300), ("lh_y", 8000, 300), ("lh_z", 8000, 300)]
    parts.append('<actuator>' + "".join(f'<position name="act_{j}" joint="{j}" kp="{kp}" kv="{kv}"/>'
                                        for j, kp, kv in acts) + '</actuator>')
    parts.append('<contact><exclude body1="arm_wrist" body2="arm_link2"/>'
                 '<exclude body1="arm_link2" body2="arm_link1"/></contact>')
    parts.append('</mujoco>')
    return "\n".join(parts)


SHOULDER_RANGE, ELBOW_RANGE = 3.1, 2.8


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def scara_ik(xy: tuple[float, float], z_tip: float, yaw: float = 0.0, elbow: int | None = None) -> dict[str, float]:
    """Joint targets putting the gripper tip at (x, y, z_tip) with the plate long axis at yaw.
    Picks the elbow configuration that stays inside the joint limits, `ELBOW` first."""
    x, y = xy[0] - ARM_BASE[0], xy[1] - ARM_BASE[1]
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1:
        raise ValueError(f"target {xy} out of reach ({math.sqrt(r2):.3f} m)")
    pref = elbow or ELBOW
    for sign in (pref, -pref):
        q2 = math.acos(c2) * sign
        q1 = _wrap(math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2)))
        if abs(q1) <= SHOULDER_RANGE and abs(q2) <= ELBOW_RANGE:
            return {"arm_z": z_tip + GRIP_DROP, "arm_shoulder": q1, "arm_elbow": q2,
                    "arm_wrist": _wrap(yaw - q1 - q2)}
    raise ValueError(f"target {xy} not reachable within joint limits")


def load(spec: CellSpec | None = None, **kw):
    spec = spec or cell_spec()
    m = mujoco.MjModel.from_xml_string(build_mjcf(spec, **kw))
    return spec, m, mujoco.MjData(m)
