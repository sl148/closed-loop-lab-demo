"""Pre-launch deck check: is the liquid handler deck set up as designed?

An overhead camera above the STARlet deck renders a depth image and a color image of the deck
as an operator set it up. For each site of the designed layout (plates first, then tip racks)
the check reads, from pixels only (no simulator object ids):
- what sits there, from the height of its top surface: nothing, a 96-well plate, the cell
  trough, or a tip rack, whose height gives the tip size (50, 300 or 1000 uL);
- where it sits: centroid and long-axis angle of the top surface, back-projected to the deck;
- which plate it is, from the color of the label strip on its front edge (a stand-in for a
  barcode read).
A site fails when its contents, plate identity, position (> POS_TOL_MM) or angle (> YAW_TOL_DEG)
differ from the design. `run_check` sets the deck up with injected mistakes, checks it, fixes
what was flagged, and checks again.

Labware heights are approximate (SBS plate 14.2 mm; racks with tips by tip length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from labloop.sim3d.perception import POS_TOL_MM, YAW_TOL_DEG

RAIL0, PITCH, CAR_W, CAR_Y0, CAR_D = 0.100, 0.0225, 0.135, 0.063, 0.497
SITE_Y0, SITE_P, CAR_TOP, DECK_TOP = 0.111, 0.096, 0.183, 0.10
FOOT = (0.12776, 0.08548)
HEIGHT = {"plate": 0.0142, "trough": 0.040, "tips50uL": 0.060, "tips300uL": 0.072, "tips1000uL": 0.105}
LABELS = [(0.85, 0.20, 0.20), (0.15, 0.60, 0.25), (0.20, 0.35, 0.90), (0.90, 0.65, 0.05), (0.60, 0.25, 0.75)]
W, H, FOVY = 1280, 960, 40.0
CAM = (0.45, 0.31, 1.30)
DEPTH_NOISE_M = 0.001


@dataclass
class Item:
    """What a site should hold (`kind`: plate, trough, tips50uL/tips300uL/tips1000uL, empty)."""
    id: str
    group: str  # plates | tips (the check order)
    carrier: str  # sources | assay | tips
    rail: int
    site: int
    kind: str
    label: int | None = None  # drug index of a dilution plate
    name: str = ""

    @property
    def center(self) -> tuple[float, float]:
        return RAIL0 + (self.rail - 1) * PITCH + CAR_W / 2, SITE_Y0 + self.site * SITE_P


@dataclass
class Placed:
    """What the operator actually put on a site."""
    kind: str
    label: int | None = None
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class Layout:
    items: list[Item] = field(default_factory=list)
    carriers: dict[str, int] = field(default_factory=dict)


def expected_layout(deck: dict, drugs: list[str], racks: list[str]) -> Layout:
    """Plates (sources, then the assay carrier's empty sites) first, then the tip racks."""
    L = Layout(carriers={"sources": deck["source_rail"], "assay": deck["dest_rail"], "tips": deck["tip_rail"]})
    cell = "trough" if deck.get("cell_source", "trough") == "trough" else "plate"
    L.items.append(Item("src0", "plates", "sources", deck["source_rail"], 0, cell, None, "cells"))
    for i, d in enumerate(drugs[:4]):
        L.items.append(Item(f"src{i + 1}", "plates", "sources", deck["source_rail"], i + 1, "plate", i, d))
    L.items.append(Item("assay0", "plates", "assay", deck["dest_rail"], 0, "empty", None, "assay site"))
    L.items.append(Item("assay1", "plates", "assay", deck["dest_rail"], 1, "empty", None, "handoff site"))
    for s, t in enumerate(racks):
        L.items.append(Item(f"tips{s}", "tips", "tips", deck["tip_rail"], s, f"tips{t}", None, f"{t} tips"))
    return L


def build_scene(layout: Layout, placed: dict[str, Placed]) -> str:
    parts = ['<mujoco model="deck_check"><compiler angle="radian"/>',
             f'<visual><global offwidth="{W}" offheight="{H}"/><map znear="0.05" zfar="5"/>'
             '<headlight ambient=".25 .25 .25" diffuse=".2 .2 .2" specular="0 0 0"/></visual>',
             '<worldbody><light pos="0.45 0.31 2.5" dir="0 0 -1" diffuse=".45 .45 .45" specular="0 0 0"/>',
             f'<camera name="top" pos="{CAM[0]} {CAM[1]} {CAM[2]}" xyaxes="1 0 0 0 1 0" fovy="{FOVY}"/>',
             f'<geom type="box" pos="0.5 0.33 {DECK_TOP / 2}" size="0.55 0.33 {DECK_TOP / 2}" rgba=".55 .58 .62 1"/>']
    for name, rail in layout.carriers.items():
        cx = RAIL0 + (rail - 1) * PITCH + CAR_W / 2
        parts.append(f'<geom type="box" pos="{cx} {CAR_Y0 + CAR_D / 2} {(DECK_TOP + CAR_TOP) / 2}" '
                     f'size="{CAR_W / 2} {CAR_D / 2} {(CAR_TOP - DECK_TOP) / 2}" rgba=".42 .47 .52 1"/>')
    for it in layout.items:
        p = placed.get(it.id)
        if p is None or p.kind == "empty":
            continue
        cx, cy = it.center
        x, y, th, h = cx + p.dx_mm / 1000, cy + p.dy_mm / 1000, math.radians(p.yaw_deg), HEIGHT[p.kind]
        rgba = {"plate": ".93 .93 .9 1", "trough": ".85 .88 .92 1"}.get(p.kind, ".25 .3 .75 1")
        parts.append(f'<body pos="{x} {y} {CAR_TOP}" euler="0 0 {th}">'
                     f'<geom type="box" pos="0 0 {h / 2}" size="{FOOT[0] / 2} {FOOT[1] / 2} {h / 2}" rgba="{rgba}"/>')
        if p.kind == "plate" and p.label is not None:
            r, g, b = LABELS[p.label % len(LABELS)]
            parts.append(f'<geom type="box" pos="0 {-FOOT[1] / 2 + 0.0045} {h + 0.0003}" size="0.04 0.0035 0.0003" rgba="{r} {g} {b} 1"/>')
        parts.append('</body>')
    parts.append('</worldbody></mujoco>')
    return "\n".join(parts)


class TopCamera:
    def __init__(self, m: mujoco.MjModel):
        self.m, self.r = m, mujoco.Renderer(m, H, W)
        self.f = (H / 2) / math.tan(math.radians(FOVY) / 2)

    def capture(self, d: mujoco.MjData, rng: np.random.Generator):
        """World x, y, z per pixel (from depth, with noise) and the color image."""
        self.r.update_scene(d, camera="top")
        self.r.enable_depth_rendering()
        depth = self.r.render().astype(float) + rng.normal(0, DEPTH_NOISE_M, (H, W))
        self.r.disable_depth_rendering()
        rgb = self.r.render().astype(float) / 255
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x = CAM[0] + (u - (W - 1) / 2) * depth / self.f
        y = CAM[1] - (v - (H - 1) / 2) * depth / self.f
        return x, y, CAM[2] - depth, rgb


def classify(height: float) -> str:
    return min(HEIGHT, key=lambda k: abs(HEIGHT[k] - height))


def read_site(it: Item, x, y, z, rgb, margin=(0.005, 0.006)) -> dict:
    cx, cy = it.center
    roi = (np.abs(x - cx) < FOOT[0] / 2 + margin[0]) & (np.abs(y - cy) < FOOT[1] / 2 + margin[1])
    mask = roi & (z > CAR_TOP + 0.004)
    foot_px = roi.sum() * (FOOT[0] * FOOT[1]) / ((FOOT[0] + 2 * margin[0]) * (FOOT[1] + 2 * margin[1]))
    if mask.sum() < 0.3 * foot_px:
        return {"kind": "empty", "area": round(float(mask.sum() / max(1, foot_px)), 2)}
    zt = float(np.median(z[mask])) - CAR_TOP
    kind = classify(zt)
    top = mask & (np.abs(z - CAR_TOP - HEIGHT[kind]) < 0.004)  # the top face only, not label or edges
    px, py = x[top], y[top]
    mx, my = float(px.mean()), float(py.mean())
    cov = np.cov(np.stack([px - mx, py - my]))
    ev, evec = np.linalg.eigh(cov)
    major = evec[:, np.argmax(ev)]
    yaw = math.degrees(math.atan2(major[1], major[0]))
    yaw = (yaw + 90) % 180 - 90
    out = {"kind": kind, "height_mm": round(zt * 1000, 1), "dx_mm": round((mx - cx) * 1000, 2),
           "dy_mm": round((my - cy) * 1000, 2), "yaw_deg": round(yaw, 2), "area": round(float(mask.sum() / foot_px), 2)}
    if kind == "plate":
        t = math.radians(yaw)
        lx = (x - mx) * math.cos(t) + (y - my) * math.sin(t)
        ly = -(x - mx) * math.sin(t) + (y - my) * math.cos(t)
        strip = mask & (np.abs(lx) < 0.035) & (ly < -FOOT[1] / 2 + 0.008) & (ly > -FOOT[1] / 2 + 0.001)
        if strip.sum() > 20:
            chroma = lambda c: (np.asarray(c, float) - np.mean(c)) / (np.linalg.norm(np.asarray(c, float) - np.mean(c)) + 1e-9)
            col = chroma(rgb[strip].mean(axis=0))  # hue direction, independent of brightness
            sim = [float(col @ chroma(c)) for c in LABELS]
            j = int(np.argmax(sim))
            out["label"] = j if sim[j] > 0.9 else None
        else:
            out["label"] = None
    return out


def verdict(it: Item, seen: dict, names: list[str]) -> tuple[bool, str]:
    tip = lambda k: k.replace("tips", "").replace("uL", " µL")
    if it.kind == "empty":
        return (True, "") if seen["kind"] == "empty" else (False, f"should be empty, holds {seen['kind']}")
    if seen["kind"] == "empty":
        return False, "missing"
    if seen["kind"] != it.kind:
        if it.kind.startswith("tips") and seen["kind"].startswith("tips"):
            return False, f"{tip(seen['kind'])} tips, needs {tip(it.kind)}"
        return False, f"{seen['kind']}, needs {it.kind}"
    if it.label is not None and seen.get("label") != it.label:
        got = names[seen["label"]] if seen.get("label") is not None and seen["label"] < len(names) else "unreadable label"
        return False, f"{got} plate here, needs {names[it.label]}"
    off = math.hypot(seen["dx_mm"], seen["dy_mm"])
    if off > POS_TOL_MM or abs(seen["yaw_deg"]) > YAW_TOL_DEG:
        return False, f"off by {off:.1f} mm" + (f", {abs(seen['yaw_deg']):.1f}°" if abs(seen["yaw_deg"]) > YAW_TOL_DEG else "")
    return True, ""


def check(layout: Layout, placed: dict[str, Placed], names: list[str], rng: np.random.Generator) -> list[dict]:
    m = mujoco.MjModel.from_xml_string(build_scene(layout, placed))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = TopCamera(m)
    x, y, z, rgb = cam.capture(d, rng)
    out = []
    for it in layout.items:  # plates first, then tips: the layout's order
        seen = read_site(it, x, y, z, rgb)
        ok, issue = verdict(it, seen, names)
        p = placed.get(it.id) or Placed("empty")
        out.append({"id": it.id, "group": it.group, "carrier": it.carrier, "site": it.site, "name": it.name,
                    "expect": {"kind": it.kind, "label": it.label},
                    "truth": {"kind": p.kind, "label": p.label, "dx_mm": p.dx_mm, "dy_mm": p.dy_mm, "yaw_deg": p.yaw_deg},
                    "seen": seen, "ok": ok, "issue": issue})
    cam.r.close()
    return out


def mistakes(layout: Layout) -> dict[str, Placed]:
    """Setup errors an operator makes: two drug plates swapped, a plate set down off its nest,
    a rack of the wrong tip size, a rack left out."""
    out = {}
    src = [i for i in layout.items if i.carrier == "sources" and i.label is not None]
    tips = [i for i in layout.items if i.group == "tips"]
    if len(src) >= 3:
        a, b = src[1], src[2]
        out[a.id], out[b.id] = Placed("plate", b.label), Placed("plate", a.label)
    if len(src) >= 4:
        out[src[3].id] = Placed("plate", src[3].label, dx_mm=3.2, dy_mm=-1.4, yaw_deg=1.5)
    if len(tips) >= 3:
        wrong = "tips50uL" if tips[2].kind != "tips50uL" else "tips300uL"
        out[tips[2].id] = Placed(wrong)
    if len(tips) >= 5:
        out[tips[4].id] = Placed("empty")
    return out


def run_check(layout: Layout, names: list[str], seed: int = 0) -> dict:
    """Set up with mistakes, check, fix what was flagged, check again."""
    rng = np.random.default_rng(seed)
    jitter = lambda: dict(dx_mm=round(float(rng.normal(0, 0.3)), 2), dy_mm=round(float(rng.normal(0, 0.3)), 2),
                          yaw_deg=round(float(rng.normal(0, 0.15)), 2))
    right = {it.id: Placed(it.kind, it.label, **jitter()) for it in layout.items if it.kind != "empty"}
    first = {**right, **mistakes(layout)}
    p1 = check(layout, first, names, rng)
    fixed = dict(first)
    for r in p1:
        if not r["ok"]:
            it = next(i for i in layout.items if i.id == r["id"])
            fixed[it.id] = Placed(it.kind, it.label, **jitter()) if it.kind != "empty" else Placed("empty")
    p2 = check(layout, fixed, names, rng)
    err = [math.hypot(r["seen"]["dx_mm"] - r["truth"]["dx_mm"], r["seen"]["dy_mm"] - r["truth"]["dy_mm"])
           for r in p1 + p2 if "dx_mm" in r["seen"] and r["seen"]["kind"] == r["truth"]["kind"]]
    mm_px = (CAM[2] - CAR_TOP) * 2 * math.tan(math.radians(FOVY) / 2) / H * 1000
    return {"tol_mm": POS_TOL_MM, "tol_deg": YAW_TOL_DEG, "mm_per_px": round(mm_px, 3), "depth_noise_mm": DEPTH_NOISE_M * 1000,
            "position_error_mm": {"mean": round(float(np.mean(err)), 3), "max": round(float(np.max(err)), 3)} if err else None,
            "names": names, "passes": [p1, p2]}
