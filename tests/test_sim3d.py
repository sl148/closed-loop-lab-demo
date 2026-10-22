import math

import mujoco
import numpy as np
import pytest

from labloop.sim3d.cell import ARM_BASE, L1, L2, load, scara_ik
from labloop.sim3d.motion import SAFE_Z, joint_q, playback, set_joints, transfer_plan, trapezoid_s, trapezoid_time
from labloop.sim3d.tips import CLEARANCE, try_pickup


def test_trapezoid_profile():
    assert trapezoid_s(0, 2) == 0 and abs(trapezoid_s(2, 2) - 1) < 1e-12
    assert abs(trapezoid_s(1, 2) - 0.5) < 1e-9
    # Short moves never reach cruise speed: t = 2 sqrt(d / a).
    assert abs(trapezoid_time(0.01, 1.0, 1.0) - 0.2) < 1e-9


def test_scara_ik_forward_consistency():
    for xy in ((1.05, -0.45), (0.25, -0.55), (0.80, -0.735)):
        q = scara_ik(xy, 0.3)
        x = ARM_BASE[0] + L1 * math.cos(q["arm_shoulder"]) + L2 * math.cos(q["arm_shoulder"] + q["arm_elbow"])
        y = ARM_BASE[1] + L1 * math.sin(q["arm_shoulder"]) + L2 * math.sin(q["arm_shoulder"] + q["arm_elbow"])
        assert math.hypot(x - xy[0], y - xy[1]) < 1e-9
    with pytest.raises(ValueError):
        scara_ik((3.0, 3.0), 0.3)


def test_transfer_places_plate_without_collisions():
    spec, m, d = load(plates={"plate0": "hotel"})
    set_joints(m, d, {**scara_ik(spec.stations["hotel"].center, 0.4), "arm_z": SAFE_Z})
    for _ in range(300):
        mujoco.mj_step(m, d)
    pb = playback(m, d, transfer_plan(spec, "plate0", "hotel", "lh"), record=False)
    p = d.xpos[m.body("plate0").id]
    assert pb.collisions == []
    assert 1000 * math.hypot(p[0] - spec.stations["lh"].center[0], p[1] - spec.stations["lh"].center[1]) < 2.0


def test_transfer_into_obstacle_is_detected():
    spec, m, d = load(plates={"plate0": "hotel"}, obstacles=[((0.85, -0.25, 0.65), (0.3, 0.3, 0.3))])
    set_joints(m, d, {**scara_ik(spec.stations["hotel"].center, 0.4), "arm_z": SAFE_Z})
    for _ in range(300):
        mujoco.mj_step(m, d)
    pb = playback(m, d, transfer_plan(spec, "plate0", "hotel", "lh"), record=False)
    assert any("obstacle" in c for c in pb.collisions)


def test_tip_pickup_contact_boundary():
    c = CLEARANCE * 1000
    assert try_pickup(0.0, 0.0)
    assert try_pickup((c - 0.1) / 1000, 0.0)
    assert not try_pickup((c + 0.1) / 1000, 0.0)


@pytest.mark.parametrize("beam_x, hits", [(None, True), (0.95, False)])
def test_arm_collides_with_the_gantry_beam_over_the_site(beam_x, hits):
    spec, m, d = load(plates={"plate0": "hotel"})
    site = spec.stations["lh"].center
    set_joints(m, d, {**scara_ik(spec.stations["hotel"].center, 0.4), "arm_z": SAFE_Z,
                      "lh_x": site[0] if beam_x is None else beam_x, "lh_y": 0.6})
    for _ in range(300):
        mujoco.mj_step(m, d)
    pb = playback(m, d, transfer_plan(spec, "plate0", "hotel", "lh"), record=False)
    assert any("lh_" in c for c in pb.collisions) == hits


ARM_ROUTES = [("hotel", "lh"), ("lh", "incubator"), ("incubator", "reader"), ("reader", "hotel"),
              ("tiprack", "rackwaste"), ("rackhotel", "tiprack")]


@pytest.mark.parametrize("src, dst", ARM_ROUTES)
def test_every_arm_route_clears_the_housing_and_devices(src, dst):
    """With the gantry parked, no arm link or carried plate/box touches the STARlet housing,
    the bins, or another device on any route the scheduler uses."""
    spec, m, d = load(plates={"plate0": src})
    set_joints(m, d, {**scara_ik(spec.stations[src].center, 0.4), "arm_z": SAFE_Z, "lh_x": 0.95, "lh_y": 0.6})
    for _ in range(300):
        mujoco.mj_step(m, d)
    pb = playback(m, d, transfer_plan(spec, "plate0", src, dst), record=False)
    assert pb.collisions == [], pb.collisions
