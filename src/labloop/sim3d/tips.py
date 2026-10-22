"""Physics-based tip pickup check. A pipetting channel (cylindrical probe) descends onto a
disposable tip whose collar is a ring of capsules around a socket. If the channel is
aligned within the collar clearance it seats on the socket floor (pickup succeeds); otherwise
it lands on the collar rim first (pickup fails). The first contact decides.

`p_fail(sigma)` turns channel positioning error into the per-pickup failure probability
that the lab's failure model uses.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

PROBE_R = 0.0036
RING_R = 0.0054  # ring centerline radius
CAP_R = 0.0010  # collar capsule radius -> inner clearance radius 4.4 mm
N_CAPS = 24
CLEARANCE = RING_R - CAP_R - PROBE_R  # 0.8 mm


def _model(dx: float, dy: float) -> mujoco.MjModel:
    caps = []
    for k in range(N_CAPS):
        a0, a1 = 2 * math.pi * k / N_CAPS, 2 * math.pi * (k + 1) / N_CAPS
        p0 = (RING_R * math.cos(a0), RING_R * math.sin(a0))
        p1 = (RING_R * math.cos(a1), RING_R * math.sin(a1))
        caps.append(f'<geom name="rim{k}" type="capsule" fromto="{p0[0]:.6f} {p0[1]:.6f} 0 {p1[0]:.6f} {p1[1]:.6f} 0" '
                    f'size="{CAP_R}"/>')
    xml = f"""<mujoco><option timestep="0.0005" gravity="0 0 0"/>
<worldbody>
  <body name="tip" pos="0 0 0">{''.join(caps)}
    <geom name="socket" type="cylinder" pos="0 0 -0.004" size="{RING_R - CAP_R} 0.0005"/>
  </body>
  <body name="channel" pos="{dx} {dy} 0.010">
    <joint name="z" type="slide" axis="0 0 1" damping="1"/>
    <geom name="probe" type="cylinder" pos="0 0 0.004" size="{PROBE_R} 0.004" mass="0.05"/>
  </body>
</worldbody>
<actuator><velocity joint="z" kv="50"/></actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def try_pickup(dx: float, dy: float, speed: float = 0.01) -> bool:
    """Descend at `speed` (m/s) until first contact. True if the probe seats on the socket."""
    m = _model(dx, dy)
    d = mujoco.MjData(m)
    d.ctrl[0] = -speed
    socket = m.geom("socket").id
    for _ in range(20000):
        mujoco.mj_step(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            other = c.geom2 if m.geom(c.geom1).name == "probe" else c.geom1
            return other == socket
    raise RuntimeError("probe never made contact")


def offset_sweep(offsets_mm=(0.0, 0.4, 0.7, 0.8, 0.9, 1.2, 2.0)) -> dict:
    return {f"{o:.1f}": try_pickup(o / 1000, 0.0) for o in offsets_mm}


def p_fail(sigma_mm: float, n: int = 300, seed: int = 0) -> dict:
    """Monte Carlo failure rate under isotropic Gaussian xy positioning error, next to the
    analytic Rayleigh tail P(|e| > clearance) = exp(-c^2 / (2 sigma^2))."""
    rng = np.random.default_rng(seed)
    fails = sum(not try_pickup(*(rng.normal(0, sigma_mm / 1000, 2))) for _ in range(n))
    c = CLEARANCE * 1000
    return {"sigma_mm": sigma_mm, "n": n, "p_fail_sim": fails / n,
            "p_fail_analytic": math.exp(-c * c / (2 * sigma_mm * sigma_mm)), "clearance_mm": round(c, 3)}
