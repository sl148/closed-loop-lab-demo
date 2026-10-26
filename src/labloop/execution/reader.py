"""Plate reader: true viability from the oracle -> noisy raw values -> control-normalized."""

from __future__ import annotations

import dataclasses

import numpy as np

from labloop.oracle.noise import NoiseModel
from labloop.oracle.surface import Oracle
from labloop.types import CONTROL_NEG, CONTROL_POS, EMPTY, PlateMap, WellRead


class PlateReader:
    def __init__(self, oracle: Oracle, noise: NoiseModel):
        self.oracle, self.noise = oracle, noise

    def read(self, plate: PlateMap, t: int, rng: np.random.Generator,
             bad_wells: set[str] = frozenset(), corrupt: set[str] = frozenset(),
             effects: dict | None = None) -> list[WellRead]:
        """bad_wells come back flagged invalid; corrupt wells (failed transfer or outlier
        read that nobody invalidated) come back valid but with a wrong value. `effects` are the
        lab's physical limits the plan broke (execution/truth.py): potency loss, the actual
        incubation time, extra well-to-well variation."""
        fx = effects or {}
        dose_scale, tp = fx.get("dose_scale", 1.0), fx.get("timepoint_h", plate.timepoint_h)
        # a tip used below its validated volume: doses scaled and spread (one draw per dose, as before)
        dose_scale *= 1 + fx.get("dose_bias", 0.0)
        noise = dataclasses.replace(self.noise, pipette_cv=fx["dose_cv"]) if fx.get("dose_cv") else self.noise
        raw: dict[tuple[int, int], float] = {}
        for (r, c), content in sorted(plate.wells.items()):
            if content == EMPTY:
                continue
            if content == CONTROL_NEG:
                v = 1.0
            elif content == CONTROL_POS:
                v = 0.0
            else:
                ds = fx.get("drug_scale") or [1.0] * len(content.doses)  # stocks that did not fully dissolve
                doses = [noise.perturb_dose(d * dose_scale * s, rng) for d, s in zip(content.doses, ds)]
                v = self.oracle.viability(plate.cell_line, doses, tp)
                dm = fx.get("dmso")
                if dm:  # the well's DMSO from each drug added, against what cells tolerate
                    pct = sum(p * (1.0 if flat else d / top) for d, p, flat, top in zip(content.doses, dm["top_pct"], dm["flat"], dm["tops"]) if d > 0)
                    v *= max(0.0, 1 - dm["kill_per_pct"] * max(0.0, pct - dm["max"]))
            raw[(r, c)] = self.noise.measure(v, plate.spec.is_edge(r, c), rng)
            if fx.get("extra_noise"):
                raw[(r, c)] *= max(0.0, 1 + fx["extra_noise"] * rng.standard_normal())
            if plate.spec.well_name(r, c) in corrupt:
                raw[(r, c)] *= rng.uniform(0.2, 1.8)
        if fx.get("crosstalk"):  # light from the four neighbours leaks through clear walls
            k, base = fx["crosstalk"], dict(raw)
            for (r, c) in raw:
                nb = [base[q] for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if q in base]
                if nb:
                    raw[(r, c)] = base[(r, c)] + k * float(np.mean(nb))

        def mean_of(label, edge=None):
            vals = [raw[k] for k, w in plate.wells.items()
                    if w == label and (edge is None or plate.spec.is_edge(*k) == edge)]
            return float(np.mean(vals)) if vals else None

        pos = mean_of(CONTROL_POS)
        pos = 0.0 if pos is None else pos
        neg_all = mean_of(CONTROL_NEG)
        # Normalize edge and interior wells to vehicle controls in the same region when the
        # layout provides both; this is what makes the edge-aware layout pay off.
        neg_region = {True: mean_of(CONTROL_NEG, True), False: mean_of(CONTROL_NEG, False)}
        out = []
        for (r, c), value in sorted(raw.items()):
            edge = plate.spec.is_edge(r, c)
            neg = neg_region[edge] if neg_region[True] is not None and neg_region[False] is not None else neg_all
            neg = 1.0 if neg is None else neg
            span = neg - pos if neg - pos > 1e-9 else 1.0
            name = plate.spec.well_name(r, c)
            out.append(WellRead(plate.plate_id, name, plate.wells[(r, c)], edge, value,
                                float((value - pos) / span), t, valid=name not in bad_wells))
        return out
