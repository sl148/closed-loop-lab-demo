"""Parametric dose-response oracle for n drugs: Hill single agents, Bliss baseline,
planted synergy pockets, and optional pairwise excess terms fitted from data (emulator.py).

Viability v in [0, 1] (1 = untreated). Bliss excess = prod_i v_i(d_i) - v(d).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class HillDrug:
    name: str
    ec50: float  # uM
    hill: float
    emax: float  # max fractional kill in [0, 1]

    def viability(self, dose) -> np.ndarray:
        d = np.asarray(dose, dtype=float)
        safe = np.where(d > 0, d, 1.0)
        frac = np.where(d > 0, safe**self.hill / (safe**self.hill + self.ec50**self.hill), 0.0)
        return 1.0 - self.emax * frac


@dataclass(frozen=True)
class SynergyPocket:
    """Gaussian bump of extra kill in log10-dose space over a subset of drugs; it is
    active only when all of those drugs are present."""

    drugs: tuple[int, ...]
    center_log10: tuple[float, ...]
    width_log10: float
    strength: float  # max Bliss excess at the center

    def excess(self, doses: np.ndarray) -> np.ndarray:
        sub = doses[..., list(self.drugs)]
        present = np.all(sub > 0, axis=-1)
        logs = np.log10(np.where(sub > 0, sub, 1.0))
        r2 = np.sum((logs - np.asarray(self.center_log10)) ** 2, axis=-1)
        return np.where(present, self.strength * np.exp(-r2 / (2 * self.width_log10**2)), 0.0)


PairExcess = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (d_i, d_j) -> Bliss excess


@dataclass(frozen=True)
class CombinationSurface:
    drugs: tuple[HillDrug, ...]
    pockets: tuple[SynergyPocket, ...] = ()
    pair_terms: dict[tuple[int, int], PairExcess] = field(default_factory=dict, hash=False)

    def bliss(self, doses) -> np.ndarray:
        doses = np.asarray(doses, float)
        return np.prod([dr.viability(doses[..., i]) for i, dr in enumerate(self.drugs)], axis=0)

    def true_excess(self, doses) -> np.ndarray:
        doses = np.asarray(doses, float)
        total = np.zeros(doses.shape[:-1])
        for p in self.pockets:
            total = total + p.excess(doses)
        for (i, j), f in self.pair_terms.items():
            both = (doses[..., i] > 0) & (doses[..., j] > 0)
            total = total + np.where(both, f(doses[..., i], doses[..., j]), 0.0)
        # Excess can't push viability below zero or above one.
        bl = self.bliss(doses)
        return np.clip(total, bl - 1.0, bl)

    def viability(self, doses) -> np.ndarray:
        return self.bliss(doses) - self.true_excess(doses)


@dataclass
class Oracle:
    """Maps (cell line, timepoint, doses) to a true viability. Noise is applied by the reader.

    Longer exposure deepens the effect: the drug effect (1 - v) scales by
    (t / t_ref) ** time_exponent, so the 24/48/72 h choice matters."""

    surfaces: dict[str, CombinationSurface]
    t_ref_h: float = 48.0
    time_exponent: float = 0.5

    def _scale(self, timepoint_h: float) -> float:
        return float((timepoint_h / self.t_ref_h) ** self.time_exponent)

    def viability(self, cell_line: str, doses, timepoint_h: float = 48.0) -> float:
        v = float(self.surfaces[cell_line].viability(np.asarray(doses, float)))
        return float(np.clip(1 - (1 - v) * self._scale(timepoint_h), 0.0, 1.0))

    def true_excess(self, cell_line: str, doses, timepoint_h: float = 48.0) -> float:
        """Bliss excess of the time-scaled surface, the quantity designers estimate."""
        doses = np.asarray(doses, float)
        singles = [np.where(np.arange(len(doses)) == i, doses, 0.0) for i in range(len(doses))]
        prod = np.prod([self.viability(cell_line, s, timepoint_h) for s in singles if s.any()] or [1.0])
        return float(prod - self.viability(cell_line, doses, timepoint_h))


def build_oracle(cfg: dict) -> Oracle:
    if cfg.get("source") == "oneil":
        return _oneil_cached(json.dumps(cfg, sort_keys=True))
    surfaces = {}
    for line, spec in cfg["cell_lines"].items():
        drugs = tuple(HillDrug(**d) for d in spec["drugs"])
        pockets = tuple(
            SynergyPocket(tuple(p.get("drugs", (0, 1))), tuple(p["center_log10"]), p["width_log10"], p["strength"])
            for p in spec.get("pockets", [])
        )
        surfaces[line] = CombinationSurface(drugs, pockets)
    return Oracle(surfaces, cfg.get("t_ref_h", 48.0), cfg.get("time_exponent", 0.5))


@lru_cache(maxsize=4)
def _oneil_cached(key: str) -> Oracle:
    from labloop.oracle.emulator import build_oneil_oracle

    return build_oneil_oracle(json.loads(key))
