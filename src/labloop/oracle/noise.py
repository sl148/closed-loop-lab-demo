"""Measurement model: turns true viability into a plate-reader value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    rel_sigma: float = 0.0  # Gaussian, relative to signal
    edge_bias: float = 0.0  # multiplicative drop on outer-ring wells (evaporation)
    pipette_cv: float = 0.0  # dose error, relative

    def perturb_dose(self, dose: float, rng: np.random.Generator) -> float:
        if self.pipette_cv == 0 or dose == 0:
            return dose
        return max(0.0, dose * (1 + self.pipette_cv * rng.standard_normal()))

    def measure(self, viability: float, is_edge: bool, rng: np.random.Generator) -> float:
        v = viability * (1 - self.edge_bias) if is_edge else viability
        if self.rel_sigma:
            v *= 1 + self.rel_sigma * rng.standard_normal()
        return float(v)


def build_noise(cfg: dict | None) -> NoiseModel:
    return NoiseModel(**(cfg or {}))
