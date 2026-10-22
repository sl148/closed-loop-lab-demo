"""Experiment history and Bliss-excess estimation from measured wells only (never the oracle)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from labloop.types import Condition, WellRead


@dataclass
class History:
    reads: list[WellRead] = field(default_factory=list)

    def add(self, reads: list[WellRead]) -> None:
        self.reads.extend(r for r in reads if r.valid)

    def tested(self) -> set[Condition]:
        return {r.content for r in self.reads if isinstance(r.content, Condition)}

    def samples(self) -> dict[Condition, list[float]]:
        acc: dict[Condition, list[float]] = defaultdict(list)
        for r in self.reads:
            if isinstance(r.content, Condition):
                acc[r.content].append(r.viability)
        return acc

    def mean_viability(self) -> dict[Condition, float]:
        return {c: float(np.mean(v)) for c, v in self.samples().items()}

    def excess_estimates(self) -> dict[Condition, float]:
        """Bliss excess = prod_i v(single_i) - v(combo), using measured single-agent wells of
        the same cell line and timepoint. Combos with an untested single are skipped."""
        mv = self.mean_viability()
        out = {}
        for c, v in mv.items():
            if not c.is_combo:
                continue
            singles = [mv.get(c.single(i)) for i in c.active]
            if all(s is not None for s in singles):
                out[c] = float(np.prod(singles)) - v
        return out

    def noise_sd(self, default: float = 0.05) -> float:
        """Pooled replicate SD of viability; `default` until there are enough replicates."""
        groups = [v for v in self.samples().values() if len(v) >= 2]
        if sum(len(v) - 1 for v in groups) < 5:
            return default
        ss = sum(float(np.sum((np.asarray(v) - np.mean(v)) ** 2)) for v in groups)
        return float(np.sqrt(ss / sum(len(v) - 1 for v in groups)))

    def ranked_excess(self) -> list[tuple[Condition, float]]:
        return sorted(self.excess_estimates().items(), key=lambda kv: (-kv[1], kv[0]))

    def best(self) -> tuple[Condition, float] | None:
        """Recommended combination: highest lower confidence bound mean - sd/sqrt(n), so a
        single lucky noisy well does not beat a replicated hit. Returns (condition, mean)."""
        est = self.excess_estimates()
        if not est:
            return None
        counts = {c: len(v) for c, v in self.samples().items()}
        sd = self.noise_sd() * np.sqrt(2)  # excess combines a combo and its singles
        c = max(est, key=lambda k: (est[k] - sd / np.sqrt(counts[k]), k))
        return c, est[c]
