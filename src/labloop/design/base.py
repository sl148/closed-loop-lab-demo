"""Designer interface and the discrete dose space shared by all designers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np

from labloop.evaluate import History
from labloop.types import Condition


@dataclass(frozen=True)
class DesignSpace:
    cell_lines: tuple[str, ...]
    ladders: tuple[tuple[float, ...], ...]  # per drug, each includes 0.0
    timepoints_h: tuple[int, ...]

    @property
    def n_drugs(self) -> int:
        return len(self.ladders)

    @cached_property
    def all_conditions(self) -> list[Condition]:
        return [
            Condition(cl, doses, t)
            for cl in self.cell_lines
            for t in self.timepoints_h
            for doses in itertools.product(*self.ladders)
            if any(d > 0 for d in doses)
        ]

    @cached_property
    def combos(self) -> list[Condition]:
        return [c for c in self.all_conditions if c.is_combo]

    @cached_property
    def singles(self) -> list[Condition]:
        return [c for c in self.all_conditions if c.is_single]

    def features(self, conds: list[Condition]) -> np.ndarray:
        """[0,1]-scaled features: per-drug log-dose index (0 = absent), cell-line one-hot,
        timepoint."""
        rows = []
        for c in conds:
            f = [self.ladders[i].index(d) / (len(self.ladders[i]) - 1) for i, d in enumerate(c.doses)]
            f += [1.0 if c.cell_line == cl else 0.0 for cl in self.cell_lines] if len(self.cell_lines) > 1 else []
            if len(self.timepoints_h) > 1:
                f.append(self.timepoints_h.index(c.timepoint_h) / (len(self.timepoints_h) - 1))
            rows.append(f)
        return np.asarray(rows, float)


class Designer(Protocol):
    name: str

    def propose(self, history: History, budget: int, pending: list[Condition],
                rng: np.random.Generator) -> list[Condition]:
        """Return up to `budget` conditions to run next. `pending` = already in flight."""
        ...


def build_space(cfg: dict) -> DesignSpace:
    def ladder(spec) -> tuple[float, ...]:
        # spec: {"max": 10, "steps": 7, "fold": 3} -> 0 plus a dilution series down from max
        doses = [spec["max"] / spec["fold"] ** i for i in range(spec["steps"])]
        return (0.0, *sorted(round(d, 6) for d in doses))

    specs = cfg.get("doses") or [cfg["dose"]] * cfg["n_drugs"]
    return DesignSpace(tuple(cfg["cell_lines"]), tuple(ladder(s) for s in specs), tuple(cfg["timepoints_h"]))
