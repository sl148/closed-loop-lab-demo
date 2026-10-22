"""Baseline designers: full grid, random, Latin hypercube, and grid-then-replicate-top-k.

Every designer measures the single-agent wells first: Bliss excess of a combination is only
estimable once its singles are measured. `groups` restricts proposals to the (cell line,
timepoint) groups of the plate being filled.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from labloop.design.base import DesignSpace
from labloop.evaluate import History
from labloop.types import Condition


def in_groups(c: Condition, groups) -> bool:
    return groups is None or c.group() in groups


def missing_singles(space: DesignSpace, history: History, pending: list[Condition], groups=None) -> list[Condition]:
    seen = history.tested() | set(pending)
    return [c for c in space.singles if c not in seen and in_groups(c, groups)]


class GridDesigner:
    """Walks the full grid in a fixed order (singles first)."""

    name = "grid"

    def __init__(self, space: DesignSpace, **_):
        self.queue = list(space.singles) + list(space.combos)

    def propose(self, history: History, budget: int, pending, rng, groups=None) -> list[Condition]:
        batch = [c for c in self.queue if in_groups(c, groups)][:budget]
        taken = set(batch)
        self.queue = [c for c in self.queue if c not in taken]
        return batch


class RandomDesigner:
    """Uniform over untested combinations, after the singles."""

    name = "random"

    def __init__(self, space: DesignSpace, **_):
        self.space = space

    def propose(self, history: History, budget: int, pending, rng, groups=None) -> list[Condition]:
        batch = missing_singles(self.space, history, pending, groups)[:budget]
        tested = history.tested() | set(pending)
        pool = [c for c in self.space.combos if c not in tested and in_groups(c, groups)]
        k = min(budget - len(batch), len(pool))
        if k > 0:
            batch += [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
        return batch


class LHSDesigner:
    """Latin hypercube over the dose-index cube of each group, snapped to the ladder;
    duplicates and already-tested points are dropped and refilled at random."""

    name = "lhs"

    def __init__(self, space: DesignSpace, **_):
        self.space = space

    def propose(self, history: History, budget: int, pending, rng, groups=None) -> list[Condition]:
        batch = missing_singles(self.space, history, pending, groups)[:budget]
        tested = history.tested() | set(pending)
        gs = [(cl, t) for cl in self.space.cell_lines for t in self.space.timepoints_h if in_groups(Condition(cl, (), t), groups)]
        per_group = max(1, (budget - len(batch)) // max(1, len(gs)))
        chosen: list[Condition] = []
        for cl, t in gs:
            n_group = 0
            sampler = qmc.LatinHypercube(d=self.space.n_drugs, seed=rng)
            for u in sampler.random(per_group * 2):
                doses = tuple(lad[min(int(x * len(lad)), len(lad) - 1)] for x, lad in zip(u, self.space.ladders))
                c = Condition(cl, doses, t)
                if c.is_combo and c not in tested and c not in chosen:
                    chosen.append(c)
                    n_group += 1
                if n_group >= per_group:
                    break
        batch += chosen[: budget - len(batch)]
        pool = [c for c in self.space.combos if c not in tested and c not in set(batch) and in_groups(c, groups)]
        k = min(budget - len(batch), len(pool))
        if k > 0:
            batch += [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
        return batch


class GridThenRefine:
    """Full grid first; afterwards replicate the top-k combinations by estimated excess."""

    name = "grid_then_refine"

    def __init__(self, space: DesignSpace, top_k: int = 8, replicates: int = 3, **_):
        self.space, self.top_k, self.replicates = space, top_k, replicates
        self.grid = GridDesigner(space)

    def propose(self, history: History, budget: int, pending, rng, groups=None) -> list[Condition]:
        batch = self.grid.propose(history, budget, pending, rng, groups)
        if batch:
            return batch
        ranked = [c for c, _ in history.ranked_excess() if in_groups(c, groups)]
        for cond in ranked[: self.top_k]:
            batch += [cond] * self.replicates
        return batch[:budget]
