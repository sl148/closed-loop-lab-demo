"""Plate layout strategies. A plate holds one (cell line, timepoint) group: cells are seeded
per plate and the whole plate is read at one time.

- sequential: row-major fill, controls appended at the end (edge wells used like any other)
- random: contents shuffled over all wells, spreading edge effects at random
- optimized: samples in interior wells first (edges get controls or stay empty unless the
  plate would overflow), half the vehicle controls inside so normalization sees the interior,
  and conditions sorted by dose so each 8-channel pass shares sources (fewer aspirate cycles)
"""

from __future__ import annotations

from itertools import groupby

import numpy as np

from labloop.types import CONTROL_NEG, CONTROL_POS, EMPTY, Condition, PlateMap, PlateSpec


class _Base:
    name = "base"

    def __init__(self, spec: PlateSpec, n_neg: int = 4, n_pos: int = 4, seed: int = 0):
        self.spec, self.n_neg, self.n_pos = spec, n_neg, n_pos
        self.rng = np.random.default_rng(seed)
        self.counter = 0

    @property
    def capacity(self) -> int:
        return self.spec.n_wells - self.n_neg - self.n_pos

    def _new_plate(self, line: str, tp: int, campaign: str) -> PlateMap:
        pm = PlateMap(f"{campaign or 'P'}{self.counter:03d}", self.spec, line, tp, campaign)
        self.counter += 1
        return pm

    def place(self, conditions: list[Condition], campaign: str = "") -> list[PlateMap]:
        plates = []
        key = lambda c: c.group()
        for (line, tp), group in groupby(sorted(conditions, key=key), key=key):
            group = list(group)
            cap = self.capacity
            for i in range(0, len(group), cap):
                pm = self._new_plate(line, tp, campaign)
                self.fill(pm, group[i: i + cap])
                plates.append(pm)
        return plates

    def fill(self, pm: PlateMap, chunk: list[Condition]) -> None:
        raise NotImplementedError


class SequentialLayout(_Base):
    name = "sequential"

    def fill(self, pm, chunk):
        contents = chunk + [CONTROL_NEG] * self.n_neg + [CONTROL_POS] * self.n_pos
        for idx, content in enumerate(contents):
            pm.wells[divmod(idx, self.spec.cols)] = content


class RandomLayout(_Base):
    name = "random"

    def fill(self, pm, chunk):
        contents = chunk + [CONTROL_NEG] * self.n_neg + [CONTROL_POS] * self.n_pos
        wells = [(r, c) for r in range(self.spec.rows) for c in range(self.spec.cols)]
        for w, content in zip([wells[i] for i in self.rng.permutation(len(wells))], contents):
            pm.wells[w] = content


class OptimizedLayout(_Base):
    name = "optimized"

    avoid_edges = True  # False: interior-first, but a full plate spills into the edges

    @property
    def capacity(self) -> int:
        if self.avoid_edges:
            interior = (self.spec.rows - 2) * (self.spec.cols - 2)
            return interior - self.n_neg // 2
        return self.spec.n_wells - self.n_neg - self.n_pos

    def fill(self, pm, chunk):
        s = self.spec
        # Column-major so a column (one 8-channel pass) holds consecutive, dose-sorted wells.
        interior = [(r, c) for c in range(s.cols) for r in range(s.rows) if not s.is_edge(r, c)]
        edge = [(r, c) for c in range(s.cols) for r in range(s.rows) if s.is_edge(r, c)]
        n_neg_in = self.n_neg // 2
        ordered = sorted(chunk, key=lambda c: (c.doses, c.cell_line))
        inner_slots = interior[: len(interior) - n_neg_in]
        samples_in, overflow = ordered[: len(inner_slots)], ordered[len(inner_slots):]
        for w, content in zip(inner_slots, samples_in):
            pm.wells[w] = content
        free_inner = [w for w in interior if w not in pm.wells]
        for w in free_inner[:n_neg_in]:
            pm.wells[w] = CONTROL_NEG
        # Controls spread over both edge columns, then any overflow samples, rest empty.
        left = [w for w in edge if w[1] == 0]
        right = [w for w in edge if w[1] == s.cols - 1]
        ctrl = [CONTROL_NEG] * (self.n_neg - n_neg_in) + [CONTROL_POS] * self.n_pos
        alternating = [w for pair in zip(left, right) for w in pair]
        for w, content in zip(alternating, ctrl):
            pm.wells[w] = content
        rest = [w for w in edge if w not in pm.wells]
        for w, content in zip(rest, overflow):
            pm.wells[w] = content


LAYOUTS = {"sequential": SequentialLayout, "random": RandomLayout, "optimized": OptimizedLayout}


def build_layout(name: str, spec: PlateSpec, cfg: dict | None, seed: int) -> _Base:
    cfg = cfg or {}
    layout = LAYOUTS[name](spec, cfg.get("n_neg", 4), cfg.get("n_pos", 4), seed)
    if "avoid_edges" in cfg:
        layout.avoid_edges = cfg["avoid_edges"]
    return layout
