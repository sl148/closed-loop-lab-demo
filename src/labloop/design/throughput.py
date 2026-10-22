"""Throughput-aware batch BO (contribution B).

A plate serves one (cell line, timepoint) group, so the batch decision is really "which
groups get the next plates", and groups differ in how long their answer takes: a 72 h plate
returns two days after a 24 h plate, and a busy incubator or reader delays everything. This
designer scores each group by

    value(g) / hours_to_result(g)

where value(g) is the summed optimistic improvement (UCB minus the best posterior mean at
tested points, clipped at 0, plus a small std tie-break) of the
best plate-full of candidates in g, and hours_to_result(g) comes from inserting a
hypothetical plate of g into the lab's live plan with the fast serial scheduler. This is
improvement-per-unit-cost (cost-aware BO) with the cost measured by the scheduler, not
assumed. An unvisited group is scored by the GP's extrapolation from visited ones (cell
line and timepoint are features), so the designer may skip groups not worth a plate.
Groups are picked greedily, each pick added to the probe plan; wells inside a plate
are filled by Thompson sampling with in-flight conditions as fantasies.
"""

from __future__ import annotations

import numpy as np
import torch

from labloop.design.base import DesignSpace
from labloop.design.bo import BODesigner, candidate_pool
from labloop.evaluate import History
from labloop.schedule.serial import FIFOScheduler
from labloop.schedule.workflow import plate_job
from labloop.types import Condition, PlateMap


class ThroughputAwareBO(BODesigner):
    def __init__(self, space: DesignSpace, beta: float = 2.0, diminish: float = 0.5, **_):
        super().__init__(space, acq="ts")
        self.ucb_beta, self.diminish = beta, diminish
        self.name = "throughput-bo"
        self.decisions: list[dict] = []

    def group_values(self, model, history: History, rng, capacity: int) -> dict[tuple, float]:
        # Incumbent: best posterior mean at tested points (stable, unlike a noisy max).
        tested = sorted(history.excess_estimates())
        with torch.no_grad():
            incumbent = float(model.posterior(torch.tensor(self.space.features(tested))).mean.max())
        out = {}
        for cl in self.space.cell_lines:
            for t in self.space.timepoints_h:
                pool = candidate_pool(self.space, history, rng, {(cl, t)})
                if not pool:
                    continue
                with torch.no_grad():
                    post = model.posterior(torch.tensor(self.space.features(pool)))
                    ucb = (post.mean.squeeze(-1) + self.ucb_beta * post.variance.squeeze(-1).sqrt()).numpy()
                std = post.variance.squeeze(-1).sqrt().numpy()
                gain = np.clip(ucb - incumbent, 0.0, None) + 0.01 * std  # std term breaks ties
                out[(cl, t)] = float(np.sort(gain)[::-1][:capacity].sum())
        return out

    def hours_to_result(self, lab, probe_plates: list[PlateMap], group: tuple, capacity: int) -> float:
        prob, _ = lab.build_problem()
        now = int(lab.env.now)
        for pm in probe_plates:
            plate_job(prob, pm, lab.durations, now)
        cl, t = group
        dummy = PlateMap(f"__probe{len(probe_plates)}", probe_plates[0].spec if probe_plates else self._spec,
                         cl, t)
        for k in range(capacity):
            dummy.wells[divmod(k, dummy.spec.cols)] = Condition(cl, (1.0,) * self.space.n_drugs, t)
        plate_job(prob, dummy, lab.durations, now)
        sched = FIFOScheduler().schedule(prob)
        return max(1.0, (sched.end[f"{dummy.plate_id}.read"] - now) / 60.0), dummy

    def choose_groups(self, history: History, n_plates: int, pending, rng, ctx) -> list[tuple] | None:
        model = self.gp.get(history, rng)
        if model is None:
            return None  # cold start: the campaign's round-robin also measures the singles
        campaign, lab = ctx["campaign"], ctx["lab"]
        cap = campaign.layout.capacity
        self._spec = campaign.layout.spec
        values = self.group_values(model, history, rng, cap)
        order, probes = [], []
        for _ in range(n_plates):
            scored = {}
            for g, v in values.items():
                hours, _ = self.hours_to_result(lab, probes, g, cap)
                scored[g] = (v / hours, v, hours)
            g = max(scored, key=lambda k: (scored[k][0], k))
            order.append(g)
            _, dummy = self.hours_to_result(lab, probes, g, cap)
            probes.append(dummy)
            values[g] *= self.diminish
            self.decisions.append({"t": int(lab.env.now), "chosen": list(g),
                                   "scores": {f"{k[0]}@{k[1]}h": [round(s[1], 4), round(s[2], 2)]
                                              for k, s in scored.items()}})
        return order
