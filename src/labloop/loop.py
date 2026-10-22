"""Campaign runner: design -> layout -> schedule -> execute -> read -> update.

One shared online `Lab` runs any number of campaigns (concurrent loops). A campaign is
`sync` (propose a round of plates, wait for all reads) or `async` (keep up to
`max_in_flight` plates going, proposing `async_plates` at a time with in-flight conditions
passed to the designer as pending).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simpy

from labloop.design.base import DesignSpace, build_space
from labloop.design.bo import ActiveLearningDesigner, BODesigner, fit_gp
from labloop.design.simple import GridDesigner, GridThenRefine, LHSDesigner, RandomDesigner
from labloop.design.throughput import ThroughputAwareBO
from labloop.evaluate import History
from labloop.execution.durations import build_durations
from labloop.execution.lab import FailureModel, Lab
from labloop.execution.reader import PlateReader
from labloop.failures.policies.rules import RulePolicy
from labloop.layout.layouts import build_layout
from labloop.oracle.noise import build_noise
from labloop.oracle.surface import Oracle, build_oracle
from labloop.schedule.cpsat import CPSATScheduler, Objective
from labloop.schedule.model import idle_time, utilization
from labloop.schedule.repair import RepairScheduler
from labloop.schedule.serial import FIFOScheduler, PriorityScheduler
from labloop.schedule.workflow import build_resources
from labloop.types import Condition, PlateMap, PlateSpec, WellRead

WELL_VOLUME_UL = 50.0

DESIGNERS = {
    "grid": GridDesigner, "random": RandomDesigner, "lhs": LHSDesigner, "grid_then_refine": GridThenRefine,
    "bo-ts": lambda s, **k: BODesigner(s, acq="ts", **k), "bo-qucb": lambda s, **k: BODesigner(s, acq="qucb", **k),
    "bo-qei": lambda s, **k: BODesigner(s, acq="qei", **k), "al": ActiveLearningDesigner,
    "throughput-bo": ThroughputAwareBO,
}


def load_deck(spec) -> dict:
    """A deck design dict, or 'path/deck_x.json' (optimized design) / 'path#baseline'."""
    if isinstance(spec, dict):
        return spec
    path, _, which = str(spec).partition("#")
    return json.loads(Path(path).read_text())[which or "best"]["design"]


def build_scheduler(name: str, cfg: dict | None = None):
    cfg = cfg or {}
    obj = Objective(**cfg.get("objective", {"makespan": 10, "flow": 1}))
    if name == "fifo":
        return FIFOScheduler()
    if name.startswith("priority"):
        return PriorityScheduler(name.split("-", 1)[1] if "-" in name else "lwr")
    if name == "cpsat":
        return CPSATScheduler(cfg.get("time_limit_s", 5.0), obj)
    if name == "repair":
        return RepairScheduler(cfg.get("repair_time_limit_s", 1.0), cfg.get("window_min", 240),
                               cfg.get("time_limit_s", 5.0),
                               Objective(obj.makespan, obj.flow, cfg.get("stability", 1)))
    raise KeyError(name)


def build_policy(name: str, cfg: dict | None = None):
    if name == "rules":
        return RulePolicy(**(cfg or {}))
    if name in ("llm", "llm-replay"):
        from labloop.failures.policies.llm import build_llm_policy

        return build_llm_policy(name, cfg or {})
    raise KeyError(name)


def true_optimum(space: DesignSpace, oracle: Oracle) -> dict:
    best, val = None, -np.inf
    for c in space.combos:
        v = oracle.true_excess(c.cell_line, c.doses, c.timepoint_h)
        if v > val:
            best, val = c, v
    return {"condition": best.to_dict(), "excess": float(val)}


@dataclass
class Campaign:
    name: str
    space: DesignSpace
    designer: object
    layout: object
    oracle: Oracle
    rng: np.random.Generator
    plates_per_round: int
    rounds: int
    mode: str = "sync"
    max_in_flight: int = 4
    async_plates: int = 1
    landscape_eval: bool = False
    history: History = field(default_factory=History)
    in_flight: dict[str, list[Condition]] = field(default_factory=dict)
    retest: list[Condition] = field(default_factory=list)
    plates_used: int = 0
    wells_used: int = 0
    reagent_nmol: float = 0.0
    progress: list[dict] = field(default_factory=list)
    landscape: list[dict] = field(default_factory=list)
    optimum: dict = field(default_factory=dict)
    done: simpy.Event | None = None
    changed: simpy.Event | None = None
    _rr: int = 0

    @property
    def plate_budget(self) -> int:
        return self.plates_per_round * self.rounds

    def pending(self) -> list[Condition]:
        return [c for conds in self.in_flight.values() for c in conds]

    def on_read(self, lab: Lab):
        def cb(pm: PlateMap, reads: list[WellRead]) -> None:
            self.history.add(reads)
            self.in_flight.pop(pm.plate_id, None)
            self.record(lab, pm.plate_id)
            if not self.changed.triggered:
                self.changed.succeed()
        return cb

    def on_retest(self, pm: PlateMap, wells: tuple[str, ...]) -> None:
        names = {pm.spec.well_name(*k): v for k, v in pm.wells.items()}
        conds = [names[w] for w in wells if w in names and isinstance(names[w], Condition)] if wells \
            else pm.conditions()
        self.retest += conds

    def record(self, lab: Lab, plate_id: str) -> None:
        best = self.history.best()
        true = self.oracle.true_excess(best[0].cell_line, best[0].doses, best[0].timepoint_h) if best else None
        self.progress.append({
            "t": int(lab.env.now), "plate": plate_id, "plates_read": self.plates_used - len(self.in_flight),
            "plates_submitted": self.plates_used, "wells": self.wells_used, "reagent_nmol": round(self.reagent_nmol, 3),
            "best": best[0].to_dict() if best else None, "best_est": best[1] if best else None,
            "best_true": true, "regret": (self.optimum["excess"] - true) if true is not None else None,
        })

    def landscape_rmse(self, n: int = 400) -> float | None:
        """RMSE of the GP's predicted excess vs the oracle on random combinations. Reuses the
        designer's GP when it has one for the current history."""
        gp = getattr(self.designer, "gp", None)
        model = gp.get(self.history, self.rng) if gp is not None else fit_gp(self.space, self.history, self.rng)[0]
        if model is None:
            return None
        import torch

        idx = self.rng.choice(len(self.space.combos), min(n, len(self.space.combos)), replace=False)
        conds = [self.space.combos[i] for i in idx]
        with torch.no_grad():
            pred = model.posterior(torch.tensor(self.space.features(conds))).mean.squeeze(-1).numpy()
        true = np.array([self.oracle.true_excess(c.cell_line, c.doses, c.timepoint_h) for c in conds])
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    def groups(self) -> list[tuple[str, int]]:
        return [(cl, t) for cl in self.space.cell_lines for t in self.space.timepoints_h]

    def _submit(self, lab: Lab, n_plates: int) -> int:
        """Fill `n_plates` plates, one (cell line, timepoint) group each. Group order comes
        from the designer when it has an opinion (throughput-aware BO), else round-robin."""
        cap = self.layout.capacity
        order = None
        if hasattr(self.designer, "choose_groups"):
            order = self.designer.choose_groups(self.history, n_plates, self.pending() + self.retest, self.rng,
                                                {"lab": lab, "campaign": self})
        if order is None:
            gs = self.groups()
            order = [gs[(self._rr + i) % len(gs)] for i in range(n_plates)]
            self._rr += n_plates
        chosen: list[Condition] = []
        plates: list[PlateMap] = []
        for g in order:
            rt = [c for c in self.retest if c.group() == g][:cap]
            for c in rt:
                self.retest.remove(c)
            conds = self.designer.propose(self.history, cap - len(rt), self.pending() + chosen + rt, self.rng,
                                          groups={g})
            conds = rt + list(conds)
            if conds:
                chosen += conds
                plates += self.layout.place(conds, campaign=self.name)
        plates = plates[: max(0, self.plate_budget - self.plates_used)]
        if not plates:
            return 0
        for pm in plates:
            self.in_flight[pm.plate_id] = pm.conditions()
            self.wells_used += len(pm.conditions())
            self.reagent_nmol += sum(sum(c.doses) for c in pm.conditions()) * WELL_VOLUME_UL * 1e-3
        self.plates_used += len(plates)
        lab.submit(plates, self.name, self.on_read(lab), self.on_retest)
        return len(plates)

    def process(self, lab: Lab):
        env = lab.env
        rnd = 0
        while self.plates_used < self.plate_budget:
            self.changed = env.event()
            if self.mode == "sync":
                if not self._submit(lab, min(self.plates_per_round, self.plate_budget - self.plates_used)):
                    break
                rnd += 1
                while self.in_flight:
                    yield self.changed
                    self.changed = env.event()
                if self.landscape_eval:
                    self.landscape.append({"t": int(env.now), "round": rnd, "plates": self.plates_used,
                                           "rmse": self.landscape_rmse()})
            else:
                room = self.max_in_flight - len(self.in_flight)
                if room >= self.async_plates or not self.in_flight:
                    if not self._submit(lab, min(self.async_plates, self.plate_budget - self.plates_used)):
                        if not self.in_flight:
                            break
                        yield self.changed
                    continue
                yield self.changed
        while self.in_flight:
            self.changed = env.event()
            yield self.changed
        if self.landscape_eval and (not self.landscape or self.landscape[-1]["plates"] != self.plates_used):
            self.landscape.append({"t": int(env.now), "round": rnd, "plates": self.plates_used,
                                   "rmse": self.landscape_rmse()})
        self.done.succeed()


def run_campaign(cfg: dict, seed: int, overrides: dict | None = None) -> dict:
    """Runs every campaign in `cfg` on one shared lab. Returns a JSON-able result dict."""
    cfg = {**cfg, **(overrides or {})}
    ss = np.random.SeedSequence(seed)
    lab_rng, *camp_seeds = ss.spawn(1 + len(cfg.get("campaigns", [{}])))
    env = simpy.Environment()
    resources = build_resources(cfg.get("resources", {}))
    durations = build_durations(cfg.get("durations"))
    if cfg.get("deck"):
        durations = dataclasses.replace(durations, deck=load_deck(cfg["deck"]))
    oracle = build_oracle(cfg["oracle"])
    reader = PlateReader(oracle, build_noise(cfg.get("noise")))
    scheduler = build_scheduler(cfg["scheduler"], cfg.get("scheduler_cfg"))
    policy = build_policy(cfg.get("policy", "rules"), cfg.get("policy_cfg"))
    failures = FailureModel(**cfg.get("failures", {}))
    if cfg.get("sim3d"):
        from labloop.sim3d.calibration import apply_sim3d

        durations, failures = apply_sim3d(cfg["sim3d"], durations, failures)
    lab = Lab(env, resources, durations, reader, scheduler, policy, np.random.default_rng(lab_rng), failures,
              plr_verify=cfg.get("plr_verify", False),
              shadow=CPSATScheduler(cfg["shadow_time_s"], Objective(10, 1)) if cfg.get("shadow_time_s") else None)

    spec = PlateSpec(**cfg.get("plate", {}))
    campaigns = []
    for i, ccfg in enumerate(cfg.get("campaigns", [{}])):
        ccfg = {**{k: cfg[k] for k in ("space", "designer", "rounds", "plates_per_round", "mode", "max_in_flight",
                                       "async_plates", "layout", "designer_cfg", "landscape_eval") if k in cfg},
                **ccfg}
        space = build_space(ccfg["space"])
        rng = np.random.default_rng(camp_seeds[i])
        layout = build_layout(ccfg.get("layout", "sequential"), spec, cfg.get("layout_cfg"), int(rng.integers(2**31)))
        designer = DESIGNERS[ccfg["designer"]](space, **ccfg.get("designer_cfg", {}))
        c = Campaign(ccfg.get("name", f"C{i}"), space, designer, layout, oracle, rng, ccfg.get("plates_per_round", 1),
                     ccfg.get("rounds", 4), ccfg.get("mode", "sync"), ccfg.get("max_in_flight", 4),
                     ccfg.get("async_plates", 1), ccfg.get("landscape_eval", False))
        c.optimum = true_optimum(space, oracle)
        c.done = env.event()
        campaigns.append(c)
        env.process(c.process(lab))
    env.run(until=simpy.AllOf(env, [c.done for c in campaigns]))

    executed = lab.executed()
    prob = _executed_problem(lab)
    return {
        "name": cfg["name"], "seed": seed, "scheduler": cfg["scheduler"], "policy": cfg.get("policy", "rules"),
        "makespan": executed.makespan,
        "utilization": utilization(prob, executed) if executed.start else {},
        "idle": idle_time(prob, executed) if executed.start else {},
        "campaigns": {c.name: {"designer": getattr(c.designer, "name", ""), "mode": c.mode, "optimum": c.optimum,
                               "plates": c.plates_used, "wells": c.wells_used, "reagent_nmol": round(c.reagent_nmol, 3),
                               "progress": c.progress, "landscape": c.landscape,
                               "group_decisions": getattr(c.designer, "decisions", []),
                               "final_best": c.progress[-1] if c.progress else None} for c in campaigns},
        "reschedules": lab.reschedules,
        "events": lab.events,
        "schedule": [{"op": o, "job": lab.ops[o].job, "kind": lab.ops[o].kind, "resource": executed.assigned[o],
                      "start": executed.start[o], "end": executed.end[o]} for o in executed.start],
        "plates": {pid: _plate_summary(lab, pm) for pid, pm in lab.plates.items()},
        "resources": {r: res.capacity for r, res in resources.items()},
        "invalid_plates": sum(1 for s in lab.plate_status.values() if s == "invalid"),
        "deck": durations.deck,
    }


def _executed_problem(lab: Lab):
    from labloop.schedule.model import Problem

    prob = Problem(resources=lab.resources)
    for oid, r in lab.run.items():
        if r.status == "done":
            prob.add_op(lab.ops[oid])
    return prob


def _plate_summary(lab: Lab, pm: PlateMap) -> dict:
    reads = {w["well"]: w for w in lab.plate_reads.get(pm.plate_id, [])} if hasattr(lab, "plate_reads") else {}
    return {"campaign": pm.campaign, "cell_line": pm.cell_line, "timepoint_h": pm.timepoint_h,
            "rows": pm.spec.rows, "cols": pm.spec.cols, "status": lab.plate_status[pm.plate_id],
            "wells": {pm.spec.well_name(r, c): (v.to_dict() if isinstance(v, Condition) else v)
                      for (r, c), v in sorted(pm.wells.items())},
            "reads": reads}
