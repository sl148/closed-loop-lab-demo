"""Online lab: executes the current plan in SimPy, detects deviations and failures, and asks
the scheduler to repair the plan.

- Ops start at max(planned start, predecessor ready + min lag) on their planned device.
- A flexible op (incubation) is ready after dur_min and keeps its slot until the successor
  starts; its end is the successor's start.
- Reschedule triggers: new batch, an overrun that threatens the plan, a blocked dispatch,
  a failure decision that asks for it. Ops already finished leave the problem; running ops
  are pinned. The scheduler sees only planned/estimated durations, never sampled actuals.
- Failures come from `FailureModel`; a `RecoveryPolicy` turns each into actions.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import simpy

from labloop.constraints.compile import Compiled
from labloop.execution.durations import DurationModel
from labloop.execution.reader import PlateReader
from labloop.execution.truth import Truth, z_factor
from labloop.failures.events import ActionType as A
from labloop.failures.events import (Decision, FailureEvent, FailureType as F, LabState, RecoveryAction,
                                     Severity)
from labloop.schedule.model import Lag, Op, Problem, Resource, Schedule, Span
from labloop.schedule.serial import FIFOScheduler
from labloop.schedule.validator import violations
from labloop.schedule.workflow import plate_job
from labloop.types import CONTROL_NEG, CONTROL_POS, PlateMap, WellRead

MID_SWAP_MIN = 2  # arm swaps all racks during a protocol: two moves plus re-teach check
ARM_SITES = 2  # tip-carrier sites the plate arm reaches (front two; MuJoCo: 0.42 / 0.52 m of 0.59 m)
STALL_REPLANS = 2000  # replans in a row without any op starting: stop the run instead of spinning
SUPPLY_RETRIES = 6  # fresh batches / reservoirs one plate may wait through before the run stops as a livelock
CORE_MOVE_MIN = 1  # LH CO-RE gripper moves a tip box between a deep site and the handoff site
OVERRUN_STEP = 0.25  # re-estimate an overrunning op as elapsed + this fraction of nominal


@dataclass
class FailureModel:
    duration_cv: float = 0.0  # fixed ops: actual = nominal * U(1 - cv, 1 + cv)
    p_tip_pickup: float = 0.0  # per dispense
    p_liquid_transfer: float = 0.0  # per dispense
    p_misplace: float = 0.0  # per arm transfer
    p_drop: float = 0.0  # per arm transfer carrying a plate
    # A re-placement (arm re-grips at the camera's estimate and sets the plate down again) fails with
    # this probability; None = same as a first placement. Measured in MuJoCo (sim3d/replace.json).
    p_replace_fail: float | None = None
    p_measurement: float = 0.0  # per read
    breakdown_per_day: float = 0.0  # per device (not incubators)
    breakdown_kinds: tuple[str, ...] = ("lh", "reader", "arm")
    wells_per_failure: int = 4
    # Camera perception (sim3d): a misplaced plate is off by a Rayleigh(sigma) offset and is
    # detected with the camera's measured rate at that offset; an undetected misplacement
    # silently corrupts the wells pipetted or read off-target. Empty curve = always detected.
    misplace_sigma_mm: float = 1.5
    # Physical costs of recovery actions, set by the world, not by the policy that picks them
    # (a policy cannot make a repair faster by asking for fewer minutes).
    retry_min: dict = field(default_factory=lambda: {"tip_pickup": 1, "plate_misplaced": 2})
    substitute_min: int = 1
    human_min: int = 30
    repair_min: int = 90
    perception_curve: tuple[tuple[float, float], ...] = ()


@dataclass
class OpRun:
    status: str = "pending"  # pending | running | ready | done
    start: int | None = None
    ready_at: int | None = None
    end: int | None = None
    resource: str | None = None
    extra: int = 0  # minutes added by recovery actions
    cancelled: bool = False  # the plate was lost before this op ran
    est_dur: int | None = None  # scheduler-facing estimate while overrunning


class Lab:
    def __init__(self, env: simpy.Environment, resources: dict[str, Resource], durations: DurationModel,
                 reader: PlateReader, scheduler, policy, rng: np.random.Generator,
                 failures: FailureModel | None = None, plr_verify: bool = False, shadow=None,
                 truth: Truth | None = None, constraints: Compiled | None = None, supplies=None):
        self.env, self.resources, self.durations = env, resources, durations
        self.reader, self.scheduler, self.policy, self.rng = reader, scheduler, policy, rng
        self.failures = failures or FailureModel()
        self.truth = truth or Truth()  # physical limits the scheduler is not told about
        # Liquids on the deck (execution/supplies.py): dilution batches and cell reservoirs are made and
        # swapped by jobs the lab orders, like tip racks. None: the old model (sources never run out).
        self.supplies = supplies
        self.plr = None  # execution/plr_bridge.PLRLab: the LH's work runs on PyLabRobot (set by loop.run_campaign)
        self.supply_job: dict[tuple[str, str], str | None] = {}  # (lh, "prep" | "cells") -> pending job
        self._supply_waits: dict[str, set] = {}  # plate -> the distinct supplies it found short (livelock guard)
        self.prep_ul: dict[str, float] = {}  # lh -> per-dose volume the next batch needs (above the plan's)
        self.supply_line: dict[str, str] = {}  # CELLS job -> cell line it brings
        self.batch_of: dict[str, int] = {}  # plate -> when the dilution batch it got was made
        self.n_supply_jobs = 0
        self.lh_line: dict[str, str] = {}  # liquid handler -> cell line it last seeded
        # Constraints the lab was given (constraints/compile.py), applied to every plate it accepts.
        self.cons = constraints or Compiled()
        self.setup_min = self.cons.setup_min
        self.cons_version = 0  # bumped by update_constraints; each plate records the version it got
        self.constraint_history = [{"t": 0, "version": 0, "specs": [c.model_dump() for c in self.cons.specs],
                                    "reason": "initial", "review_key": None}]
        self.plate_version: dict[str, int] = {}
        self.family: dict[str, str] = {}  # op -> family (setups)
        self.res_last: dict[str, tuple[str, int]] = {}  # resource -> (family, end) of its last setup op
        self.plr_verify = plr_verify
        self.shadow = shadow
        self.ops: dict[str, Op] = {}
        self.lags: list[Lag] = []
        # Deck zones: an LH's plate site and gantry area, held by one span (load .. unload) at a time.
        self.spans: dict[str, Span] = {}  # any op of a span -> span
        self.zone: dict[str, Span | None] = {r: None for r in resources}
        self.zone_res: dict[Span, str] = {}
        self.release: dict[str, int] = {}
        self.start_ub: dict[str, int] = {}
        self.run: dict[str, OpRun] = {}
        self.plates: dict[str, PlateMap] = {}
        self.plate_status: dict[str, str] = {}
        self.bad_wells: dict[str, set[str]] = {}  # invalidated: excluded from the data
        self.corrupt: dict[str, set[str]] = {}  # failed wells: wrong values unless invalidated
        self.plate_reads: dict[str, list[dict]] = {}
        # Tips as a consumable (only with a deck design): inventory per liquid handler and
        # tip type; a rack-swap job (arm brings full racks from the rack hotel) is ordered at a
        # low-water mark or when a dispense would run short.
        self.tip_cap: dict[str, int] = {}
        self.racks: dict[str, dict[str, list[int]]] = {}  # lh -> tip type -> tips left per rack
        self.rack_sites: dict[str, list[int]] = {}  # tip type -> tip-carrier sites (PLR fill order)
        self.tip_order: dict[str, str | None] = {}
        self.tip_racks: dict[str, list[tuple[str, int]]] = {}  # TIPS job -> (tip type, rack index) it replaces
        self.n_tip_jobs = 0
        self.on_read: dict[str, Callable[[PlateMap, list[WellRead]], None]] = {}
        self.on_retest: dict[str, Callable[[PlateMap, tuple[str, ...]], None]] = {}
        self.on_quarantine: dict[str, Callable[[PlateMap], int]] = {}  # campaign forgets a plate's data and re-runs it
        self.plan = Schedule()
        self.down: dict[str, list[tuple[int, int]]] = {r: [] for r in resources}
        self.occupancy: dict[str, set[str]] = {r: set() for r in resources}
        self.pending_reasons: list[str] = []
        self.reschedules: list[dict] = []
        self.events: list[dict] = []
        self.attempts: dict[tuple, int] = {}
        self.wake = env.event()
        self.idle = env.event()
        self.env.process(self._controller())
        if self.failures.breakdown_per_day > 0:
            for r in resources:
                if r.rstrip("0123456789") in self.failures.breakdown_kinds:
                    self.env.process(self._breakdowns(r))

    # ------------------------------------------------------------------ public
    def submit(self, plates: list[PlateMap], campaign: str, on_read, on_retest, on_quarantine=None) -> None:
        prob = Problem(resources=self.resources)
        for pm in plates:
            ops = plate_job(prob, pm, self.durations, release=int(self.env.now))
            self.cons.apply(prob, pm)
            for op in ops:
                self.run[op.id] = OpRun()
            self.plates[pm.plate_id] = pm
            self.plate_status[pm.plate_id] = "queued"
            self.plate_version[pm.plate_id] = self.cons_version
            self.on_read[pm.plate_id], self.on_retest[pm.plate_id] = on_read, on_retest
            if on_quarantine:
                self.on_quarantine[pm.plate_id] = on_quarantine
        self.ops.update(prob.ops)
        self.lags += prob.lags
        self._add_spans(prob.spans)
        self.release.update(prob.release)
        self.start_ub.update(prob.start_ub)
        self.family.update(prob.family)
        self._log("submit", campaign=campaign, plates=[p.plate_id for p in plates])
        self.request("new_batch")

    def update_constraints(self, compiled: Compiled, scope: str = "queued", reason: str = "",
                           review_key: str | None = None) -> dict:
        """A new constraint set mid-run (an approved review). Plates submitted later get it; with
        scope="queued" so do the ops not yet started of plates already in the lab: their constraint
        lags, start deadlines and incubation caps are rebuilt from the new set. Changeover families
        are rebuilt for every plate, so a cleaning limit counts from dispenses already done."""
        old, new = {c.id: c for c in self.cons.specs}, {c.id: c for c in compiled.specs}
        self.cons, self.setup_min = compiled, compiled.setup_min
        self.cons_version += 1
        rederived = []
        for pid, pm in self.plates.items():
            layer = self._plate_layer(pm, compiled)
            for oid in [o for o in self.family if o.startswith(pid + ".") and o not in layer.family]:
                del self.family[oid]
            self.family.update(layer.family)
            pending = {o for o in layer.ops if o in self.run and self.run[o].status == "pending"}
            if scope != "queued" or not pending:
                continue
            mine = lambda l: l.after in pending and l.before.startswith(pid + ".")
            self.lags = [l for l in self.lags if not mine(l)] + [l for l in layer.lags if mine(l)]
            for oid in pending:
                if oid in layer.start_ub:
                    self.start_ub[oid] = layer.start_ub[oid]
                else:
                    self.start_ub.pop(oid, None)
                if self.ops[oid].kind == "incubate":
                    self.ops[oid] = dataclasses.replace(self.ops[oid], dur_max=layer.ops[oid].dur_max)
            self.plate_version[pid] = self.cons_version
            rederived.append(pid)
        self.res_last = {}  # last setup op per resource, from what already ran
        for oid, r in self.run.items():
            if r.status == "done" and not r.cancelled and oid in self.family and self.ops[oid].kind in self.setup_min \
                    and (r.resource not in self.res_last or r.end >= self.res_last[r.resource][1]):
                self.res_last[r.resource] = (self.family[oid], r.end)
        info = {"version": self.cons_version, "added": sorted(new.keys() - old.keys()),
                "removed": sorted(old.keys() - new.keys()),
                "changed": [{"id": k, "from": old[k].minutes, "to": new[k].minutes}
                            for k in sorted(old.keys() & new.keys()) if old[k].minutes != new[k].minutes],
                "scope": scope, "plates_rederived": len(rederived), "reason": reason, "review_key": review_key}
        self.constraint_history.append({"t": int(self.env.now), "version": self.cons_version,
                                        "specs": [c.model_dump() for c in compiled.specs], "reason": reason,
                                        "review_key": review_key})
        self._log("constraints_update", **info)
        self.request("constraints_update")
        return info

    def _plate_layer(self, pm: PlateMap, compiled: Compiled) -> Problem:
        """A plate's job with `compiled` applied, as if it were submitted now with its own release."""
        prob = Problem(resources=self.resources)
        plate_job(prob, pm, self.durations, release=self.release[pm.plate_id])
        compiled.apply(prob, pm)
        return prob

    def _add_spans(self, spans: list[Span]) -> None:
        for sp in spans:
            for oid in (sp.first, sp.link, sp.last, *sp.inner):
                self.spans[oid] = sp

    def request(self, reason: str) -> None:
        self.pending_reasons.append(reason)
        self._poke()

    def busy(self) -> bool:
        return any(r.status != "done" for r in self.run.values())

    def executed(self) -> Schedule:
        s = Schedule(solver="executed")
        for oid, r in self.run.items():
            if r.status == "done" and not r.cancelled:
                s.start[oid], s.end[oid], s.assigned[oid] = r.start, r.end, r.resource
        return s

    # ------------------------------------------------------------------ helpers
    def _poke(self) -> None:
        if not self.wake.triggered:
            self.wake.succeed()

    def _log(self, kind: str, **kw) -> None:
        self.events.append({"t": int(self.env.now), "kind": kind, **kw})

    def _preds(self, oid: str) -> list[Lag]:
        return [l for l in self.lags if l.after == oid]

    def _succ(self, oid: str) -> Lag | None:
        """The chain successor's lag (start-anchored constraint lags are not hand-overs)."""
        return next((l for l in self.lags if l.before == oid and l.ref == "end"), None)

    def _anchor(self, lag: Lag) -> int | None:
        r = self.run[lag.before]
        if lag.ref == "start":
            return r.start
        # A finished op's real end: an incubation ends when the plate leaves the incubator, not
        # when its time is up (the plate may wait inside for the arm). The plan counts the same way.
        return r.end if r.status == "done" and r.end is not None else r.ready_at

    def _nominal(self, oid: str, res: str) -> int:
        return self.ops[oid].dur_on(res)

    def _is_down(self, res: str, t: int) -> bool:
        return any(a <= t < b for a, b in self.down[res])

    # ------------------------------------------------------------------ rescheduling
    def build_problem(self) -> tuple[Problem, Schedule]:
        now = int(self.env.now)
        prob = Problem(resources=self.resources)
        live = [o for o, r in self.run.items() if r.status != "done"]
        live_set = set(live)
        for oid in live:
            op, r = self.ops[oid], self.run[oid]
            if r.status in ("running", "ready"):
                prob.pinned[oid] = r.resource
                prob.start_lb[oid] = prob.start_ub[oid] = r.start
                if op.fixed:
                    est = r.est_dur or (self._nominal(oid, r.resource) + r.extra)
                    prob.dur_override[oid] = max(est, now - r.start + 1)
                else:
                    if now > r.start + op.dur_hi:  # window already blown; let it end now
                        op = dataclasses.replace(op, dur_max=now - r.start + 1)
                    prob.end_lb[oid] = max(now, r.start + op.dur_min)
            else:
                prob.start_lb[oid] = now
                if oid in self.start_ub and self.start_ub[oid] >= now:
                    prob.start_ub[oid] = self.start_ub[oid]
            prob.add_op(op)
            prob.release[op.job] = min(prob.release.get(op.job, now), self.release[op.job])
        for lag in self.lags:
            if lag.after not in live_set:
                continue
            if lag.before in live_set:
                prob.lags.append(lag)
                continue
            before = self.run[lag.before]
            if self.run[lag.after].status == "pending":
                t = before.start if lag.ref == "start" else before.end
                lo = t + lag.min_lag
                prob.start_lb[lag.after] = max(prob.start_lb[lag.after], lo)
                if lag.max_lag is not None and t + lag.max_lag >= prob.start_lb[lag.after]:
                    prob.start_ub[lag.after] = min(prob.start_ub.get(lag.after, 10**9), t + lag.max_lag)
        for sp in dict.fromkeys(self.spans.values()):
            if sp.last not in live_set:
                continue
            if self.run[sp.first].status == "pending":
                prob.spans.append(sp)
                continue
            held = self.zone_res[sp]  # the plate is on this LH's deck: keep it there
            prob.spans.append(dataclasses.replace(sp, res=held))
            if sp.first not in live_set:
                prob.span_start[sp.link] = self.run[sp.first].start
            if sp.link in live_set and sp.link not in prob.pinned:
                prob.fixed_res[sp.link] = held
        prob.family = {o: f for o, f in self.family.items() if o in live_set}
        prob.setup_min = dict(self.setup_min)
        prob.res_last = dict(self.res_last)
        prob.unavailable = {r: [(max(a, now), b) for a, b in w if b > now] for r, w in self.down.items() if w}
        prob.unavailable = {r: w for r, w in prob.unavailable.items() if w}
        prior = Schedule(solver="prior")
        for oid in live:
            if oid in self.plan.start:
                prior.start[oid], prior.end[oid] = self.plan.start[oid], self.plan.end[oid]
                prior.assigned[oid] = self.plan.assigned[oid]
        return prob, prior

    @staticmethod
    def _objective(prob: Problem, sched: Schedule) -> int:
        """makespan*10 + sum of job completions (the online objective, without stability)."""
        if not sched.start:
            return -1
        flow = sum(max(sched.end[o.id] for o in prob.job_ops(j)) for j in prob.jobs())
        return 10 * sched.makespan + flow

    def _reschedule(self) -> None:
        if len(self.reschedules) - getattr(self, "_progress", 0) > STALL_REPLANS:  # replanning with nothing starting: a bug
            raise RuntimeError(f"lab stalled at t={int(self.env.now)}: {STALL_REPLANS} replans and no op started")
        reasons, self.pending_reasons = self.pending_reasons, []
        prob, prior = self.build_problem()
        if not prob.ops:
            return
        sched = self.scheduler.schedule(prob, prior=prior if prior.start else None, now=int(self.env.now))
        errs = violations(prob, sched) if sched.start else ["no solution"]
        # Lag slips (a plate waiting longer than its max lag) are tolerated and logged.
        soft = lambda e: "->" in e or "after its upper bound" in e
        hard = [e for e in errs if not soft(e)]
        if hard:
            fb = FIFOScheduler().schedule(prob)
            fb.solver = f"{sched.solver}->fifo"
            sched, errs = fb, violations(prob, fb)
        moved = sum(1 for o in prior.start if o in sched.start and o not in prob.pinned
                    and sched.start[o] != prior.start[o])
        self.reschedules.append({
            "t": int(self.env.now), "reasons": sorted(set(reasons)), "solver": sched.solver,
            "status": sched.status, "time_s": round(sched.solve_time_s, 4), "n_ops": len(prob.ops),
            "n_free": len(prob.ops) - len(prob.pinned), "moved": moved,
            "est_makespan": sched.makespan, "lag_slips": sum(1 for e in errs if soft(e)),
        })
        if self.shadow is not None and prior.start:
            # Quality reference: a long from-scratch solve of the same problem.
            ref = self.shadow.schedule(prob)
            self.reschedules[-1].update(shadow_makespan=ref.makespan, shadow_time_s=round(ref.solve_time_s, 4),
                                        shadow_obj=self._objective(prob, ref), obj=self._objective(prob, sched))
        self.plan = sched

    # ------------------------------------------------------------------ dispatch
    def _ready_to_start(self, oid: str, now: int) -> bool:
        if self.run[oid].status != "pending" or oid not in self.plan.start:
            return False
        if now < self.plan.start[oid]:
            return False
        for lag in self._preds(oid):
            pr = self.run[lag.before]
            if pr.status not in ("ready", "done") or now < self._anchor(lag) + lag.min_lag:
                return False
        res = self.plan.assigned[oid]
        if self._is_down(res, now):
            return False
        sp = self.spans.get(oid)
        if sp and oid == sp.first:  # the arm enters the deck zone: the zone must be free
            lh = self.plan.assigned.get(sp.link)
            if lh is None or self.zone[lh] is not None or self._is_down(lh, now):
                return False
            if self.ops[sp.link].kind == "dispense" and self.durations.deck and not self._tips_ok(lh, self.ops[oid].job):
                return False
            if self.ops[sp.link].kind == "dispense" and self.supplies and not self._supplies_ok(lh, self.ops[oid].job, now):
                return False
        if sp and (oid == sp.link or oid in sp.inner):  # runs inside its own held zone
            return self.zone[res] is sp
        return len(self.occupancy[res]) < self.resources[res].capacity

    def _dispatch(self) -> bool:
        now = int(self.env.now)
        started = False
        for oid in sorted((o for o, r in self.run.items() if r.status == "pending" and o in self.plan.start),
                          key=lambda o: (self.plan.start[o], o)):
            if self._ready_to_start(oid, now):
                self._start(oid, now)
                started = True
        return started

    def _start(self, oid: str, now: int) -> None:
        r = self.run[oid]
        r.status, r.start, r.resource = "running", now, self.plan.assigned[oid]
        self._progress = len(self.reschedules)
        self.occupancy[r.resource].add(oid)
        sp = self.spans.get(oid)
        if sp and oid == sp.first:
            lh = self.plan.assigned[sp.link]
            self.zone[lh], self.zone_res[sp] = sp, lh
        for lag in self._preds(oid):
            pr = self.run[lag.before]
            if pr.status == "ready":  # flexible predecessor hands the plate over now
                self._finish(lag.before, now)
            t = pr.start if lag.ref == "start" else pr.end
            if lag.max_lag is not None and t is not None and now - t > lag.max_lag:
                self._lag_slip(lag, now - t)
        op = self.ops[oid]
        if op.kind == "dispense" and self.durations.deck:
            self._use_tips(r.resource, op.job)
        if op.kind == "dispense" and self.supplies:
            self._use_supplies(r.resource, op.job, now)
        if op.kind == "dispense" and self.plr:
            self._plr_plate(r.resource, op.job)
        if op.id.endswith(".tip_swap"):
            self._log("tip_swap", lh=r.resource, job=op.job)
        if op.kind == "dispense":
            self.plate_status[op.job] = "dispensing"
            self._changeover(r.resource, op.job)
            self._carryover(op.job)
            if self.plr_verify:
                self._verify_protocol(op.job)
            if oid in self.start_ub and now > self.start_ub[oid]:
                self._failure(FailureEvent(F.TIME_WINDOW, now, plate=op.job, op=oid, severity=Severity.HIGH,
                                           detail={"why": "reagent expired before dispense"}))
        self.env.process(self._run_op(oid))

    # ------------------------------------------------------------------ tips
    def _init_tips(self, lh: str, need: dict[str, int]) -> None:
        from labloop.protocol.optimizer import rack_split

        if not self.tip_cap:
            racks = self.durations.deck.get("racks_on_deck", 5)
            split = rack_split(need, racks)
            self.tip_cap = {t: 96 * n for t, n in split.items()}
            site = 0
            for t, n in split.items():  # same order the PLR bridge fills TIP_CAR_480_A00
                self.rack_sites[t] = list(range(site, site + n))
                site += n
        if lh not in self.racks:
            self.racks[lh] = {t: [96] * len(s) for t, s in self.rack_sites.items()}
            self.tip_order[lh] = None

    def _have(self, lh: str, t: str) -> int:
        return sum(self.racks[lh].get(t, []))

    def _tips_ok(self, lh: str, job: str) -> bool:
        need = self.durations.tips(self.plates[job])
        self._init_tips(lh, need)
        # A plate needing more tips than the deck holds starts on full racks and swaps mid-run.
        if all(self._have(lh, t) >= min(n, self.tip_cap.get(t, n)) for t, n in need.items()):
            return True
        self._order_tips(lh, reason="short", before=f"{job}.mv_hotel_lh")
        return False

    def _use_tips(self, lh: str, job: str) -> None:
        need = self.durations.tips(self.plates[job])
        self._init_tips(lh, need)
        mid = 0
        for t, n in need.items():  # tips come from the first non-empty rack in site order
            racks = self.racks[lh].setdefault(t, [96])
            while n > 0:
                j = next((j for j, c in enumerate(racks) if c > 0), None)
                if j is None:  # every rack of this type is empty: the arm swaps them all mid-run
                    racks[:] = [96] * len(racks)
                    mid += 1
                    continue
                take = min(n, racks[j])
                racks[j] -= take
                n -= take
        if mid:
            self.run[f"{job}.dispense"].extra += mid * MID_SWAP_MIN
            self._log("tip_mid_swap", lh=lh, job=job, swaps=mid)
        # Low-water mark: not enough left for another plate like this one.
        if any(self._have(lh, t) < n for t, n in need.items()):
            self._order_tips(lh, reason="low")

    def _order_tips(self, lh: str, reason: str, before: str | None = None) -> None:
        if self.tip_order.get(lh):
            if before and not self.problem_has_lag(f"{self.tip_order[lh]}.tip_swap", before):
                self.lags.append(Lag(f"{self.tip_order[lh]}.tip_swap", before, 0, None))
                self.request("tip_wait")
            return
        self.n_tip_jobs += 1
        job = f"TIPS{self.n_tip_jobs:03d}_{lh}"
        arm = tuple(r for r in self.resources if r.startswith("arm"))
        # Racks to replace: every rack that is empty or partly used, each at its own site.
        left = {t: list(c) for t, c in self.racks[lh].items()}
        swap = [(t, j) for t, c in left.items() for j in sorted(range(len(c)), key=lambda j: c[j]) if c[j] < 96]
        swap = swap or [(next(iter(left)), 0)]
        self.tip_racks[job] = swap
        # Front sites: the arm takes the used box into the rack waste bin and seats a full box from the
        # rack hotel in the same site. Deep sites are out of the arm's reach: the LH CO-RE gripper moves
        # the box to a free handoff site on the plate carrier and back. The LH then registers the racks.
        ops, inner = [], []
        for t, j in swap:
            s = self.rack_sites[t][j]
            if s < ARM_SITES:
                ops += [Op(f"{job}.mv_lh_rackwaste_s{s}", job, "transfer", arm, self.durations.arm("lh>rackwaste")),
                        Op(f"{job}.mv_rackhotel_lh_s{s}", job, "transfer", arm, self.durations.arm("rackhotel>lh"))]
            else:
                core = [Op(f"{job}.core_out_s{s}", job, "tip_swap", (lh,), CORE_MOVE_MIN),
                        Op(f"{job}.core_in_s{s}", job, "tip_swap", (lh,), CORE_MOVE_MIN)]
                ops += [core[0],
                        Op(f"{job}.mv_handoff_rackwaste_s{s}", job, "transfer", arm, self.durations.arm("handoff>rackwaste")),
                        Op(f"{job}.mv_rackhotel_handoff_s{s}", job, "transfer", arm, self.durations.arm("rackhotel>handoff")),
                        core[1]]
                inner += [o.id for o in core]
        ops.append(Op(f"{job}.tip_swap", job, "tip_swap", (lh,), 1))
        now = int(self.env.now)
        for op in ops:
            self.ops[op.id] = op
            self.run[op.id] = OpRun()
        for a, b in zip(ops, ops[1:]):
            self.lags.append(Lag(a.id, b.id, 0, self.durations.transfer_max_lag_min))
        self._add_spans([Span(ops[0].id, ops[-1].id, ops[-1].id, inner=tuple(inner))])
        if before:  # the plate that ran short waits for the fresh racks
            self.lags.append(Lag(f"{job}.tip_swap", before, 0, None))
        self.release[job] = now
        self.tip_order[lh] = job
        self._log("tip_order", lh=lh, job=job, reason=reason, left={t: sum(c) for t, c in left.items()},
                  sites=[self.rack_sites[t][j] for t, j in swap])
        self.request("tip_resupply")

    # ------------------------------------------------------------------ liquids (execution/supplies.py)
    def _supplies_ok(self, lh: str, job: str, now: int) -> bool:
        pm, S = self.plates[job], self.supplies
        ok_d, ok_c = S.drug_ok(lh, pm, now + 2), S.cells_ok(lh, pm)
        if not (ok_d and ok_c):  # a plate that no fresh batch or reservoir can serve would reorder forever
            seen = self._supply_waits.setdefault(job, set())
            key = (lh, "d" if not ok_d else "c", (S.batch.get(lh).made if S.batch.get(lh) else -1) if not ok_d
                   else (S.cells.get(lh) or {}).get("left_ml"))
            seen.add(key)
            if len(seen) > SUPPLY_RETRIES:
                n, b = S.need(pm), S.batch.get(lh)
                short = {k: (v, round(b.left.get(k, 0), 1)) for k, v in n["drug"].items() if b and b.left.get(k, 0) - 20 < v}
                raise RuntimeError(f"supply livelock at t={now}: {job} on {lh} still blocked after {len(seen)} supplies; "
                                   f"drug need/left {short}, cells need {n['cells_ml']:.2f} ml, seated {S.cells.get(lh)}")
        if not ok_d:
            want = S.batch_ul(pm)
            if want > S.plan.ul_per_well:  # the plan's batch cannot serve this plate: the next batch is made bigger
                self.prep_ul[lh] = max(self.prep_ul.get(lh, 0.0), want)
            self._order_prep(lh, "needed", before=f"{job}.mv_hotel_lh")
        if not ok_c:
            self._order_cells(lh, pm.cell_line, before=f"{job}.mv_hotel_lh")
        return ok_d and ok_c

    def _use_supplies(self, lh: str, job: str, now: int) -> None:
        S, pm = self.supplies, self.plates[job]
        S.consume(lh, pm)
        b = S.batch.get(lh)
        if b:
            self.batch_of[job] = b.made
        # low-water and age marks: make the next batch before a plate has to wait for it
        nxt = S.need(pm)
        if b and (any(b.left.get(k, 0) - 20 < 2 * v for k, v in nxt["drug"].items()) or
                  (S.plan.max_age_min is not None and now - b.made > S.plan.max_age_min - 180)):
            self._order_prep(lh, "low")

    def _new_supply_job(self, lh: str, kind: str, ops: list[Op], span: bool, before: str | None, why: str, **log) -> str:
        job = ops[0].job
        now = int(self.env.now)
        for op in ops:
            self.ops[op.id] = op
            self.run[op.id] = OpRun()
        for a, b in zip(ops, ops[1:]):
            self.lags.append(Lag(a.id, b.id, 0, self.durations.transfer_max_lag_min))
        if span:
            self._add_spans([Span(ops[0].id, ops[-1].id, ops[-1].id)])
        if before:
            self.lags.append(Lag(ops[-1].id, before, 0, None))
        self.release[job] = now
        self.supply_job[(lh, kind)] = job
        self._log(f"{kind}_order", lh=lh, job=job, reason=why, **log)
        self.request(f"{kind}_resupply")
        return job

    def _order_prep(self, lh: str, why: str, before: str | None = None) -> None:
        cur = self.supply_job.get((lh, "prep"))
        if cur:
            last = f"{cur}.prep"
            if before and not self.problem_has_lag(last, before):
                self.lags.append(Lag(last, before, 0, None))
                self.request("prep_wait")
            return
        self.n_supply_jobs += 1
        job = f"PREP{self.n_supply_jobs:03d}_{lh}"
        S = self.supplies
        dur = S.plan.prep_minutes(len(S.drugs), {d: self.prep_tops[d] for d in S.drugs}, max(len(l) for l in S.ladders))
        self._new_supply_job(lh, "prep", [Op(f"{job}.prep", job, "prep", (lh,), dur)], False, before, why)

    def _order_cells(self, lh: str, line: str, before: str | None = None) -> None:
        cur = self.supply_job.get((lh, "cells"))
        if cur:  # one swap at a time: the plate waits for it (its own line's swap is ordered once it is done)
            last = f"{cur}.cell_swap"
            if before and not self.problem_has_lag(last, before):
                self.lags.append(Lag(last, before, 0, None))
                self.request("cells_wait")
            return
        self.n_supply_jobs += 1
        job = f"CELLS{self.n_supply_jobs:03d}_{lh}"
        arm = tuple(r for r in self.resources if r.startswith("arm"))
        ops = []
        if self.supplies.cells.get(lh):  # the reservoir on the deck goes back to the supply hotel
            ops.append(Op(f"{job}.mv_lh_rackhotel", job, "transfer", arm, self.durations.arm("lh>rackhotel")))
        ops += [Op(f"{job}.mv_rackhotel_lh", job, "transfer", arm, self.durations.arm("rackhotel>lh")),
                Op(f"{job}.cell_swap", job, "prep", (lh,), 1)]
        self.supply_line[job] = line
        self._new_supply_job(lh, "cells", ops, True, before, "line" if self.supplies.cells.get(lh) else "first",
                             line=line, prev=(self.supplies.cells.get(lh) or {}).get("line"))

    def _use_prep_tips(self, lh: str, n: int) -> None:
        """A dilution series takes fresh tips of the drug tip type, from the same racks the plates use."""
        if not self.durations.deck or lh not in self.racks:
            return
        racks = self.racks[lh].setdefault(self.durations.deck.get("drug_tip", "300uL"), [96])
        while n > 0:
            j = next((j for j, c in enumerate(racks) if c > 0), None)
            if j is None:
                racks[:] = [96] * len(racks)
                continue
            take = min(n, racks[j])
            racks[j] -= take
            n -= take

    def _batch_made(self, oid: str) -> None:
        from labloop.execution.supplies import TIPS_PER_DRUG

        from labloop.execution.prep_protocol import tips as prep_tips

        job, lh = self.ops[oid].job, self.run[oid].resource
        S = self.supplies
        tr = S.plan.protocol(S.drugs, {d: self.prep_tops[d] for d in S.drugs}, max(len(l) for l in S.ladders))
        self._use_prep_tips(lh, sum(prep_tips(tr).values()) or TIPS_PER_DRUG * len(S.drugs))
        if self.plr:  # the batch is pipetted on the simulated STARlet, volumes tracked by PyLabRobot
            rep = self.plr.lh(lh).make_batch(tr, S.plan)
            if rep["errors"]:
                self._log("plr_error", lh=lh, job=job, errors=rep["errors"][:5])
        ul = self.prep_ul.pop(lh, None)
        if ul and self.plr:  # the bigger batch on the STARlet: dilution wells hold the scaled volume
            self.plr.lh(lh).scale_batch(ul)
        b = self.supplies.make_batch(lh, int(self.env.now), self.prep_tops, job, ul)
        if ul:
            self._log("prep_scaled", lh=lh, job=job, ul_per_well=round(ul, 1), plan_ul=S.plan.ul_per_well)
        self.supply_job[(lh, "prep")] = None
        self._log("prep", lh=lh, job=job, made=b.made, ul_per_well=self.supplies.plan.ul_per_well,
                  stock_ul={d: round(v, 1) for d, v in self.supplies.stock_ul.items()})

    def _plr_plate(self, lh: str, job: str) -> None:
        """Dispense the plate on the PyLabRobot STARlet from the sources as they stand."""
        from labloop.protocol.deck import DeckDesign
        from labloop.protocol.planner import plan_plate

        pm = self.plates[job]
        # one dilution-plate column per dose of the whole ladder, as the batches are made (not per plate)
        ladders = tuple((0.0, *l) for l in self.supplies.ladders) if self.supplies else None
        rep = self.plr.lh(lh).run_plate(pm, plan_plate(pm, DeckDesign(**self.durations.deck), ladders))
        if rep["errors"]:
            self._log("plr_error", lh=lh, plate=job, errors=rep["errors"][:5])

    def _cells_seated(self, oid: str) -> None:
        job, lh = self.ops[oid].job, self.run[oid].resource
        line = self.supply_line[job]
        self.supplies.fill_cells(lh, line, int(self.env.now), job)
        if self.plr:
            self.plr.lh(lh).seat_cells(self.supplies.cells[lh]["left_ml"])
        self.supply_job[(lh, "cells")] = None
        self._log("cell_swap", lh=lh, job=job, line=line, left_ml=round(self.supplies.cells[lh]["left_ml"], 2))

    def problem_has_lag(self, a: str, b: str) -> bool:
        return any(l.before == a and l.after == b for l in self.lags)

    def _refill_tips(self, job: str) -> None:
        lh = next(l for l, j in self.tip_order.items() if j == job)
        unused = 0
        for t, j in self.tip_racks.pop(job):
            unused += self.racks[lh][t][j]  # a partly used rack goes to the waste with its tips
            self.racks[lh][t][j] = 96
        self.tip_order[lh] = None
        self._log("tip_refill", lh=lh, job=job, tips_discarded_unused=unused)

    def _verify_protocol(self, job: str) -> None:
        """Run the plate's pipetting protocol on the PLR STARlet simulator; wells whose
        tracked volume is off become a liquid-transfer failure."""
        from labloop.execution.plr_bridge import PLRBridge

        pm = self.plates[job]
        n_drugs = len(pm.conditions()[0].doses) if pm.conditions() else 0
        rep = PLRBridge(n_drugs).run(pm)
        self._log("protocol", plate=job, passes=rep.passes, tips=rep.tips_used, rack_swaps=rep.rack_swaps,
                  refills=rep.refills, model_cycles=self.durations.cycles(pm), wells_bad=rep.wells_bad)
        if rep.wells_bad:
            self.corrupt.setdefault(job, set()).update(rep.wells_bad)
            self._failure(FailureEvent(F.LIQUID_TRANSFER, int(self.env.now), plate=job, wells=tuple(rep.wells_bad),
                                       detail={"source": "plr volume tracker"}))

    def _finish(self, oid: str, now: int) -> None:
        r = self.run[oid]
        r.status, r.end = "done", now
        self.occupancy[r.resource].discard(oid)
        if oid in self.family and self.ops[oid].kind in self.setup_min:
            self.res_last[r.resource] = (self.family[oid], now)
        sp = self.spans.get(oid)
        if sp and oid == sp.last:
            self.zone[self.zone_res[sp]] = None
        op = self.ops[oid]
        if not op.fixed and now - r.start > op.dur_hi:
            self._failure(FailureEvent(F.TIME_WINDOW, now, plate=op.job, op=oid, severity=Severity.HIGH,
                                       detail={"why": "incubation exceeded read window",
                                               "over_min": now - r.start - op.dur_hi}))

    def _lag_slip(self, lag: Lag, gap: int) -> None:
        self._log("lag_slip", op=lag.after, gap=gap, max_lag=lag.max_lag, constraint=lag.tag)

    def _actual(self, oid: str) -> int:
        nominal = self._nominal(oid, self.run[oid].resource)
        cv = self.failures.duration_cv
        if cv <= 0:
            return nominal
        return max(1, int(round(nominal * self.rng.uniform(1 - cv, 1 + cv))))

    def _run_op(self, oid: str):
        op, r = self.ops[oid], self.run[oid]
        if op.fixed:
            yield self.env.timeout(self._actual(oid))
            yield from self._inject(oid)
            while r.extra:
                extra, r.extra = r.extra, 0
                yield self.env.timeout(extra)
            if op.kind == "read":
                self._read(op.job)
            self._finish(oid, int(self.env.now))
            r.ready_at = r.end
            if op.id.endswith(".tip_swap"):
                self._refill_tips(op.job)
            if op.id.endswith(".prep"):
                self._batch_made(oid)
            if op.id.endswith(".cell_swap"):
                self._cells_seated(oid)
        else:
            yield self.env.timeout(op.dur_min)
            r.status, r.ready_at = "ready", int(self.env.now)
            if self._succ(oid) is None:
                self._finish(oid, int(self.env.now))
        if op.kind == "transfer" and op.id.endswith("mv_reader_hotel"):
            self.plate_status[op.job] = "done" if self.plate_status[op.job] != "invalid" else "invalid"
        self._poke()

    # ------------------------------------------------------------------ failures
    def _inject(self, oid: str):
        """Sample failures for a finished op; recovery actions may add time to it."""
        f, op, r = self.failures, self.ops[oid], self.run[oid]
        if op.job not in self.plates:  # tip-rack resupply jobs carry no plate
            return
        checks = []
        if op.kind == "dispense":
            checks = [(F.TIP_PICKUP, f.p_tip_pickup), (F.LIQUID_TRANSFER, f.p_liquid_transfer)]
        elif op.kind == "transfer":
            if f.p_drop > 0 and self.rng.random() < f.p_drop:
                self._drop(oid)
                if r.extra:
                    extra, r.extra = r.extra, 0
                    yield self.env.timeout(extra)
                return
            checks = [(F.PLATE_MISPLACED, f.p_misplace)]
        elif op.kind == "read":
            checks = [(F.MEASUREMENT, f.p_measurement)]
        for ftype, p in checks:
            attempt = 1
            while p > 0 and self.rng.random() < (p if attempt == 1 or ftype is not F.PLATE_MISPLACED
                                                 or f.p_replace_fail is None else f.p_replace_fail):
                wells = ()
                if ftype is F.PLATE_MISPLACED and f.perception_curve:
                    dx, dy = self.rng.normal(0, f.misplace_sigma_mm, 2)
                    offset = float(np.hypot(dx, dy))
                    xs, ys = zip(*f.perception_curve)
                    detected = self.rng.random() < float(np.interp(offset, xs, ys))
                    self._log("camera_check", plate=op.job, op=oid, dx_mm=round(float(dx), 2), dy_mm=round(float(dy), 2),
                              offset_mm=round(offset, 2), detected=bool(detected))
                    if not detected:
                        names = sorted(self.plates[op.job].spec.well_name(*w) for w, c in self.plates[op.job].wells.items()
                                       if not isinstance(c, str))
                        # Only a plate about to be pipetted or read is harmed by sitting off-target.
                        harmful = oid.endswith(("mv_hotel_lh", "mv_incubator_reader"))
                        k = min(2 * f.wells_per_failure, len(names)) if harmful else 0
                        lost = tuple(sorted(names[i] for i in self.rng.choice(len(names), k, replace=False))) if k else ()
                        self.corrupt.setdefault(op.job, set()).update(lost)
                        self._log("undetected_misplacement", plate=op.job, op=oid, offset_mm=round(offset, 2),
                                  wells=list(lost))
                        break
                if ftype in (F.LIQUID_TRANSFER, F.MEASUREMENT):
                    names = sorted(self.plates[op.job].spec.well_name(*w) for w, c in self.plates[op.job].wells.items()
                                   if not isinstance(c, str))
                    k = min(f.wells_per_failure, len(names))
                    wells = tuple(sorted(names[i] for i in self.rng.choice(len(names), k, replace=False))) if k else ()
                if wells:
                    self.corrupt.setdefault(op.job, set()).update(wells)
                ev = FailureEvent(ftype, int(self.env.now), plate=op.job, op=oid, resource=r.resource,
                                  wells=wells, attempt=attempt)
                decision = self._failure(ev)
                if ftype in (F.LIQUID_TRANSFER,):
                    break  # detected after the fact; no in-place retry to re-sample
                if not any(a.type in (A.RETRY, A.REMEASURE, A.HUMAN, A.SUBSTITUTE) for a in decision.actions):
                    break
                if ftype is F.MEASUREMENT and any(a.type is A.REMEASURE for a in decision.actions):
                    attempt += 1
                    continue
                if any(a.type is A.SUBSTITUTE for a in decision.actions):
                    break  # next tip: assume it succeeds
                attempt += 1
                if attempt > 4:
                    break
            if r.extra:
                extra, r.extra = r.extra, 0
                yield self.env.timeout(extra)

    def _drop(self, oid: str) -> None:
        """The arm lost the plate on this transfer: the plate's remaining steps cannot run. After
        the read the data is already taken; only the spill needs cleaning."""
        job, now = self.ops[oid].job, int(self.env.now)
        if self.run[f"{job}.read"].status == "done":
            self._log("plate_dropped", plate=job, op=oid, cancelled=0, after_read=True)
            self._failure(FailureEvent(F.PLATE_DROPPED, now, plate=job, op=oid, resource=self.run[oid].resource,
                                       severity=Severity.MEDIUM, detail={"after_read": True}))
            return
        rest = [o for o, op in self.ops.items() if op.job == job and self.run[o].status == "pending"]
        for o in rest:
            r = self.run[o]
            r.status, r.start, r.end, r.ready_at, r.cancelled = "done", now, now, now, True
        for sp, res in list(self.zone_res.items()):  # an LH zone this plate held is free again
            if self.ops[sp.last].job == job and self.zone.get(res) is sp:
                self.zone[res] = None
        self._log("plate_dropped", plate=job, op=oid, cancelled=len(rest), after_read=False)
        self.plate_status[job] = "invalid"
        self._failure(FailureEvent(F.PLATE_DROPPED, now, plate=job, op=oid, resource=self.run[oid].resource,
                                   severity=Severity.HIGH))
        self.on_read[job](self.plates[job], [])  # the campaign stops waiting for this plate

    def _state(self, ev: FailureEvent) -> LabState:
        now = int(self.env.now)
        res = {}
        for name, rr in self.resources.items():
            kind = name.rstrip("0123456789")
            res[name] = {"kind": kind, "capacity": rr.capacity,
                         "down_until": max((b for a, b in self.down[name] if a <= now < b), default=None),
                         "alternatives": [o for o in self.resources if o != name and o.rstrip("0123456789") == kind]}
        plate = None
        if ev.plate:
            pm = self.plates[ev.plate]
            plate = {"id": pm.plate_id, "campaign": pm.campaign, "cell_line": pm.cell_line,
                     "timepoint_h": pm.timepoint_h, "n_wells": len(pm.conditions()),
                     "status": self.plate_status[pm.plate_id]}
        recent = [e for e in self.events if e["kind"] == "failure"][-5:]
        budget = {"plates_submitted": len(self.plates),
                  "plates_invalid": sum(1 for s in self.plate_status.values() if s == "invalid")}
        return LabState(now, res, plate, budget, recent)

    def _failure(self, ev: FailureEvent) -> Decision:
        decision = self.policy.decide(ev, self._state(ev))
        self._log("failure", event=ev.to_dict(), decision=decision.to_dict())
        for act in decision.actions:
            self._apply(act, ev)
        if decision.reschedule:
            self.request(f"failure:{ev.type.value}")
        return decision

    def _apply(self, act: RecoveryAction, ev: FailureEvent) -> None:
        now = int(self.env.now)
        f = self.failures
        if act.type is A.RETRY and ev.op:
            self.run[ev.op].extra += f.retry_min.get(ev.type.value, 1)
        elif act.type is A.SUBSTITUTE and ev.op:
            self.run[ev.op].extra += f.substitute_min
        elif act.type is A.HUMAN and ev.op:
            self.run[ev.op].extra += f.human_min
        elif act.type is A.REMEASURE and ev.op:
            self.run[ev.op].extra += self.durations.read(self.plates[ev.plate])
            self.corrupt.setdefault(ev.plate, set()).difference_update(ev.wells)
        elif act.type in (A.WAIT_REPAIR, A.REASSIGN):
            pass  # the device is already down for its physical repair time; both lead to a replan
        elif act.type is A.INVALIDATE_WELLS:
            plate = act.plate or ev.plate
            self.bad_wells.setdefault(plate, set()).update(act.wells or ev.wells)
        elif act.type is A.INVALIDATE_PLATE:
            plate = act.plate or ev.plate
            self.plate_status[plate] = "invalid"
        elif act.type is A.RETEST:
            plate = act.plate or ev.plate
            wells = act.wells or ev.wells
            if plate in self.on_retest:
                self.on_retest[plate](self.plates[plate], tuple(wells))

    def _breakdowns(self, res: str):
        rate = self.failures.breakdown_per_day / (24 * 60)
        while True:
            yield self.env.timeout(max(1, int(self.rng.exponential(1 / rate))))
            if not self.busy():
                continue
            # The device finishes the op it is running, then is down for the repair time,
            # whatever the recovery policy decides about the work queued on it.
            now = int(self.env.now)
            busy_until = max([now] + [self.run[o].start + (self.run[o].est_dur or self._nominal(o, res))
                                      for o in self.occupancy[res] if self.ops[o].fixed])
            self.down[res].append((busy_until, busy_until + self.failures.repair_min))
            ev = FailureEvent(F.DEVICE_DOWN, now, resource=res, severity=Severity.HIGH,
                              detail={"down_from": busy_until, "repair_min": self.failures.repair_min})
            self._failure(ev)

    def quarantine(self, plates: list[str], reason: str) -> dict:
        """Data from these plates is no longer trusted: their campaigns drop it and re-run the conditions."""
        done = [p for p in plates if p in self.on_quarantine and p in self.plate_reads
                and self.plate_status.get(p) not in ("invalid", "quarantined")]
        reads = sum(self.on_quarantine[p](self.plates[p]) for p in done)
        for p in done:
            self.plate_status[p] = "quarantined"
        self._log("quarantine", plates=done, reads_dropped=reads, reason=reason)
        return {"plates": done, "reads_dropped": reads}

    # ------------------------------------------------------------------ physical limits
    def _changeover(self, lh: str, job: str) -> None:
        """Seeding a different cell line on a liquid handler that was not cleaned carries cells over."""
        line, prev = self.plates[job].cell_line, self.lh_line.get(lh)
        self.lh_line[lh] = line
        if prev in (None, line):
            return
        if "dispense" in self.setup_min:  # the lab knows to clean: it does, waiting if the plan left too little time
            last = self.res_last.get(lh)
            short = self.setup_min["dispense"] - (int(self.env.now) - last[1]) if last else 0
            if short > 0:
                self.run[f"{job}.dispense"].extra += short
            self._log("clean", lh=lh, plate=job, prev=prev, line=line, delay=max(0, short))
            return
        if not self.truth.changeover_wells:
            return
        pm = self.plates[job]
        names = sorted(pm.spec.well_name(*w) for w, c in pm.wells.items() if not isinstance(c, str))
        k = min(self.truth.changeover_wells, len(names))
        wells = sorted(names[i] for i in self.rng.choice(len(names), k, replace=False)) if k else []
        self.corrupt.setdefault(job, set()).update(wells)
        self._log("truth", plate=job, limit="changeover", lh=lh, prev=prev, wells=len(wells))

    def _carryover(self, job: str) -> None:
        """Contact dispensing with a reused tip into wells that already hold liquid mixes neighbours' contents."""
        what = self.truth.carryover(self.durations.deck)
        if not what:
            return
        pm = self.plates[job]
        names = sorted(pm.spec.well_name(*w) for w, c in pm.wells.items() if not isinstance(c, str))
        k = min(self.truth.carryover_wells, len(names))
        wells = sorted(names[i] for i in self.rng.choice(len(names), k, replace=False)) if k else []
        self.corrupt.setdefault(job, set()).update(wells)
        self._log("truth", plate=job, limit="carryover", reagent=what, wells=len(wells))

    def _effects(self, job: str) -> dict:
        if not self.truth.on:
            return {}
        inc = self.run[f"{job}.incubate"]
        made = self.batch_of.get(job, self.release[job])  # with supplies, drug dilutions age from their batch
        fx = self.truth.plate_effects(made, self.run[f"{job}.dispense"].start, inc.start,
                                      inc.end - inc.start, t_out=self.run[f"{job}.mv_incubator_reader"].start,
                                      t_read=self.run[f"{job}.read"].start)
        broken = {k: v for k, v in fx.items() if k.endswith(("_over_min", "_short_min"))}
        if broken:
            self._log("truth", plate=job, limit="+".join(k.removesuffix("_over_min").removesuffix("_short_min")
                                                          for k in broken), **broken)
        if self.supplies:
            pe = self.truth.prep_effects(self.supplies.plan, self.supplies.drugs, self.prep_tops)
            if pe:
                fx = {**fx, **pe}
                self._log("truth", plate=job, limit="prep", **{k: (v if k != "dmso" else round(sum(v["top_pct"]), 2))
                                                                for k, v in pe.items()})
        lw = self.truth.labware_effects(self.durations.deck, self.plates[job].spec.n_wells)
        if lw:
            noise = fx.get("extra_noise", 0.0) + lw.pop("extra_noise", 0.0)
            fx = {**fx, **lw, **({"extra_noise": noise} if noise else {})}
            self._log("truth", plate=job, limit="labware", assay_plate=(self.durations.deck or {}).get("assay_plate", "clear"),
                      **{k: v for k, v in lw.items() if k in ("crosstalk", "tip_under_range")})
        return fx

    def _qc_fail(self, job: str, reads) -> bool:
        if self.truth.qc_z_min is None:
            return False
        vals = lambda lab: [w.value for w in reads if w.content == lab]
        z = z_factor(vals(CONTROL_POS), vals(CONTROL_NEG))
        if z is None or z >= self.truth.qc_z_min:
            return False
        self._log("qc_fail", plate=job, z=round(z, 3), z_min=self.truth.qc_z_min)
        return True

    # ------------------------------------------------------------------ reads
    def _read(self, job: str) -> None:
        pm = self.plates[job]
        bad = self.bad_wells.get(job, set())
        reads = self.reader.read(pm, int(self.env.now), self.rng, bad_wells=bad,
                                 corrupt=self.corrupt.get(job, set()) - bad, effects=self._effects(job))
        if self.plate_status[job] != "invalid" and self._qc_fail(job, reads):
            self.plate_status[job] = "invalid"  # the plate is re-run: its conditions go back in the queue
            self.on_retest[job](pm, ())
        if self.plate_status[job] == "invalid":
            reads = [dataclasses.replace(w, valid=False) for w in reads]
        else:
            self.plate_status[job] = "read"
        self._log("read", plate=job, n_valid=sum(w.valid for w in reads), n=len(reads),
                  corrupt_accepted=len(self.corrupt.get(job, set()) - bad) if self.plate_status[job] != "invalid" else 0)
        self.plate_reads[job] = [{"well": w.well, "value": round(w.value, 4), "viability": round(w.viability, 4),
                                  "valid": w.valid} for w in reads]
        self.on_read[job](pm, reads)

    # ------------------------------------------------------------------ controller
    def _overruns(self, now: int) -> bool:
        """Re-estimate running fixed ops past their planned end; True if the plan breaks."""
        broken = False
        for oid, r in self.run.items():
            if r.status != "running" or not self.ops[oid].fixed or oid not in self.plan.end:
                continue
            expected_end = r.start + (r.est_dur or (self.plan.end[oid] - self.plan.start[oid]))
            if now >= expected_end:
                nominal = self._nominal(oid, r.resource)
                r.est_dur = now - r.start + max(1, math.ceil(OVERRUN_STEP * nominal))
                new_end = r.start + r.est_dur
                succ = self._succ(oid)
                if succ and self.plan.start.get(succ.after, 10**12) < new_end + succ.min_lag:
                    broken = True
                if self.resources[r.resource].capacity == 1 and any(
                        self.run[o].status == "pending" and self.plan.assigned.get(o) == r.resource
                        and self.plan.start.get(o, 10**12) < new_end for o in self.plan.start):
                    broken = True
        return broken

    def _next_wake(self, now: int) -> int | None:
        times = [self.plan.start[o] for o, r in self.run.items() if r.status == "pending" and o in self.plan.start
                 and self.plan.start[o] > now]
        for oid, r in self.run.items():
            if r.status == "running" and self.ops[oid].fixed and oid in self.plan.end:
                times.append(r.start + (r.est_dur or (self.plan.end[oid] - self.plan.start[oid])))
        for w in self.down.values():
            times += [b for a, b in w if b > now]
        for oid, r in self.run.items():
            if r.status == "pending":
                for lag in self._preds(oid):
                    t = self._anchor(lag)
                    if t is not None and t + lag.min_lag > now:
                        times.append(t + lag.min_lag)
        times = [t for t in times if t > now]
        return min(times) if times else None

    def _stalled(self, now: int) -> bool:
        """A pending op is due by plan and its predecessors are ready, but it cannot start."""
        for oid, r in self.run.items():
            if r.status != "pending" or oid not in self.plan.start or self.plan.start[oid] > now:
                continue
            if all(self.run[l.before].status in ("ready", "done") for l in self._preds(oid)) and \
                    not self._ready_to_start(oid, now):
                return True
        return False

    def _controller(self):
        stall_guard = -1
        while True:
            now = int(self.env.now)
            if self._overruns(now):
                self.request("overrun")
            if self.pending_reasons:
                self._reschedule()
            self._dispatch()
            if self._stalled(now) and stall_guard != now:
                stall_guard = now
                self.request("blocked")
                continue
            if not self.busy() and not self.idle.triggered:
                self.idle.succeed()
            if self.wake.triggered:
                self.wake = self.env.event()
            nxt = self._next_wake(now)
            waits = [self.wake]
            if nxt is not None:
                waits.append(self.env.timeout(nxt - now))
            yield simpy.AnyOf(self.env, waits)
            if self.idle.triggered and self.busy():
                self.idle = self.env.event()
