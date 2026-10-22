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

from labloop.execution.durations import DurationModel
from labloop.execution.reader import PlateReader
from labloop.failures.events import ActionType as A
from labloop.failures.events import (Decision, FailureEvent, FailureType as F, LabState, RecoveryAction,
                                     Severity)
from labloop.schedule.model import Lag, Op, Problem, Resource, Schedule, Span
from labloop.schedule.serial import FIFOScheduler
from labloop.schedule.validator import violations
from labloop.schedule.workflow import plate_job
from labloop.types import PlateMap, WellRead

MID_SWAP_MIN = 2  # arm swaps all racks during a protocol: two moves plus re-teach check
OVERRUN_STEP = 0.25  # re-estimate an overrunning op as elapsed + this fraction of nominal


@dataclass
class FailureModel:
    duration_cv: float = 0.0  # fixed ops: actual = nominal * U(1 - cv, 1 + cv)
    p_tip_pickup: float = 0.0  # per dispense
    p_liquid_transfer: float = 0.0  # per dispense
    p_misplace: float = 0.0  # per arm transfer
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
    est_dur: int | None = None  # scheduler-facing estimate while overrunning


class Lab:
    def __init__(self, env: simpy.Environment, resources: dict[str, Resource], durations: DurationModel,
                 reader: PlateReader, scheduler, policy, rng: np.random.Generator,
                 failures: FailureModel | None = None, plr_verify: bool = False, shadow=None):
        self.env, self.resources, self.durations = env, resources, durations
        self.reader, self.scheduler, self.policy, self.rng = reader, scheduler, policy, rng
        self.failures = failures or FailureModel()
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
        self.tips: dict[str, dict[str, int]] = {}
        self.tip_order: dict[str, str | None] = {}
        self.tip_racks: dict[str, dict[str, int]] = {}  # TIPS job -> racks it replaces, per tip type
        self.n_tip_jobs = 0
        self.on_read: dict[str, Callable[[PlateMap, list[WellRead]], None]] = {}
        self.on_retest: dict[str, Callable[[PlateMap, tuple[str, ...]], None]] = {}
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
    def submit(self, plates: list[PlateMap], campaign: str, on_read, on_retest) -> None:
        prob = Problem(resources=self.resources)
        for pm in plates:
            ops = plate_job(prob, pm, self.durations, release=int(self.env.now))
            for op in ops:
                self.run[op.id] = OpRun()
            self.plates[pm.plate_id] = pm
            self.plate_status[pm.plate_id] = "queued"
            self.on_read[pm.plate_id], self.on_retest[pm.plate_id] = on_read, on_retest
        self.ops.update(prob.ops)
        self.lags += prob.lags
        self._add_spans(prob.spans)
        self.release.update(prob.release)
        self.start_ub.update(prob.start_ub)
        self._log("submit", campaign=campaign, plates=[p.plate_id for p in plates])
        self.request("new_batch")

    def _add_spans(self, spans: list[Span]) -> None:
        for sp in spans:
            for oid in (sp.first, sp.link, sp.last):
                self.spans[oid] = sp

    def request(self, reason: str) -> None:
        self.pending_reasons.append(reason)
        self._poke()

    def busy(self) -> bool:
        return any(r.status != "done" for r in self.run.values())

    def executed(self) -> Schedule:
        s = Schedule(solver="executed")
        for oid, r in self.run.items():
            if r.status == "done":
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
        return next((l for l in self.lags if l.before == oid), None)

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
                lo = before.end + lag.min_lag
                prob.start_lb[lag.after] = max(prob.start_lb[lag.after], lo)
                if lag.max_lag is not None and before.end + lag.max_lag >= prob.start_lb[lag.after]:
                    prob.start_ub[lag.after] = min(prob.start_ub.get(lag.after, 10**9), before.end + lag.max_lag)
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
            if pr.status not in ("ready", "done") or now < pr.ready_at + lag.min_lag:
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
        if sp and oid == sp.link:  # runs inside its own held zone
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
        self.occupancy[r.resource].add(oid)
        sp = self.spans.get(oid)
        if sp and oid == sp.first:
            lh = self.plan.assigned[sp.link]
            self.zone[lh], self.zone_res[sp] = sp, lh
        for lag in self._preds(oid):
            pr = self.run[lag.before]
            if pr.status == "ready":  # flexible predecessor hands the plate over now
                self._finish(lag.before, now)
            if lag.max_lag is not None and now - pr.end > lag.max_lag:
                self._lag_slip(lag, now - pr.end)
        op = self.ops[oid]
        if op.kind == "dispense" and self.durations.deck:
            self._use_tips(r.resource, op.job)
        if op.kind == "tip_swap":
            self._log("tip_swap", lh=r.resource, job=op.job)
        if op.kind == "dispense":
            self.plate_status[op.job] = "dispensing"
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
            self.tip_cap = {t: 96 * n for t, n in rack_split(need, racks).items()}
        if lh not in self.tips:
            self.tips[lh] = dict(self.tip_cap)
            self.tip_order[lh] = None

    def _tips_ok(self, lh: str, job: str) -> bool:
        need = self.durations.tips(self.plates[job])
        self._init_tips(lh, need)
        # A plate needing more tips than the deck holds starts on full racks and swaps mid-run.
        if all(self.tips[lh].get(t, 0) >= min(n, self.tip_cap.get(t, n)) for t, n in need.items()):
            return True
        self._order_tips(lh, reason="short", before=f"{job}.mv_hotel_lh")
        return False

    def _use_tips(self, lh: str, job: str) -> None:
        need = self.durations.tips(self.plates[job])
        self._init_tips(lh, need)
        mid = 0
        for t, n in need.items():
            have, cap = self.tips[lh].get(t, 0), max(1, self.tip_cap.get(t, n))
            short = max(0, n - have)
            k = -(-short // cap)  # racks-full swaps the arm makes while the head waits
            mid += k
            self.tips[lh][t] = have + k * cap - n
        if mid:
            self.run[f"{job}.dispense"].extra += mid * MID_SWAP_MIN
            self._log("tip_mid_swap", lh=lh, job=job, swaps=mid)
        # Low-water mark: not enough left for another plate like this one.
        if any(self.tips[lh].get(t, 0) < n for t, n in need.items()):
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
        # Racks to replace: every rack that is empty or partly used.
        have = self.tips.get(lh, {})
        racks = {t: -(-(cap - have.get(t, 0)) // 96) for t, cap in self.tip_cap.items()}
        racks = {t: k for t, k in racks.items() if k > 0} or {next(iter(self.tip_cap)): 1}
        self.tip_racks[job] = racks
        # Per rack: the arm takes the used box off the deck into the rack waste bin, then fetches
        # a full box from the rack hotel and seats it; the LH then registers the new racks.
        ops = []
        for i in range(sum(racks.values())):
            ops += [Op(f"{job}.mv_lh_rackwaste_{i}", job, "transfer", arm, self.durations.arm("lh>rackwaste")),
                    Op(f"{job}.mv_rackhotel_lh_{i}", job, "transfer", arm, self.durations.arm("rackhotel>lh"))]
        ops.append(Op(f"{job}.tip_swap", job, "tip_swap", (lh,), 1))
        now = int(self.env.now)
        for op in ops:
            self.ops[op.id] = op
            self.run[op.id] = OpRun()
        for a, b in zip(ops, ops[1:]):
            self.lags.append(Lag(a.id, b.id, 0, self.durations.transfer_max_lag_min))
        self._add_spans([Span(ops[0].id, ops[-1].id, ops[-1].id)])
        if before:  # the plate that ran short waits for the fresh racks
            self.lags.append(Lag(f"{job}.tip_swap", before, 0, None))
        self.release[job] = now
        self.tip_order[lh] = job
        self._log("tip_order", lh=lh, job=job, reason=reason, left=dict(have), racks=racks)
        self.request("tip_resupply")

    def problem_has_lag(self, a: str, b: str) -> bool:
        return any(l.before == a and l.after == b for l in self.lags)

    def _refill_tips(self, job: str) -> None:
        lh = next(l for l, j in self.tip_order.items() if j == job)
        unused = 0
        for t, k in self.tip_racks.pop(job).items():
            cap, have = self.tip_cap[t], self.tips[lh].get(t, 0)
            if k > (cap - have) // 96:  # a partly used rack goes to the waste with its tips
                unused += have % 96
                have -= have % 96
            self.tips[lh][t] = min(cap, have + 96 * k)
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
        sp = self.spans.get(oid)
        if sp and oid == sp.last:
            self.zone[self.zone_res[sp]] = None
        op = self.ops[oid]
        if not op.fixed and now - r.start > op.dur_hi:
            self._failure(FailureEvent(F.TIME_WINDOW, now, plate=op.job, op=oid, severity=Severity.HIGH,
                                       detail={"why": "incubation exceeded read window",
                                               "over_min": now - r.start - op.dur_hi}))

    def _lag_slip(self, lag: Lag, gap: int) -> None:
        self._log("lag_slip", op=lag.after, gap=gap, max_lag=lag.max_lag)

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
            if op.kind == "tip_swap":
                self._refill_tips(op.job)
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
            checks = [(F.PLATE_MISPLACED, f.p_misplace)]
        elif op.kind == "read":
            checks = [(F.MEASUREMENT, f.p_measurement)]
        for ftype, p in checks:
            attempt = 1
            while p > 0 and self.rng.random() < p:
                wells = ()
                if ftype is F.PLATE_MISPLACED and f.perception_curve:
                    offset = float(np.hypot(*self.rng.normal(0, f.misplace_sigma_mm, 2)))
                    xs, ys = zip(*f.perception_curve)
                    if self.rng.random() >= float(np.interp(offset, xs, ys)):
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

    # ------------------------------------------------------------------ reads
    def _read(self, job: str) -> None:
        pm = self.plates[job]
        bad = self.bad_wells.get(job, set())
        reads = self.reader.read(pm, int(self.env.now), self.rng, bad_wells=bad,
                                 corrupt=self.corrupt.get(job, set()) - bad)
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
                    pr = self.run[lag.before]
                    if pr.ready_at is not None and pr.ready_at + lag.min_lag > now:
                        times.append(pr.ready_at + lag.min_lag)
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
