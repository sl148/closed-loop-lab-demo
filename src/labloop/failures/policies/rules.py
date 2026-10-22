"""Rule-based recovery policy: the default table from the design doc. Also the
deterministic fallback for the LLM policy."""

from __future__ import annotations

from labloop.failures.events import ActionType as A
from labloop.failures.events import Decision, FailureEvent, FailureType as F, LabState, RecoveryAction


class RulePolicy:
    name = "rules"

    def __init__(self, tip_retry_min: int = 1, misplace_fix_min: int = 2, human_min: int = 30,
                 repair_min: int = 90, max_attempts: int = 2):
        self.tip_retry_min, self.misplace_fix_min = tip_retry_min, misplace_fix_min
        self.human_min, self.repair_min, self.max_attempts = human_min, repair_min, max_attempts

    def decide(self, ev: FailureEvent, state: LabState) -> Decision:
        acts: list[RecoveryAction]
        if ev.type is F.TIP_PICKUP:
            # Retry at the same position, then fall through to the next tip.
            kind = A.RETRY if ev.attempt < self.max_attempts else A.SUBSTITUTE
            acts = [RecoveryAction(kind, self.tip_retry_min)]
            why = "retry same tip" if kind is A.RETRY else "retry failed; use next tip"
        elif ev.type is F.LIQUID_TRANSFER:
            acts = [RecoveryAction(A.INVALIDATE_WELLS, plate=ev.plate, wells=ev.wells),
                    RecoveryAction(A.RETEST, plate=ev.plate, wells=ev.wells)]
            why = "affected wells unusable; retest them in the next batch"
        elif ev.type is F.PLATE_MISPLACED:
            if ev.attempt < self.max_attempts:
                acts = [RecoveryAction(A.RETRY, self.misplace_fix_min)]
                why = "re-place plate with the arm"
            else:
                acts = [RecoveryAction(A.HUMAN, self.human_min)]
                why = "arm re-placement failed; operator fixes it"
        elif ev.type is F.DEVICE_DOWN:
            alts = state.resources.get(ev.resource, {}).get("alternatives", [])
            if alts:
                acts = [RecoveryAction(A.REASSIGN, self.repair_min, resource=ev.resource)]
                why = f"move work to {', '.join(alts)} while {ev.resource} is repaired"
            else:
                acts = [RecoveryAction(A.WAIT_REPAIR, self.repair_min, resource=ev.resource)]
                why = "no alternative device; wait for repair and reschedule"
        elif ev.type is F.MEASUREMENT:
            if ev.attempt < self.max_attempts:
                acts = [RecoveryAction(A.REMEASURE, plate=ev.plate)]
                why = "re-read the plate"
            else:
                acts = [RecoveryAction(A.INVALIDATE_WELLS, plate=ev.plate, wells=ev.wells),
                        RecoveryAction(A.RETEST, plate=ev.plate, wells=ev.wells)]
                why = "outliers persist; retest the wells"
        elif ev.type is F.TIME_WINDOW:
            acts = [RecoveryAction(A.INVALIDATE_PLATE, plate=ev.plate),
                    RecoveryAction(A.RETEST, plate=ev.plate)]
            why = "timing constraint violated; results void, rerun the plate's conditions"
        else:
            acts, why = [RecoveryAction(A.SKIP)], "unknown failure"
        return Decision(acts, reschedule=True, rationale=why, policy=self.name)
