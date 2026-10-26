"""Standard failure events and recovery actions shared by every recovery policy (rules,
LLM). Policies only see and emit these types; the lab applies the actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class FailureType(str, Enum):
    TIP_PICKUP = "tip_pickup"  # physics contact check / tip tracker mismatch
    LIQUID_TRANSFER = "liquid_transfer"  # volume tracker mismatch, short aspiration
    PLATE_MISPLACED = "plate_misplaced"  # camera perception vs planned layout
    PLATE_DROPPED = "plate_dropped"  # gripper lost the plate in transit: spilled, contaminated
    DEVICE_DOWN = "device_down"  # device status / timeout
    MEASUREMENT = "measurement"  # reader outliers
    TIME_WINDOW = "time_window"  # incubation window or reagent expiry violated


class Severity(str, Enum):
    LOW = "low"  # recoverable in place
    MEDIUM = "medium"  # costs time or wells
    HIGH = "high"  # invalidates a plate or needs a human


class ActionType(str, Enum):
    RETRY = "retry"  # repeat the step in place: adds `minutes` to the op
    SUBSTITUTE = "substitute"  # use the next tip / another device: adds `minutes`
    REMEASURE = "remeasure"  # read the plate again: adds a read duration
    WAIT_REPAIR = "wait_repair"  # device unavailable for `minutes`
    REASSIGN = "reassign"  # device out of service for `minutes`; reschedule onto alternatives
    HUMAN = "human"  # operator intervention: `minutes` of delay on the op
    INVALIDATE_WELLS = "invalidate_wells"  # mark `wells` of `plate` invalid
    INVALIDATE_PLATE = "invalidate_plate"
    RETEST = "retest"  # re-queue the invalidated conditions in the campaign's next batch
    SKIP = "skip"  # accept the loss, no further action


@dataclass(frozen=True)
class FailureEvent:
    type: FailureType
    t: int
    plate: str | None = None
    op: str | None = None
    resource: str | None = None
    wells: tuple[str, ...] = ()
    attempt: int = 1  # how many times this failure has hit the same target
    severity: Severity = Severity.MEDIUM
    detail: dict = field(default_factory=dict, hash=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"], d["severity"] = self.type.value, self.severity.value
        return d


@dataclass(frozen=True)
class RecoveryAction:
    type: ActionType
    minutes: int = 0
    plate: str | None = None
    wells: tuple[str, ...] = ()
    resource: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class Decision:
    actions: list[RecoveryAction]
    reschedule: bool
    rationale: str = ""
    policy: str = ""

    def to_dict(self) -> dict:
        return {"actions": [a.to_dict() for a in self.actions], "reschedule": self.reschedule,
                "rationale": self.rationale, "policy": self.policy}


@dataclass
class LabState:
    """What a policy may look at when deciding."""

    t: int
    resources: dict[str, dict]  # name -> {kind, capacity, down_until, alternatives}
    plate: dict | None  # plate id, campaign, cell line, timepoint, n wells, status
    budget: dict  # plates used / allowed, retest wells queued
    recent: list[dict]  # recent failure events (dicts), newest last

    def to_dict(self) -> dict:
        return asdict(self)
