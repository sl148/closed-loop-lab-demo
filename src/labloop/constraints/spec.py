"""Constraint vocabulary (M9): what the constraint agent may emit, and nothing else.

Each spec is typed, scoped to the plate workflow's steps, and carries the archive passage it
came from. `compile.py` turns specs into scheduler primitives; the agent never writes solver code.

Steps of a plate (schedule/workflow.py): load (hotel -> LH), dispense, to_incubator, incubate,
to_reader, read, unload (reader -> hotel).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

Step = Literal["load", "dispense", "to_incubator", "incubate", "to_reader", "read", "unload"]
STEP_OP = {"load": "mv_hotel_lh", "dispense": "dispense", "to_incubator": "mv_lh_incubator", "incubate": "incubate",
           "to_reader": "mv_incubator_reader", "read": "read", "unload": "mv_reader_hotel"}


class Source(BaseModel):
    doc: str  # archive document id (data/kb/manifest.yaml)
    page: int | None = None
    quote: str = ""  # verbatim passage the value comes from
    verified: bool = False  # set by the quote check, never by the agent


class Scope(BaseModel):
    cell_lines: list[str] | None = None  # None = every plate
    timepoints_h: list[int] | None = None

    def covers(self, cell_line: str, timepoint_h: int) -> bool:
        return (self.cell_lines is None or cell_line in self.cell_lines) and \
            (self.timepoints_h is None or timepoint_h in self.timepoints_h)


class _Base(BaseModel):
    id: str
    hard: bool = True  # hard limits from documents need a person to relax them
    scope: Scope = Field(default_factory=Scope)
    source: Source | None = None
    rationale: str = ""
    evidence: Literal["doc", "data"] = "doc"  # "data": the lab's own reads showed the limit (review/stats.py)
    value_assumed: bool = False  # the source shows the limit exists but gives no number; soft until reviewed
    data: dict | None = None  # for evidence="data": the flag (hypothesis, factor, score, effect, p)

    @model_validator(mode="after")
    def _assumed_from_rationale(self):
        # older agent runs marked an assumed value only in the rationale text
        if "value_assumed" not in self.model_fields_set and "[value assumed" in self.rationale:
            self.value_assumed = True
        return self


class MaxBetween(_Base):
    """At most `minutes` from `a` (its start or end) to the start of `b`.
    E.g. seeded cells reach the incubator within 30 min of the dispense start."""
    kind: Literal["max_between"] = "max_between"
    a: Step
    a_anchor: Literal["start", "end"] = "end"
    b: Step
    minutes: int


class MinBetween(_Base):
    """At least `minutes` from `a` (start or end) to the start of `b` (e.g. equilibration)."""
    kind: Literal["min_between"] = "min_between"
    a: Step
    a_anchor: Literal["start", "end"] = "end"
    b: Step
    minutes: int


class DeadlineFromRelease(_Base):
    """`step` starts within `minutes` of the plate's release (reagents prepared then)."""
    kind: Literal["deadline_from_release"] = "deadline_from_release"
    step: Step
    minutes: int


class ReadWindow(_Base):
    """The read happens at most `minutes` after the nominal timepoint."""
    kind: Literal["read_window"] = "read_window"
    minutes: int


class Changeover(_Base):
    """Plates with a different `by` attribute on the same device need `minutes` of cleaning
    between their `step`s (e.g. cell-line switch on a liquid handler)."""
    kind: Literal["changeover"] = "changeover"
    step: Step = "dispense"
    by: Literal["cell_line"] = "cell_line"
    minutes: int


ConstraintSpec = Annotated[MaxBetween | MinBetween | DeadlineFromRelease | ReadWindow | Changeover,
                           Field(discriminator="kind")]


class ConstraintSet(BaseModel):
    task: str = ""
    constraints: list[ConstraintSpec] = Field(default_factory=list)
