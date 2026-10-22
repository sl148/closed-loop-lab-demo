"""Pydantic schemas for LLM recovery decisions. Coordination state never lives in prose:
the model returns this structure; free text is allowed only in `rationale`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ActionName = Literal["retry", "substitute", "remeasure", "wait_repair", "reassign", "human",
                     "invalidate_wells", "invalidate_plate", "retest", "skip"]


class ActionOut(BaseModel):
    type: ActionName
    minutes: int = Field(0, description="Time cost or downtime in minutes, 0-240")
    wells: list[str] = Field(default_factory=list, description="Well names, subset of the event's wells")
    resource: str | None = Field(None, description="Device name for wait_repair / reassign")


class DecisionOut(BaseModel):
    actions: list[ActionOut]
    reschedule: bool
    rationale: str = Field(description="One or two sentences")
