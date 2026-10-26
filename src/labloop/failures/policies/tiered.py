"""Tiered recovery (M9): rules first, Claude only when the rules run out.

Tier 1, the rule table, handles every known failure in microseconds and for free. Tier 2, the
LLM policy, is called only when the rules would hand the problem to a person, when no rule
covers the event, or when one plate keeps failing; it sees the rules' proposal and why it was
escalated, and its proposal is validated with the rules as fallback (failures/policies/llm.py).
Tier 3 is whatever person-in-the-loop action the chosen decision contains. A dropped plate goes
straight to the rules: the spill needs a person whatever is decided.
"""

from __future__ import annotations

import dataclasses

from labloop.failures.events import ActionType as A
from labloop.failures.events import Decision, FailureEvent, FailureType as F, LabState
from labloop.failures.policies.rules import RulePolicy

REPEAT = 2  # other steps of the same plate that already failed: the next failure is a compound case


def escalation(d: Decision, ev: FailureEvent, state: LabState) -> str | None:
    if ev.type is F.PLATE_DROPPED:
        return None
    if any(a.type is A.HUMAN for a in d.actions):
        return "rules exhausted: the next step is an operator"
    if any(a.type is A.SKIP for a in d.actions):
        return "no rule covers this event"
    ops = {r["event"].get("op") for r in state.recent if r.get("event", {}).get("plate") == ev.plate} - {ev.op}
    if ev.plate and len(ops) >= REPEAT:
        return "repeated failures on one plate"
    return None


class TieredPolicy:
    def __init__(self, llm, name: str = "tiered"):
        self.rules, self.llm, self.name = RulePolicy(), llm, name
        self.stats = {"rules": 0, "llm": 0, "llm->rules": 0}

    def decide(self, ev: FailureEvent, state: LabState) -> Decision:
        d = self.rules.decide(ev, state)
        why = escalation(d, ev, state)
        if why is None:
            self.stats["rules"] += 1
            d.policy = f"{self.name}:rules"
            return d
        ev = dataclasses.replace(ev, detail={**ev.detail, "escalated": why,
                                             "rule_proposal": [a.type.value for a in d.actions]})
        out = self.llm.decide(ev, state)
        tier = "llm" if out.policy == self.llm.name else "llm->rules"
        self.stats[tier] += 1
        out.policy = f"{self.name}:{tier}"
        return out
