"""Decision providers for the LLM recovery policy.

- LiveProvider: Claude via the Anthropic SDK (credentials resolved by the SDK), structured
  output parsed into DecisionOut; every call is appended to a JSONL decision log.
- ReplayProvider: answers from a decision log, keyed by the hash of the prompt payload, so a
  recorded run replays bit-identically with zero API calls.
- ScriptedProvider: offline stand-in (NOT an LLM) that returns the rule policy's decision
  through the same schema; used by tests to exercise the plumbing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labloop.agent.schemas import DecisionOut

SYSTEM = """You are the failure-recovery planner of an automated drug-combination screening lab.
Devices: liquid handlers (lh*), plate readers (reader*), a plate-moving arm (arm*), incubators
(incubator*, many plate slots). Each assay plate is dispensed, incubated 24/48/72 h, then read.

You receive one failure event and the lab state as JSON. Choose recovery actions. Their time
costs are physical and fixed; the `minutes` field is ignored:
- retry: repeat the step in place (tip pickup 1 min, plate re-placement 2 min). It can fail again.
- substitute: next tip or spare part (1 min); assume it works.
- human: an operator fixes it (30 min); always works.
- remeasure: read the plate again (one read time); clears the outlier wells if it succeeds.
- wait_repair / reassign: a broken device is already down for its 90 min repair whatever you
  choose; reassign says queued work should move to an alternative device if one exists.
- invalidate_wells / invalidate_plate: exclude data from the analysis.
- retest: queue the affected conditions again in the campaign's next batch.
- skip: accept the situation with no action.
A dropped plate is spilled and contaminated: it cannot be picked up and used again.
When event.detail.escalated is set, a fixed rule table already handled the routine options and
passed the case to you (rule_proposal is what it would do next); pick what is best now.

Wells hit by a liquid-transfer or measurement failure carry WRONG values unless remeasured or
invalidated; wrong data silently misleads the optimizer, which is worse than lost wells.
Time is also a cost: every minute added delays all downstream plates. Plates whose timing
window was violated produce invalid biology. Prefer the cheapest action that keeps the data
trustworthy. Set reschedule=true whenever timing or device availability changes."""


def payload_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class LiveProvider:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", log_path: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.log_path = Path(log_path) if log_path else None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def decide(self, payload: dict) -> DecisionOut:
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": json.dumps(payload, sort_keys=True)}],
            output_format=DecisionOut,
        )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RuntimeError(f"no decision (stop_reason={resp.stop_reason})")
        out = resp.parsed_output
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"key": payload_key(payload), "model": self.model,
                                    "decision": out.model_dump()}) + "\n")
        return out


class ReplayProvider:
    def __init__(self, log_path: str):
        self.table = {}
        for line in Path(log_path).read_text().splitlines():
            rec = json.loads(line)
            self.table[rec["key"]] = DecisionOut.model_validate(rec["decision"])

    def decide(self, payload: dict) -> DecisionOut:
        key = payload_key(payload)
        if key not in self.table:
            raise KeyError(f"decision {key} not in replay log")
        return self.table[key]


class ScriptedProvider:
    """Offline stand-in: the rule policy's answer, round-tripped through the schema."""

    def __init__(self):
        from labloop.failures.policies.rules import RulePolicy

        self.rules = RulePolicy()

    def decide(self, payload: dict) -> DecisionOut:
        from labloop.failures.policies.llm import event_from_payload, state_from_payload

        d = self.rules.decide(event_from_payload(payload), state_from_payload(payload))
        return DecisionOut(actions=[{"type": a.type.value, "minutes": a.minutes, "wells": list(a.wells),
                                     "resource": a.resource} for a in d.actions],
                           reschedule=d.reschedule, rationale="scripted: " + d.rationale)
