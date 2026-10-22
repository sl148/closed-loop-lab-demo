"""LLM recovery policy (contribution C). Same interface as RulePolicy: FailureEvent +
LabState in, Decision out. The model only proposes; this module validates the proposal
(action names, minute bounds, wells within the event, devices that exist) and falls back to
the rule policy on any provider error, refusal, or invalid proposal. Every decision is
tagged with which path produced it."""

from __future__ import annotations

from labloop.agent.schemas import DecisionOut
from labloop.failures.events import (ActionType, Decision, FailureEvent, FailureType, LabState, RecoveryAction,
                                     Severity)
from labloop.failures.policies.rules import RulePolicy

MAX_MINUTES = 240


def to_payload(ev: FailureEvent, state: LabState) -> dict:
    ev_d = ev.to_dict()
    ev_d.pop("t", None)  # absolute time does not change the decision; keeps replay keys stable
    st = state.to_dict()
    st.pop("t", None)
    st["recent"] = [{"type": r["event"]["type"], "decision": [a["type"] for a in r["decision"]["actions"]]}
                    for r in st.get("recent", [])]
    return {"event": ev_d, "state": st}


def event_from_payload(p: dict) -> FailureEvent:
    e = p["event"]
    return FailureEvent(FailureType(e["type"]), 0, e.get("plate"), e.get("op"), e.get("resource"),
                        tuple(e.get("wells", ())), e.get("attempt", 1), Severity(e.get("severity", "medium")),
                        e.get("detail", {}))


def state_from_payload(p: dict) -> LabState:
    s = p["state"]
    return LabState(0, s["resources"], s["plate"], s["budget"], [])


class LLMPolicy:
    def __init__(self, provider, name: str = "llm"):
        self.provider, self.name = provider, name
        self.fallback = RulePolicy()
        self.stats = {"llm": 0, "invalid": 0, "error": 0}

    def _validate(self, out: DecisionOut, ev: FailureEvent, state: LabState) -> list[RecoveryAction]:
        acts = []
        for a in out.actions:
            if not 0 <= a.minutes <= MAX_MINUTES:
                raise ValueError(f"minutes {a.minutes} out of range")
            wells = tuple(a.wells)
            if not set(wells) <= set(ev.wells):
                raise ValueError("wells outside the event")
            if a.resource is not None and a.resource not in state.resources:
                raise ValueError(f"unknown resource {a.resource}")
            if a.type in ("wait_repair", "reassign") and (a.resource or ev.resource) is None:
                raise ValueError("downtime without a device")
            acts.append(RecoveryAction(ActionType(a.type), a.minutes, ev.plate, wells, a.resource))
        if not acts:
            raise ValueError("empty decision")
        return acts

    def decide(self, ev: FailureEvent, state: LabState) -> Decision:
        try:
            out = self.provider.decide(to_payload(ev, state))
        except Exception as exc:  # provider down, refusal, replay miss: deterministic fallback
            self.stats["error"] += 1
            d = self.fallback.decide(ev, state)
            d.policy, d.rationale = f"{self.name}->rules", f"provider error ({type(exc).__name__}); " + d.rationale
            return d
        try:
            acts = self._validate(out, ev, state)
        except ValueError as exc:
            self.stats["invalid"] += 1
            d = self.fallback.decide(ev, state)
            d.policy, d.rationale = f"{self.name}->rules", f"invalid proposal ({exc}); " + d.rationale
            return d
        self.stats["llm"] += 1
        return Decision(acts, out.reschedule, out.rationale, self.name)


def build_llm_policy(name: str, cfg: dict) -> LLMPolicy:
    from labloop.agent.claude_client import LiveProvider, ReplayProvider, ScriptedProvider

    provider = cfg.get("provider", "live" if name == "llm" else "replay")
    if provider == "live":
        p = LiveProvider(cfg.get("model", "claude-opus-5-5"), cfg.get("effort", "low"), cfg.get("log"))
    elif provider == "replay":
        p = ReplayProvider(cfg["log"])
    elif provider == "scripted":
        p = ScriptedProvider()
    else:
        raise KeyError(provider)
    return LLMPolicy(p, name)
