from pathlib import Path

import yaml

from labloop.agent.schemas import DecisionOut
from labloop.failures.events import FailureEvent, FailureType, LabState
from labloop.failures.policies.llm import LLMPolicy, to_payload
from labloop.failures.policies.rules import RulePolicy
from labloop.loop import run_campaign
from labloop.metrics import run_metrics

STATE = LabState(0, {"lh1": {"kind": "lh", "capacity": 1, "down_until": None, "alternatives": []}},
                 {"id": "P1"}, {}, [])
EV = FailureEvent(FailureType.MEASUREMENT, 5, plate="P1", op="P1.read", resource="reader1", wells=("A1", "B2"))


class Fixed:
    def __init__(self, out):
        self.out = out

    def decide(self, payload):
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def test_valid_llm_decision_is_used():
    pol = LLMPolicy(Fixed(DecisionOut(actions=[{"type": "invalidate_wells", "wells": ["A1"]}], reschedule=False,
                                      rationale="x")))
    d = pol.decide(EV, STATE)
    assert d.policy == "llm" and d.actions[0].wells == ("A1",)


def test_invalid_or_failing_provider_falls_back_to_rules():
    bad_wells = DecisionOut(actions=[{"type": "invalidate_wells", "wells": ["H12"]}], reschedule=False, rationale="x")
    for provider in (Fixed(bad_wells), Fixed(RuntimeError("down")),
                     Fixed(DecisionOut(actions=[], reschedule=True, rationale="x"))):
        d = LLMPolicy(provider).decide(EV, STATE)
        assert d.policy == "llm->rules"
        assert [a.type for a in d.actions] == [a.type for a in RulePolicy().decide(EV, STATE).actions]


def test_payload_is_time_invariant():
    ev2 = FailureEvent(FailureType.MEASUREMENT, 999, plate="P1", op="P1.read", resource="reader1", wells=("A1", "B2"))
    assert to_payload(EV, STATE) == to_payload(ev2, STATE)


def test_scripted_llm_matches_rules_end_to_end_and_skipping_corrupts_data():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "level1.yaml").read_text())
    cfg = dict(cfg, rounds=2, failures={"p_measurement": 0.5, "p_liquid_transfer": 0.5, "p_tip_pickup": 0.3})
    rules = run_metrics(run_campaign(cfg, 1))
    scripted = run_metrics(run_campaign(dict(cfg, policy="llm", policy_cfg={"provider": "scripted"}), 1))
    assert rules["n_failures"] > 0
    assert scripted["corrupt_accepted"] == rules["corrupt_accepted"] == 0
    assert scripted["llm_fallbacks"] == 0
    # A policy that ignores failures lets wrong values into the data.
    import labloop.failures.policies.rules as R
    from labloop.failures.events import ActionType, Decision, RecoveryAction

    class Skip:
        name = "skip"

        def decide(self, ev, st):
            return Decision([RecoveryAction(ActionType.SKIP)], False, "", "skip")

    import labloop.loop as L
    orig = L.build_policy
    L.build_policy = lambda name, cfg=None: Skip()
    try:
        skipped = run_metrics(run_campaign(cfg, 1))
    finally:
        L.build_policy = orig
    assert skipped["corrupt_accepted"] > 0
