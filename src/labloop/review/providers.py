"""Who answers a review request (the same pattern as agent/claude_client.py for recovery decisions).

- ReplayReviewProvider: answers from a decision log (JSONL, keyed by request). With on_missing="stop" an
  unanswered request stops the run (ReviewPending) so the agent can answer it elsewhere; the rerun then
  replays every earlier answer bit-identically with no API calls. A request that reaches its logged
  answer with a different history means the run diverged (NondeterminismError).
- ScriptedReviewProvider: fixed rules, for tests and server-only ablations.
- LiveReviewProvider: runs the review agent in this process (needs the API key; the local PC only).
"""

from __future__ import annotations

import json
from pathlib import Path

from labloop.agent.claude_client import payload_key


class ReviewPending(Exception):
    def __init__(self, request: dict):
        super().__init__(request)  # SimPy re-raises an exception as type(e)(*e.args): keep the request as the arg
        self.request = request

    def __str__(self) -> str:
        r = self.request
        return f"review {r.get('index')} ({r.get('kind')}) needs an answer: {r.get('key')}"


class NondeterminismError(Exception):
    pass


def request_key(request: dict) -> str:
    return payload_key({k: request[k] for k in ("kind", "index", "evidence", "constraints")})


def read_log(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return {e["key"]: e for e in (json.loads(line) for line in p.read_text().splitlines() if line.strip())}


class ReplayReviewProvider:
    def __init__(self, log_path: str, on_missing: str = "stop"):
        self.log, self.on_missing = read_log(log_path), on_missing

    def answer(self, request: dict) -> dict:
        e = self.log.get(request["key"])
        if e is None:
            if self.on_missing == "stop":
                raise ReviewPending(request)
            raise KeyError(f"no logged answer for review {request['key']}")
        if e.get("prefix") and e["prefix"] != request["prefix"]:
            raise NondeterminismError(f"review {request['index']}: history differs from the logged run")
        return e["answer"]


class ScriptedReviewProvider:
    """Data review: add the flagged limit at a fixed assumed value. Schedule review: replace a costly
    assumed value with `replace` = {"kind": ..., "minutes": ..., "source": {...}} if given, else no change."""

    def __init__(self, add_minutes: int = 15, replace: dict | None = None):
        self.add_minutes, self.replace = add_minutes, replace or {}

    def answer(self, request: dict) -> dict:
        ev = request["evidence"]
        if request["kind"] == "data":
            f = ev["flag"]
            spec = {"id": f["hypothesis"], "kind": f["hypothesis_kind"], "minutes": self.add_minutes, "hard": False,
                    "value_assumed": True, "evidence": "data", "rationale": "scripted review",
                    "source": {"doc": "lab_data", "quote": f.get("summary", ""), "verified": False},
                    "data": {k: f.get(k) for k in ("hypothesis", "factor", "score", "effect", "p")}}
            return {"action": "add", "spec": spec, "quarantine": True}
        c = {x["id"]: x for x in request["constraints"]}.get(ev.get("constraint_id"))
        if c and self.replace and c["kind"] == self.replace.get("kind"):
            return {"action": "replace", "constraint_id": c["id"],
                    "spec": {**c, "minutes": self.replace["minutes"], "value_assumed": False, "source": self.replace["source"]}}
        return {"action": "none", "reason": "scripted: nothing to replace"}


class LiveReviewProvider:
    def __init__(self, kb: str = "data/kb", model: str = "claude-haiku-5-5", log_path: str | None = None):
        self.kb, self.model, self.log_path = kb, model, log_path

    def answer(self, request: dict) -> dict:
        from labloop.agent.review_agent import review_agent

        out = review_agent(request, kb=self.kb, model=self.model)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(log_entry(request, out)) + "\n")
        return out["answer"]


def log_entry(request: dict, out: dict) -> dict:
    return {"key": request["key"], "kind": request["kind"], "index": request["index"], "t": request["t"],
            "prefix": request["prefix"], "answer": out["answer"], "model": out.get("model"), "usage": out.get("usage"),
            "trace": out.get("trace", []), "turns": out.get("turns", [])}
