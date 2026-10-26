"""The in-loop reviewer: every few plate reads, look at the lab's data and schedule, and when a check
fires, ask the agent (through a provider) and apply its decision under the lab's own rules.

Config (cfg["review"]):
    data: true            data-side review (review/stats.py: the agent's unproven hypotheses vs the reads)
    sched: true           schedule-side review (review/triggers.py)
    provider: replay | scripted | live
    log: path             decision log (replay/live)
    hypotheses: path      the open-mode run summary whose abstained hypotheses the data review tests
    every_reads: 2        check after this many reads (lab-wide)
    consecutive: 1        a hypothesis must pass this many checks in a row
    method: history, p0: 0.06   review/stats.py check_hypotheses
    alpha: 0.01, max_calls: 4, scope: queued, person: deny, kb: data/kb
    sched_min_n: 10, sched_frac: 0.5, sched_min_minutes: 30   (costly-constraint trigger)
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from labloop.agent.constraint_agent import SPEC, numbers_in, same_limit
from labloop.constraints.compile import ConstraintError, check, compile_constraints
from labloop.review.evidence import observable_view, prefix_hash, rounded
from labloop.review.providers import (LiveReviewProvider, ReplayReviewProvider, ScriptedReviewProvider,
                                      request_key)
from labloop.review.stats import check_hypotheses, plate_factors, plate_scores
from labloop.review.triggers import costly, infeasible

TIGHTER_IS_BIGGER = {"min_between", "changeover"}


def apply_answer(answer: dict, lab, archive, person: str = "deny") -> tuple[list | None, str]:
    """The lab's own check of a decision; returns (new spec list, note) or (None, why not)."""
    cur = {c.id: c.model_dump() for c in lab.cons.specs}
    act = answer.get("action")
    if act in (None, "none"):
        return None, "no change"
    if act == "escalate":
        return None, f"escalated to a person ({person})"
    try:
        c = SPEC.validate_python(answer["spec"])
        check(c)
    except (KeyError, ValidationError, ConstraintError) as e:
        return None, f"rejected: invalid spec ({str(e)[:200]})"
    d = c.model_dump()
    if act == "add":
        if c.hard or c.evidence != "data":
            return None, "rejected: a limit added from data must be soft and say so"
        if c.id in cur or any(same_limit(o, d) for o in cur.values()):
            return None, "rejected: already covered"
        return [*cur.values(), d], f"added {c.id} ({c.minutes} min, soft)"
    old = cur.get(answer.get("constraint_id"))
    if not old or old["kind"] != c.kind:
        return None, "rejected: no such constraint"
    if old["hard"] and not old.get("value_assumed") and person != "approve":
        return None, "rejected: hard limit, a person said no"
    if act == "replace":
        s = c.source
        if not (s and archive and archive.verify_quote(s.doc, s.page, s.quote)):
            return None, "rejected: quote not verified"
        if not {float(c.minutes), c.minutes / 60, c.minutes / 1440} & numbers_in(s.quote):
            return None, "rejected: value not in the quote"
    elif act == "loosen":
        looser = c.minutes < old["minutes"] if c.kind in TIGHTER_IS_BIGGER else c.minutes > old["minutes"]
        if not looser or old["hard"]:
            return None, "rejected: not a loosening of a soft limit"
    else:
        return None, f"rejected: unknown action {act}"
    return [d if k == old["id"] else v for k, v in cur.items()], f"{act} {old['id']}: {old['minutes']} -> {c.minutes} min"


def build_provider(rc: dict):
    kind = rc.get("provider", "replay")
    if kind == "replay":
        return ReplayReviewProvider(rc["log"], rc.get("on_missing", "stop"))
    if kind == "scripted":
        return ScriptedReviewProvider(rc.get("add_minutes", 15), rc.get("replace"))
    if kind == "live":
        return LiveReviewProvider(rc.get("kb", "data/kb"), rc.get("model", "claude-haiku-5-5"), rc.get("log"))
    raise KeyError(kind)


def unproven_hypotheses(path: str | None) -> list[dict]:
    """The agent's own hypotheses it abstained on (no number in the archive), from an open-mode summary."""
    if not path or not Path(path).exists():
        return []
    s = json.loads(Path(path).read_text())
    return [h for h in s.get("hypotheses", []) if h["id"] in s.get("abstained", {})]


class Reviewer:
    def __init__(self, rc: dict, cfg: dict, provider=None, archive=None, hypotheses: list[dict] | None = None):
        self.rc, self.provider = rc, provider or build_provider(rc)
        self.archive = archive
        if self.archive is None and Path(rc.get("kb", "data/kb"), "chunks.jsonl").exists():
            from labloop.kb.archive import Archive

            self.archive = Archive(Path(rc.get("kb", "data/kb")))
        self.hyps = hypotheses if hypotheses is not None else unproven_hypotheses(rc.get("hypotheses"))
        self.task = {"campaigns": [{"space": {"timepoints_h": c.get("space", cfg.get("space", {})).get("timepoints_h", [])}}
                                   for c in cfg.get("campaigns", [{}])]}
        self.every, self.consecutive = rc.get("every_reads", 2), rc.get("consecutive", 1)
        self.alpha, self.max_calls = rc.get("alpha", 0.01), rc.get("max_calls", 4)
        self.method, self.p0 = rc.get("method", "history"), rc.get("p0", 0.06)
        self.scope, self.person = rc.get("scope", "queued"), rc.get("person", "deny")
        self.reads = self.calls = 0
        self.streak: dict[str, int] = {}
        self.done_hyp: set[str] = set()
        self.done_cons: set[tuple] = set()
        self.last_infeasible_t = 0
        self.records: list[dict] = []

    def after_read(self, lab) -> None:
        self.reads += 1
        if self.reads % self.every or self.calls >= self.max_calls:
            return
        view = observable_view(lab)
        if self.rc.get("data", True):
            self._data(lab, view)
        if self.rc.get("sched", True) and self.calls < self.max_calls:
            self._sched(lab, view)

    # ------------------------------------------------------------------ checks
    def _data(self, lab, view: dict) -> None:
        hyps = [h for h in self.hyps if h["id"] not in self.done_hyp]
        if not hyps:
            return
        scores, factors = plate_scores(view), plate_factors(view)
        res = check_hypotheses(scores, factors, hyps, alpha=self.alpha, method=self.method, p0=self.p0)
        for r in res:
            self.streak[r["hypothesis"]] = self.streak.get(r["hypothesis"], 0) + 1 if r["flag"] else 0
        hit = next((r for r in res if self.streak.get(r["hypothesis"], 0) >= self.consecutive), None)
        if not hit:
            return
        h = next(x for x in hyps if x["id"] == hit["hypothesis"])
        exposed = [p for p in scores if p in factors and factors[p][hit["factor"]] > 0]
        flag = {**{k: hit.get(k) for k in ("hypothesis", "factor", "score", "effect", "p", "p_holm", "n", "arms", "method",
                                            "exposed_anomalous", "unexposed_anomalous", "p0")},
                "hypothesis_kind": h["kind"], "hypothesis_concern": h["concern"],
                "summary": (f"{hit.get('exposed_anomalous')} of {len(exposed)} plates after an uncleaned cell-line switch "
                            f"have drug wells reading above the untreated controls (historical rate {self.p0:.0%}, "
                            f"p={hit['p']:.4f})" if hit.get("method") == "history" else
                            f"{hit['score']} is higher on plates with {hit['factor']} ({len(exposed)} of {hit['n']} plates, "
                            f"one-sided p={hit['p']:.4f})")}
        plates = [{"plate": p, "cell_line": factors[p]["cell_line"], "lh": factors[p]["lh"], "previous_line": factors[p]["prev_line"],
                   hit["factor"]: factors[p][hit["factor"]], hit["score"]: scores[p][hit["score"]],
                   "max_viability": scores[p]["max_viability"]}
                  for p in sorted(scores, key=lambda q: factors.get(q, {}).get("read_t", 0)) if p in factors]
        self.done_hyp.add(h["id"])
        self._ask(lab, view, "data", {"flag": flag, "plates": plates, "plates_read": len(scores)})

    def _sched(self, lab, view: dict) -> None:
        bad = None if self.last_infeasible_t else infeasible(lab, 0)  # once per run
        if bad:
            self.last_infeasible_t = int(lab.env.now)
            self._ask(lab, view, "sched", bad)
            return
        hits = costly(lab, view, self.done_cons, self.rc.get("sched_min_n", 10), self.rc.get("sched_frac", 0.5),
                      self.rc.get("sched_min_minutes", 30))
        if hits:
            ev = hits[0]
            self.done_cons.add((ev["constraint_id"], ev["minutes"]))
            self._ask(lab, view, "sched", ev)

    # ------------------------------------------------------------------ ask, check, apply
    def _ask(self, lab, view: dict, kind: str, evidence: dict) -> None:
        self.calls += 1
        req = {"kind": kind, "index": self.calls, "t": int(lab.env.now), "evidence": rounded(evidence),
               "constraints": rounded([c.model_dump() for c in lab.cons.specs]), "cfg": self.task}
        req["key"], req["prefix"] = request_key(req), prefix_hash(lab, view)
        answer = self.provider.answer(req)  # may stop the run (ReviewPending) for an answer from elsewhere
        specs, note = apply_answer(answer, lab, self.archive, self.person)
        info = lab.update_constraints(compile_constraints([SPEC.validate_python(s) for s in specs]), self.scope,
                                      reason=f"review {self.calls} ({kind}): {note}", review_key=req["key"]) if specs else None
        quar = None
        if specs and kind == "data" and answer.get("quarantine"):  # drop data the flag says was hurt
            fac = evidence["flag"]["factor"]
            quar = lab.quarantine([p["plate"] for p in evidence["plates"] if (p.get(fac) or 0) > 0],
                                  reason=f"review {self.calls}: {evidence['flag']['hypothesis']}")
        rec = {"index": self.calls, "t": req["t"], "kind": kind, "key": req["key"], "evidence": req["evidence"],
               "answer": answer, "applied": bool(specs), "note": note, "update": info, "quarantine": quar}
        self.records.append(rec)
        lab._log("review", index=self.calls, review=kind, key=req["key"], action=answer.get("action"),
                 applied=bool(specs), note=note)
