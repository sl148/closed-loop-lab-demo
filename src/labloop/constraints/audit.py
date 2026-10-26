"""How well an executed run kept each constraint: per spec, plates (or switches) checked,
violations, and the worst overrun in minutes. Works on the JSON result of `run_campaign`, so it
scores any run against any constraint set (e.g. agent-found constraints against the gold set)."""

from __future__ import annotations

from labloop.constraints.spec import (STEP_OP, Changeover, DeadlineFromRelease, MaxBetween, MinBetween,
                                      ReadWindow)

TIGHT_MIN = 2  # a kept limit within this many minutes is binding the schedule


def _over(c, rows: dict, p: str, plate: dict, release: int) -> int | None:
    op = lambda step: rows.get(f"{p}.{STEP_OP[step]}")
    if isinstance(c, (MaxBetween, MinBetween)):
        a, b = op(c.a), op(c.b)
        if not a or not b:
            return None
        gap = b["start"] - a[c.a_anchor]
        return gap - c.minutes if isinstance(c, MaxBetween) else c.minutes - gap
    if isinstance(c, DeadlineFromRelease):
        s = op(c.step)
        return None if not s else s["start"] - release - c.minutes
    if isinstance(c, ReadWindow):
        inc = op("incubate")
        return None if not inc else inc["end"] - inc["start"] - plate["timepoint_h"] * 60 - c.minutes
    return None


def audit(specs: list, run: dict, plates: set[str] | None = None, since: int | None = None) -> dict[str, dict]:
    """`plates` limits the check to those plates (a changeover counts at the plate after the switch);
    `since` skips checks whose later op started before that minute (it ran under an earlier set)."""
    late = lambda r: since is None or (r is not None and r["start"] >= since)
    rows = {r["op"]: r for r in run["schedule"]}
    release = {p: e["t"] for e in run["events"] if e["kind"] == "submit" for p in e["plates"]}
    out = {}
    for c in specs:
        overs: list[int] = []
        if isinstance(c, Changeover):
            # A switch is kept only if the lab cleaned: idle time between plates is not a clean.
            cleaned = {e["plate"] for e in run["events"] if e["kind"] == "clean"}
            by_res: dict[str, list[tuple[dict, str]]] = {}
            for p, plate in run["plates"].items():
                r = rows.get(f"{p}.{STEP_OP[c.step]}")
                if r:
                    by_res.setdefault(r["resource"], []).append((r, plate[c.by]))
            for seq in by_res.values():
                seq.sort(key=lambda x: x[0]["start"])
                overs += [c.minutes - (b["start"] - a["end"]) if b["job"] in cleaned else max(1, c.minutes)
                          for (a, fa), (b, fb) in zip(seq, seq[1:])
                          if fa != fb and (plates is None or b["job"] in plates) and late(b)]
        else:
            for p, plate in run["plates"].items():
                step = c.b if isinstance(c, (MaxBetween, MinBetween)) else getattr(c, "step", "read")
                if (plates is None or p in plates) and c.scope.covers(plate["cell_line"], plate["timepoint_h"])                         and late(rows.get(f"{p}.{STEP_OP[step]}")):
                    o = _over(c, rows, p, plate, release.get(p, 0))
                    if o is not None:
                        overs.append(o)
        bad = [o for o in overs if o > 0]
        out[c.id] = {"n": len(overs), "violated": len(bad), "rate": round(len(bad) / len(overs), 4) if overs else 0.0,
                     "worst_over_min": max(bad, default=0),
                     "tight": sum(-TIGHT_MIN <= o <= 0 for o in overs)}  # kept, but right at the limit: it binds
    return out


def audit_versions(run: dict) -> dict[int, dict[str, dict]]:
    """Each plate checked against the constraint set it ran under (runs with mid-run updates)."""
    from labloop.constraints.compile import load_constraints

    out = {}
    for h in run.get("constraint_history", []):
        plates = {p for p, v in run.get("plate_version", {}).items() if v == h["version"]}
        if plates:
            out[h["version"]] = audit(load_constraints(h["specs"]).specs, run, plates, since=h["t"])
    return out
