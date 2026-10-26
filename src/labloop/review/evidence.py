"""What a real lab could see while it runs, shaped like a run result so review/stats.py reads it.

A whitelist, not a copy: the simulator's hidden-limit bookkeeping (truth events, how many wells it
corrupted, misplacements it did not detect, the camera's true offsets) never reaches a review.
"""

from __future__ import annotations

import hashlib
import json

from labloop.types import Condition

HIDDEN_EVENTS = {"truth", "undetected_misplacement"}
HIDDEN_FIELDS = {"corrupt_accepted", "dx_mm", "dy_mm", "offset_mm"}
WALL = ("time_s", "ms")  # wall-clock fields: never part of anything hashed


def observable_view(lab) -> dict:
    plates = {}
    for pid, pm in lab.plates.items():
        reads = {w["well"]: w for w in getattr(lab, "plate_reads", {}).get(pid, [])}
        if not reads:
            continue
        plates[pid] = {"cell_line": pm.cell_line, "timepoint_h": pm.timepoint_h, "rows": pm.spec.rows, "cols": pm.spec.cols,
                       "status": lab.plate_status[pid], "reads": reads,
                       "wells": {pm.spec.well_name(r, c): (v.to_dict() if isinstance(v, Condition) else v)
                                 for (r, c), v in sorted(pm.wells.items())}}
    schedule = [{"op": oid, "job": lab.ops[oid].job, "kind": lab.ops[oid].kind, "resource": r.resource,
                 "start": r.start, "end": r.end}
                for oid, r in lab.run.items() if r.status == "done" and not r.cancelled]
    events = [{k: v for k, v in e.items() if k not in HIDDEN_FIELDS} for e in lab.events if e["kind"] not in HIDDEN_EVENTS]
    return {"plates": plates, "schedule": schedule, "events": events}


def strip_wall(x):
    if isinstance(x, dict):
        return {k: strip_wall(v) for k, v in x.items() if not k.endswith(WALL)}
    if isinstance(x, list):
        return [strip_wall(v) for v in x]
    return x


def rounded(x, nd: int = 6):
    """Floats rounded so a request hashes the same on every machine."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, nd) for v in x]
    return x


def prefix_hash(lab, view: dict) -> str:
    """Fingerprint of everything observable so far; a replayed run must reach each review with the same one."""
    body = {"events": view["events"], "reschedules": strip_wall(lab.reschedules)}
    return hashlib.sha256(json.dumps(rounded(body), sort_keys=True, default=str).encode()).hexdigest()[:16]
