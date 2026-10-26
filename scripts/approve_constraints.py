"""Human gate between the constraint agent and the lab (M9).

    uv run python scripts/approve_constraints.py results/constraints/run            # review
    uv run python scripts/approve_constraints.py results/constraints/run --accept all --out configs/constraints_agent.yaml
    uv run python scripts/approve_constraints.py results/constraints/run --accept seeding_rest,changeover --out ...

Shows each proposal with its source quote; writes only the accepted ones, stamped with who and
when, to a constraint file that runs can use (`constraints: configs/constraints_agent.yaml`).
"""

from __future__ import annotations

import argparse
import getpass
import json
import time
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--accept", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    run = Path(args.run)
    cs = json.loads((run / "constraints.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    for c in cs["constraints"]:
        s = c.get("source") or {}
        print(f"[{c['id']}] {c['kind']} {c['minutes']} min  hard={c['hard']}\n"
              f"    {s.get('doc')} p.{s.get('page')}: \"{s.get('quote', '')}\"\n    {c.get('rationale', '')}")
    for slot, why in summary.get("abstained", {}).items():
        print(f"[{slot}] abstained: {why}")
    print(f"usage: {summary['usage']}")
    if not args.accept:
        return
    keep = {c["id"] for c in cs["constraints"]} if args.accept == "all" else set(args.accept.split(","))
    approved = [c for c in cs["constraints"] if c["id"] in keep]
    out = {"task": cs.get("task", ""), "approved_by": getpass.getuser(), "approved_at": time.strftime("%Y-%m-%d %H:%M"),
           "from_run": str(run), "constraints": approved}
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"wrote {len(approved)} constraints to {args.out}")


if __name__ == "__main__":
    main()
