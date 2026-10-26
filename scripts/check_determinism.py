"""Does a seeded campaign replay exactly? (M10 A1; run on the server)

    PYTHONPATH=src python scripts/check_determinism.py --config configs/level4.yaml --rounds 2 \
        --set scheduler=repair "--set=scheduler_cfg={deterministic: true}" ...

Runs the campaign twice in fresh processes with different PYTHONHASHSEED values and compares the
results with wall-clock fields removed. Prints the first difference, the makespan and the wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WALL = ("time_s", "shadow_time_s", "solve_time_s", "ms")


def strip(x):
    """Drop wall-clock fields, recursively."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in WALL and not k.endswith("_time_s")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def first_diff(a, b, path="") -> str | None:
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:80]} vs {str(b)[:80]}"


CHILD = """
import json, sys, yaml
from labloop.loop import run_campaign
cfg, seed, out = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
open(out, "w").write(json.dumps(run_campaign(cfg, seed)))
"""


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    for c in cfg.get("campaigns", []):
        c["rounds"] = min(args.rounds, c.get("rounds", args.rounds))
    runs = []
    with tempfile.TemporaryDirectory() as d:
        for hs in ("1", "2"):
            out = Path(d) / f"run{hs}.json"
            t = time.perf_counter()
            subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg), str(args.seed), str(out)], check=True,
                           env={**os.environ, "PYTHONHASHSEED": hs})
            runs.append((json.loads(out.read_text()), time.perf_counter() - t))
    (a, ta), (b, tb) = runs
    diff = first_diff(strip(a), strip(b))
    print(json.dumps({"identical": diff is None, "first_diff": diff, "makespan_h": [a["makespan"] / 60, b["makespan"] / 60],
                      "replans": len(a["reschedules"]), "wall_s": [round(ta), round(tb)],
                      "solvers": sorted({r["solver"] for r in a["reschedules"]})}))


if __name__ == "__main__":
    main()
