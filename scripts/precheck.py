"""Pre-launch check of a campaign config with its constraints (labloop.precheck).

    uv run python scripts/precheck.py --config configs/level4.yaml \
        --set constraints=configs/constraints_gold.yaml --set deck=configs/deck_level4.json --n 10 --workers 3

Prints the report and writes results/precheck/<name>.json. Exit code 1 when the gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from labloop.precheck import precheck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-violation-rate", type=float, default=0.05)
    ap.add_argument("--out", default="results/precheck")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    rep = precheck(cfg, n=args.n, workers=args.workers, max_violation_rate=args.max_violation_rate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['name']}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
