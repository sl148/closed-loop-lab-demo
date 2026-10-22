"""Run a level config over seeds for one or more variants and print a summary table.

    uv run python scripts/run_level.py --config configs/level1.yaml --seeds 10 \
        --vary scheduler=fifo,priority-lwr,cpsat,repair

Each variant writes results/<level>/<key>=<value>.json (all seeds). `--vary` may be given
several times (cartesian product). `--variant label:k=v;k=v` adds a named joint variant
instead (repeatable). `--set key=value` overrides a config key (YAML value).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from labloop.loop import run_campaign
from labloop.metrics import summarize


def _run(args):
    cfg, seed, overrides = args
    return run_campaign(cfg, seed, overrides)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--vary", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    axes = [(kv.split("=", 1)[0], kv.split("=", 1)[1].split(",")) for kv in args.vary] or [("variant", ["base"])]
    variants = [dict(zip([a for a, _ in axes], combo)) for combo in itertools.product(*[v for _, v in axes])]
    if args.variant:
        variants = []
        for spec in args.variant:
            label, kvs = spec.split(":", 1)
            v = {"variant": label}
            for kv in kvs.split(";"):
                k, val = kv.split("=", 1)
                v[k] = yaml.safe_load(val)
            variants.append(v)

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for var in variants:
        overrides = {k: v for k, v in var.items() if k != "variant"}
        jobs = [(cfg, s, overrides) for s in range(args.seeds)]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as ex:
                runs = list(ex.map(_run, jobs))
        else:
            runs = [_run(j) for j in jobs]
        tag = var["variant"] if "variant" in var and var["variant"] != "base" else ",".join(
            f"{k}={v}" for k, v in var.items())
        (out_dir / f"{tag}.json").write_text(json.dumps(runs))
        s = summarize(runs)
        rows.append((tag, s))
        print(f"{tag:42} " + "  ".join(f"{k}={v}" for k, v in s.items()), flush=True)
    (out_dir / "summary.json").write_text(json.dumps({t: s for t, s in rows}, indent=1))


if __name__ == "__main__":
    main()
