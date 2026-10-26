"""Run seeds of a campaign with in-loop reviews answered from a decision log (server side of M10 A6).

    PYTHONPATH=src python scripts/run_review.py --config configs/level4.yaml --set ... \
        --variant agent_review --seeds 3 --workers 3 --log results/m9/reviews/agent_review/decisions.jsonl

Seeds whose next review has no logged answer stop early and leave a request file in
results/m9/reviews/<variant>/requests/; scripts/review_driver.py answers them with the agent (locally,
where the API key is) and runs this again. When no seed is pending, the runs are written to
results/m9/level4/<variant>.json. status.json says which.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from labloop.loop import run_campaign


def _run(args):
    cfg, seed = args
    try:
        return run_campaign(cfg, seed)
    except Exception as e:  # e.g. NondeterminismError: report, do not hide
        return {"seed": seed, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="results/m9/level4")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["review"] = {**cfg.get("review", {}), "provider": "replay", "on_missing": "stop", "log": args.log}
    rdir = Path(args.log).parent
    req_dir = rdir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    for f in req_dir.glob("*.json"):
        f.unlink()
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(_run, [(cfg, s) for s in range(args.seeds)]))
    pending, errors = [], []
    for r in runs:
        if "error" in r:
            errors.append(r)
        elif "pending" in r:
            req = {**r["pending"], "seed": r["seed"]}
            (req_dir / f"seed{r['seed']}_{req['key']}.json").write_text(json.dumps(req, indent=1))
            pending.append({"seed": r["seed"], "index": req["index"], "kind": req["kind"], "key": req["key"]})
    status = {"variant": args.variant, "pending": pending, "errors": errors,
              "done": [r["seed"] for r in runs if "pending" not in r and "error" not in r]}
    if not pending and not errors:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"{args.variant}.json").write_text(json.dumps(sorted(runs, key=lambda r: r["seed"])))
        status["written"] = str(Path(args.out) / f"{args.variant}.json")
    (rdir / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status))


if __name__ == "__main__":
    main()
