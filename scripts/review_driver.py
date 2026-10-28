"""Pause-and-resume driver for in-loop reviews (M10 A6). Runs on the local PC, where the API key is.

    PYTHONPATH=src python scripts/review_driver.py --host scottlee@130.126.138.27 --repo '~/labloop2' \
        --variant agent_review --sets-file scripts/m9_review_sets.txt --seeds 3

Each iteration: push the decision log to the server, run scripts/run_review.py there (all seeds; a seed
stops at its first review without a logged answer), pull the requests, answer each with the review
agent here, append the answers to the log, and go again. The simulations always run on the server and
replay earlier answers bit-identically; the key never leaves this machine. Stops when no seed is
pending, on an error, or after --max-iters.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from labloop.review.providers import log_entry, read_log


def load_env(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def sh(host: str, cmd: str, timeout: int = 600) -> str:
    return subprocess.run(["ssh", host, cmd], capture_output=True, text=True, check=True, timeout=timeout).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--repo", default="~/labloop2")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default="configs/level4.yaml")
    ap.add_argument("--sets-file", required=True, help="one --set value per line (same as run_level variants)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--kb", default="data/kb")
    ap.add_argument("--max-iters", type=int, default=8)
    args = ap.parse_args()
    load_env()
    rdir = Path("results/m9/reviews") / args.variant
    rdir.mkdir(parents=True, exist_ok=True)
    log = rdir / "decisions.jsonl"
    log.touch()
    remote = f"results/m9/reviews/{args.variant}"
    sets = " ".join(shlex.quote(f"--set={s.strip()}") for s in Path(args.sets_file).read_text().splitlines() if s.strip())
    # MUJOCO_GL=egl: per-event MuJoCo re-grasps render the deck camera on the server
    run = (f"cd {args.repo} && mkdir -p {remote} && export PYTHONPATH=src PYTHONHASHSEED=0 MUJOCO_GL=egl LABLOOP_TORCH_THREADS=2 "
           f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 && .venv/bin/python scripts/run_review.py --config {args.config} {sets} "
           f"--variant {args.variant} --seeds {args.seeds} --workers {args.workers} --log {remote}/decisions.jsonl "
           f"> {remote}/run.log 2>&1")
    # each round runs detached on the server, so a dropped ssh session (or this process being stopped) does not
    # kill it; a restarted driver attaches to a round that is still running instead of starting a second one
    busy = f"pgrep -f '[r]un_review.py .*--variant {args.variant} ' > /dev/null && echo busy || echo idle"
    sh(args.host, f"mkdir -p {args.repo}/{remote}")
    for it in range(1, args.max_iters + 1):
        if sh(args.host, busy).strip() == "busy":
            print(f"[{it}] attaching to the round already running on the server ...", flush=True)
        else:
            subprocess.run(["scp", "-q", str(log), f"{args.host}:{args.repo}/{remote}/decisions.jsonl"], check=True)
            print(f"[{it}] running {args.seeds} seeds on the server ...", flush=True)
            sh(args.host, f"nohup setsid bash -c {shlex.quote(run)} > /dev/null 2>&1 < /dev/null &")
            time.sleep(20)
        while sh(args.host, busy).strip() == "busy":
            time.sleep(60)
        status = json.loads(sh(args.host, f"tail -1 {args.repo}/{remote}/run.log").strip().splitlines()[-1])
        if status["errors"]:
            raise SystemExit(f"server errors: {status['errors']}")
        if not status["pending"]:
            print(f"[{it}] done: {status.get('written')}", flush=True)
            return
        known = read_log(log)
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["scp", "-q", "-r", f"{args.host}:{args.repo}/{remote}/requests", d], check=True)
            reqs = [json.loads(p.read_text()) for p in sorted(Path(d, "requests").glob("*.json"))]
        from labloop.agent.review_agent import review_agent

        answered = set()
        for req in reqs:
            if req["key"] in known:
                raise SystemExit(f"request {req['key']} is already answered: the run is not replaying")
            if req["key"] in answered:  # two seeds asked the same question
                continue
            out = review_agent(req, kb=args.kb, model=args.model)
            with open(log, "a") as f:
                f.write(json.dumps(log_entry(req, out)) + "\n")
            answered.add(req["key"])
            print(f"[{it}] seed {req['seed']} review {req['index']} ({req['kind']}): {out['answer'].get('action')} "
                  f"${out['usage']['usd']:.4f}", flush=True)
    raise SystemExit("max iterations reached")


if __name__ == "__main__":
    main()
