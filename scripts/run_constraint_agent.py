"""Run the constraint agent (M9) on the document archive.

    uv run python scripts/run_constraint_agent.py --config configs/level4.yaml \\
        --set deck=configs/deck_level4.json --slots read_equilibration,reagent_stability,changeover \\
        --out results/constraints/pilot [--sim-ssh user@host:~/labloop]

--mode open lets the agent decide what to look for (plan, ground, sweep) instead of --slots.
--labware also runs the labware phase (agent/labware.py) into the same --out; --labware-only skips
the timing limits.

Credentials: ANTHROPIC_API_KEY in the environment or in ./.env (gitignored; read here, never
printed or copied). With --sim-ssh the agent's `simulate` tool runs on that host over ssh, so
the key stays on this machine while the simulations use the server.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from labloop.agent.constraint_agent import SLOTS, run_agent


def load_env(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--slots", default=",".join(SLOTS))
    ap.add_argument("--mode", choices=["slots", "open"], default="slots")
    ap.add_argument("--plans", type=int, default=1, help="open mode: independent brainstorms whose hypotheses are merged")
    ap.add_argument("--out", default="results/constraints/run")
    ap.add_argument("--kb", default="data/kb")
    ap.add_argument("--model", default="claude-opus-5-5", help="claude-haiku-5-5 for cheap test runs")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--sim-ssh", help="user@host:dir of a labloop copy to run simulations in")
    ap.add_argument("--labware", action="store_true", help="also choose the assay plate and tips")
    ap.add_argument("--labware-only", action="store_true")
    ap.add_argument("--prep", action="store_true", help="also plan the liquids (agent/prep.py)")
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    sim_cmd = None
    if args.sim_ssh:
        host, _, d = args.sim_ssh.partition(":")
        sim_cmd = ["ssh", host, f"cd {d or '~/labloop'} && PYTHONPATH=src .venv/bin/python -m labloop.agent.constraint_agent"]
    if not (args.labware_only or args.prep_only):
        summary = run_agent(cfg, slots=args.slots.split(","), out=args.out, kb=args.kb, effort=args.effort,
                            max_turns=args.max_turns, sim_cmd=sim_cmd, model=args.model, mode=args.mode, plans=args.plans)
        print(json.dumps({k: summary[k] for k in ("proposed", "abstained", "usage")}, indent=1))
    if args.prep or args.prep_only:
        from labloop.agent.prep import run_prep

        pr = run_prep(cfg, args.out, kb=args.kb, model=args.model, effort=args.effort)
        print(json.dumps({k: pr[k] for k in ("plan", "usage")}, indent=1))
    if args.labware or args.labware_only:
        from labloop.agent.labware import run_labware

        lw = run_labware(cfg, args.out, kb=args.kb, model=args.model, effort=args.effort)
        print(json.dumps({k: lw[k] for k in ("deck_rules", "abstained", "usage")}, indent=1))


if __name__ == "__main__":
    main()
