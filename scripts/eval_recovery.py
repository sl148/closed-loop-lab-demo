"""Rules vs LLM recovery policy on injected failures (M6).

    # offline plumbing check (no API calls)
    uv run python scripts/eval_recovery.py --provider scripted --seeds 3
    # live: calls Claude, records results/m6/decisions_seed<k>.jsonl  (costs API credits).
    # Credentials: ANTHROPIC_API_KEY in the environment or in ./.env (gitignored).
    uv run python scripts/eval_recovery.py --provider live --seeds 5
    # replay a recorded live run bit-identically with zero API calls
    uv run python scripts/eval_recovery.py --provider replay --seeds 5

Scenario: configs/level3.yaml with failure rates raised so each run sees a few dozen
failures, random designer (discovery fixed), repair scheduler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from labloop.loop import run_campaign
from labloop.metrics import run_metrics

OUT = Path("results/m6")
FAILURES = {"duration_cv": 0.2, "p_tip_pickup": 0.3, "p_liquid_transfer": 0.3, "p_misplace": 0.1,
            "p_measurement": 0.3, "breakdown_per_day": 1.0}
KEYS = ("n_failures", "makespan_h", "recovery_min", "corrupt_accepted", "wells_invalid", "invalid_plates",
        "regret", "llm_fallbacks")


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader (KEY=VALUE lines) so the live run can pick up ANTHROPIC_API_KEY
    without exporting it in the shell. .env is gitignored."""
    import os

    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["live", "replay", "scripted"], default="scripted")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", type=int, nargs="*", help="run only these seeds (e.g. finish an interrupted live run)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.provider == "live":
        # Fail fast: without credentials every decision would silently fall back to the rules.
        import anthropic

        try:
            anthropic.Anthropic().models.retrieve(args.model)
        except Exception as exc:
            raise SystemExit(f"Claude API unavailable ({type(exc).__name__}: {exc}). Set ANTHROPIC_API_KEY "
                             "in the environment or in ./.env, then rerun.")
    cfg = yaml.safe_load(Path("configs/level3.yaml").read_text())
    cfg.update(designer="random", scheduler="repair", rounds=args.rounds, failures=FAILURES)
    cfg["resources"] = {**cfg["resources"], "readers": 2}

    rows = {"rules": [], f"llm-{args.provider}": []}
    for seed in (args.only if args.only else range(args.seeds)):
        rows["rules"].append(run_metrics(run_campaign(dict(cfg, policy="rules"), seed)))
        log = OUT / f"decisions_seed{seed}.jsonl"
        if args.provider == "live" and log.exists():
            log.unlink()
        pcfg = {"provider": args.provider, "model": args.model, "effort": args.effort, "log": str(log)}
        run = run_campaign(dict(cfg, policy="llm" if args.provider == "live" else "llm-replay", policy_cfg=pcfg), seed)
        rows[f"llm-{args.provider}"].append(run_metrics(run))
        (OUT / f"run_{args.provider}_seed{seed}.json").write_text(json.dumps(
            [e for e in run["events"] if e["kind"] == "failure"], indent=1))

    summary = {}
    for name, ms in rows.items():
        summary[name] = {k: round(float(np.mean([m.get(k, 0) for m in ms])), 3) for k in KEYS}
        print(f"{name:14} " + "  ".join(f"{k}={v}" for k, v in summary[name].items()))
    (OUT / f"summary_{args.provider}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
