"""Seeded campaigns replay exactly with the deterministic scheduler (needed to replay agent reviews)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_repair_runs_replay_bit_identically_across_processes():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_determinism.py"), "--config", str(ROOT / "configs" / "level2.yaml"),
         "--rounds", "2", "--set", "scheduler=repair", "--set=scheduler_cfg={deterministic: true}"],
        capture_output=True, text=True, check=True, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["identical"], res["first_diff"]
    assert res["replans"] > 0
