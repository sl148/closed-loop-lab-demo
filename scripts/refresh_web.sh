#!/usr/bin/env bash
# Re-export every data file the web page plays from the current v3 runs (run on the server):
#   bash scripts/refresh_web.sh
# - Console replay: agent_review_v3 seed 0, compared with blind/agent_open/gold (export_loop --compare v3)
# - Live mode: the 12 combo replays (scripts/export_combos.sh)
# - Results tab baselines: the pre-registered runs (results/m9/confirm/level4)
set -eu
export PYTHONPATH=src MUJOCO_GL=egl
P=${P:-.venv/bin/python}
$P scripts/export_loop.py --variant agent_review_v3 --seed 0 --compare v3 --deck configs/deck_level4_agentlw2.json
bash scripts/export_combos.sh
$P scripts/export_baselines.py --res results/m9/confirm/level4
$P scripts/headline_table.py
