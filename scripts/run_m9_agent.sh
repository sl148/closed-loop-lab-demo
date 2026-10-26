#!/usr/bin/env bash
# M9 condition (b): Level 4 with the hidden limits, scheduled with the constraint agent's output.
# Same settings as scripts/run_m9_truth.sh, so results/m9/level4/agent.json compares with blind/gold.
#   P=.venv/bin/python RUN=results/constraints/pilot_haiku bash scripts/run_m9_agent.sh
set -u
export LABLOOP_TORCH_THREADS=2 PYTHONPATH=src
P=${P:-"uv run python"}
RUN=${RUN:-results/constraints/pilot_haiku}
L=results/logs
mkdir -p $L
BASE="designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json;sim3d={misplace_sigma_mm: 1.5, perception: true};truth=configs/truth.yaml"
$P scripts/run_level.py --workers 3 --config configs/level4.yaml --seeds 3 --out results/m9 \
   --variant "agent:$BASE;constraints=$RUN/constraints.json" \
   > $L/m9_agent.txt 2>&1
echo done > $L/M9_AGENT_DONE
