#!/usr/bin/env bash
# M9 step 1-2 check: Level 4 (console settings) with the lab's hidden physical limits
# (configs/truth.yaml), scheduled blind vs with the gold constraints (configs/constraints_gold.yaml).
#   P=.venv/bin/python bash scripts/run_m9_truth.sh
set -u
export LABLOOP_TORCH_THREADS=2 PYTHONPATH=src
P=${P:-"uv run python"}
L=results/logs
mkdir -p $L
BASE="designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json;sim3d={misplace_sigma_mm: 1.5, perception: true};truth=configs/truth.yaml"
$P scripts/run_level.py --workers 3 --config configs/level4.yaml --seeds 3 --out results/m9 \
   --variant "blind:$BASE" \
   --variant "gold:$BASE;constraints=configs/constraints_gold.yaml" \
   > $L/m9_truth.txt 2>&1
echo done > $L/M9_TRUTH_DONE
