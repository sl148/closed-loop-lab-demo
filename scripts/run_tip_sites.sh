#!/usr/bin/env bash
# Reruns after site-specific tip-box swaps (arm direct for front sites, LH CO-RE gripper + handoff for
# deep ones): MuJoCo calibration with the new routes, then M8 Level 2 (replay) and Level 4.
#   P=.venv/bin/python bash scripts/run_tip_sites.sh
set -u
export LABLOOP_TORCH_THREADS=2
P=${P:-"uv run python"}
L=results/logs
mkdir -p $L
$P scripts/sim3d_calibrate.py > $L/calibrate.txt 2>&1
G="$P scripts/run_level.py --workers 3"
$G --config configs/level2.yaml --seeds 10 --out results/m8 \
   --set landscape_eval=false --set mode=async --set max_in_flight=6 --set async_plates=1 --set scheduler=repair \
   --variant "baseline-deck:designer=bo-ts;deck=configs/deck_level2.json#baseline" \
   --variant "optimized-deck:designer=bo-ts;deck=configs/deck_level2.json" > $L/m8_l2.txt 2>&1
echo done > $L/TIPL2_DONE
$G --config configs/level4.yaml --seeds 5 --out results/m8 \
   --variant "baseline-deck:designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json#baseline" \
   --variant "optimized-deck:designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json" > $L/m8_l4.txt 2>&1
echo done > $L/TIPSITES_DONE
