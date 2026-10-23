#!/usr/bin/env bash
# The web Console's run: Level 4 on the optimized deck with the MuJoCo calibration (camera
# checks), throughput-aware BO + repair, and the GP belief logged for the Discovery map. The
# viewer plays seed 0 (repair uses wall-clock limits, so reruns differ slightly).
#   P=.venv/bin/python bash scripts/run_console.sh
set -u
export LABLOOP_TORCH_THREADS=2
P=${P:-"uv run python"}
L=results/logs
mkdir -p $L
$P scripts/run_level.py --workers 2 --config configs/level4.yaml --seeds 3 --out results/console \
   --variant "console:designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json;sim3d={misplace_sigma_mm: 1.5, perception: true};belief_log=true" \
   > $L/console.txt 2>&1
echo done > $L/CONSOLE_DONE
