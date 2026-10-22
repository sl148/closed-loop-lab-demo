#!/usr/bin/env bash
# Remaining experiments after the memory-limited first batch, one job at a time, 3 workers
# (each BO worker needs ~2.5-3.5 GB RAM).   bash scripts/run_rest.sh
set -u
export LABLOOP_TORCH_THREADS=2
L=results/logs
G="uv run python scripts/run_level.py --workers 3"
$G --config configs/level2.yaml --seeds 10 --out results/m2 --vary designer=al,bo-ts,bo-qucb,bo-qei > $L/m2_bo.txt 2>&1
uv run python scripts/repair_scaling.py --sizes 8 16 32 64 --seeds 3 > $L/scaling.txt 2>&1
uv run python scripts/eval_recovery.py --provider live --seeds 5 --only 4 > $L/m6_seed4.txt 2>&1
uv run python scripts/eval_recovery.py --provider replay --seeds 5 > $L/m6_replay.txt 2>&1
$G --config configs/level4.yaml --seeds 5 --out results/level4_sim3d \
   --variant "camera:designer=throughput-bo;scheduler=repair;sim3d={misplace_sigma_mm: 1.5, perception: true}" \
   --variant "always-detected:designer=throughput-bo;scheduler=repair;sim3d={misplace_sigma_mm: 1.5, perception: false}" \
   > $L/level4_sim3d.txt 2>&1
uv run python scripts/export_web.py > $L/export.txt 2>&1
uv run python scripts/make_results.py >> $L/export.txt 2>&1
echo done > $L/REST_DONE
