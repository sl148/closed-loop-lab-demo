#!/usr/bin/env bash
# Reruns after the housing / waste-block / tip-box changes: decks re-optimized with carriers kept
# clear of the waste block, tip boxes disposed and replaced per rack, and MuJoCo arm times from
# collision-free fold-and-stage motions.   P=.venv/bin/python bash scripts/run_deck_v2.sh
set -u
export LABLOOP_TORCH_THREADS=2
P=${P:-"uv run python"}
L=results/logs
mkdir -p $L
G="$P scripts/run_level.py --workers 3"
$G --config configs/level2.yaml --seeds 10 --out results/m8 \
   --set landscape_eval=false --set mode=async --set max_in_flight=6 --set async_plates=1 --set scheduler=repair \
   --variant "baseline-deck:designer=bo-ts;deck=configs/deck_level2.json#baseline" \
   --variant "optimized-deck:designer=bo-ts;deck=configs/deck_level2.json" > $L/m8_l2.txt 2>&1
$G --config configs/level4.yaml --seeds 5 --out results/m8 \
   --variant "baseline-deck:designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json#baseline" \
   --variant "optimized-deck:designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json" > $L/m8_l4.txt 2>&1
$G --config configs/level4.yaml --seeds 5 --out results/level4_sim3d \
   --variant "camera:designer=throughput-bo;scheduler=repair;sim3d={misplace_sigma_mm: 1.5, perception: true}" \
   --variant "always-detected:designer=throughput-bo;scheduler=repair;sim3d={misplace_sigma_mm: 1.5, perception: false}" \
   > $L/level4_sim3d.txt 2>&1
echo done > $L/DECK2_DONE
