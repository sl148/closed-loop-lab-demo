#!/usr/bin/env bash
# Reruns every schedule-dependent experiment after adding LH deck zones (load -> dispense ->
# unload holds the LH, so the arm and the gantry never work in the zone at once), plus M8.
# Two chains on a shared server: GP sweeps with 3 workers, non-GP sweeps with 6.
#   P=.venv/bin/python bash scripts/run_zones.sh
set -u
export LABLOOP_TORCH_THREADS=2
P=${P:-"uv run python"}
L=results/logs
mkdir -p $L

gp_chain() {
  local G="$P scripts/run_level.py --workers 3"
  $G --config configs/level2.yaml --seeds 10 --out results/m8 \
     --set landscape_eval=false --set mode=async --set max_in_flight=6 --set async_plates=1 --set scheduler=repair \
     --variant "baseline-deck:designer=bo-ts;deck=configs/deck_level2.json#baseline" \
     --variant "optimized-deck:designer=bo-ts;deck=configs/deck_level2.json" > $L/m8_l2.txt 2>&1
  $G --config configs/level4.yaml --seeds 5 --out results/m8 \
     --variant "baseline-deck:designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json#baseline" \
     --variant "optimized-deck:designer=throughput-bo;scheduler=repair;deck=configs/deck_level4.json" > $L/m8_l4.txt 2>&1
  $G --config configs/level2.yaml --seeds 10 --out results/m4 --set landscape_eval=false \
     --variant "bo-sync-fifo:designer=bo-ts;mode=sync;scheduler=fifo" \
     --variant "bo-sync-cpsat:designer=bo-ts;mode=sync;scheduler=cpsat" \
     --variant "bo-async-repair:designer=bo-ts;mode=async;max_in_flight=3;async_plates=1;scheduler=repair" \
     --variant "tbo-async-repair:designer=throughput-bo;mode=async;max_in_flight=3;async_plates=1;scheduler=repair" \
     > $L/m4.txt 2>&1
  $G --config configs/level4.yaml --seeds 5 --out results/level4 \
     --variant "bo-fifo:designer=bo-ts;scheduler=fifo" \
     --variant "bo-repair:designer=bo-ts;scheduler=repair" \
     --variant "tbo-repair:designer=throughput-bo;scheduler=repair" > $L/level4.txt 2>&1
  $G --config configs/level4.yaml --seeds 5 --out results/level4_sim3d \
     --variant "camera:designer=throughput-bo;scheduler=repair;sim3d={misplace_sigma_mm: 1.5, perception: true}" \
     --variant "always-detected:designer=throughput-bo;scheduler=repair;sim3d={misplace_sigma_mm: 1.5, perception: false}" \
     > $L/level4_sim3d.txt 2>&1
  echo done > $L/GP_DONE
}

plain_chain() {
  local R="$P scripts/run_level.py --workers 6"
  $R --config configs/level3.yaml --seeds 10 --out results/m3 --set designer=random --set landscape_eval=false \
     --vary scheduler=fifo,priority-lwr,cpsat,repair > $L/m3_sync.txt 2>&1
  $R --config configs/level3.yaml --seeds 10 --out results/m3_async --set designer=random --set landscape_eval=false \
     --set mode=async --set max_in_flight=6 --set async_plates=1 --set rounds=8 \
     --vary scheduler=fifo,priority-lwr,cpsat,repair > $L/m3_async.txt 2>&1
  $P scripts/run_level.py --workers 3 --config configs/level3.yaml --seeds 3 --out results/m3_shadow \
     --set designer=random --set landscape_eval=false --set mode=async --set max_in_flight=6 --set async_plates=1 \
     --set rounds=8 --set shadow_time_s=10 --vary scheduler=repair > $L/m3_shadow.txt 2>&1
  $P scripts/repair_scaling.py --sizes 8 16 32 64 --seeds 3 > $L/scaling.txt 2>&1
  echo done > $L/PLAIN_DONE
}

gp_chain & plain_chain & wait
echo done > $L/ZONES_DONE
