#!/usr/bin/env bash
# Every experiment behind the README figures and the web replay, in order.
# Takes a few hours on a 20-core / 32 GB machine.   bash scripts/run_all.sh
# GP-based sweeps run 6 workers (10 concurrent GP workers ran out of memory).
set -u
export LABLOOP_TORCH_THREADS=2
R="uv run python scripts/run_level.py --workers 10"
G="uv run python scripts/run_level.py --workers 6"
L=results/logs
mkdir -p $L
# M2: experiment selection (scheduler and layout fixed), and layouts (designer fixed)
$G --config configs/level2.yaml --seeds 10 --out results/m2 \
   --vary designer=${M2_DESIGNERS:-grid,random,lhs,al,bo-ts,bo-qucb,bo-qei} >> $L/m2_designers.txt 2>&1
$G --config configs/level2.yaml --seeds 10 --out results/m2_layout --set designer=bo-ts \
   --vary layout=sequential,random,optimized > $L/m2_layouts.txt 2>&1
# M3: schedulers under execution uncertainty (discovery fixed: random designer), sync and async
$R --config configs/level3.yaml --seeds 10 --out results/m3 --set designer=random --set landscape_eval=false \
   --vary scheduler=fifo,priority-lwr,cpsat,repair > $L/m3_sync.txt 2>&1
$R --config configs/level3.yaml --seeds 10 --out results/m3_async --set designer=random --set landscape_eval=false \
   --set mode=async --set max_in_flight=6 --set async_plates=1 --set rounds=8 \
   --vary scheduler=fifo,priority-lwr,cpsat,repair > $L/m3_async.txt 2>&1
# M3: repair quality gap vs a 10 s from-scratch CP-SAT shadow solve at every replan
$R --config configs/level3.yaml --seeds 3 --workers 3 --out results/m3_shadow --set designer=random \
   --set landscape_eval=false --set mode=async --set max_in_flight=6 --set async_plates=1 --set rounds=8 \
   --set shadow_time_s=10 --vary scheduler=repair > $L/m3_shadow.txt 2>&1
# M4: BO-then-schedule vs async vs throughput-aware (demo story step 3)
$G --config configs/level2.yaml --seeds 10 --out results/m4 --set landscape_eval=false \
   --variant "bo-sync-fifo:designer=bo-ts;mode=sync;scheduler=fifo" \
   --variant "bo-sync-cpsat:designer=bo-ts;mode=sync;scheduler=cpsat" \
   --variant "bo-async-repair:designer=bo-ts;mode=async;max_in_flight=3;async_plates=1;scheduler=repair" \
   --variant "tbo-async-repair:designer=throughput-bo;mode=async;max_in_flight=3;async_plates=1;scheduler=repair" \
   > $L/m4.txt 2>&1
# Level 4: everything at once
$G --config configs/level4.yaml --seeds 5 --workers 5 --out results/level4 \
   --variant "bo-fifo:designer=bo-ts;scheduler=fifo" \
   --variant "bo-repair:designer=bo-ts;scheduler=repair" \
   --variant "tbo-repair:designer=throughput-bo;scheduler=repair" > $L/level4.txt 2>&1
echo done > $L/ALL_DONE
