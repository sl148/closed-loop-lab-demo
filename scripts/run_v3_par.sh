#!/usr/bin/env bash
# The v3 headline variants (same as `run_l4_det.sh v3`), each in its own process so they run at once.
set -u
export PYTHONHASHSEED=0 PYTHONPATH=src LABLOOP_TORCH_THREADS=2 MUJOCO_GL=egl
P=${P:-.venv/bin/python}; L=results/logs; mkdir -p $L
V3="designer=throughput-bo;scheduler=repair;scheduler_cfg={deterministic: true, stability: 0, objective: {makespan: 1, flow: 10}};sim3d={misplace_sigma_mm: 1.5, perception: true, replace: true};truth=configs/truth_prep.yaml;plr=true"
run() { $P scripts/run_level.py --workers 3 --config configs/level4.yaml --seeds 3 --out results/m9 --variant "$1:$2" > $L/v3_$1.txt 2>&1; }
run blind_v3 "$V3;deck=configs/deck_level4.json;prep=default" &
run agent_open_v3 "$V3;deck=configs/deck_level4_agentlw2.json;constraints=results/constraints/open_haiku/constraints.json;prep=results/constraints/open_haiku/prep.json" &
run gold_v3 "$V3;deck=configs/deck_level4_gold.json;constraints=configs/constraints_gold.yaml;prep=configs/prep_gold.json" &
wait
echo done > $L/L4_V3_DONE
