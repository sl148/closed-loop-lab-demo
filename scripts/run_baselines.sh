#!/usr/bin/env bash
# Method baselines on the v3 lab (Results tab, Baselines panel). Every run knows all real limits and uses the
# gold labware and prep, so only the method differs; base_ours_v3 is ours (throughput-aware BO + CP-SAT repair).
#   bash scripts/run_baselines.sh [VARIANT ...]     # default: all; skips variants whose results exist
set -u
export PYTHONHASHSEED=0 PYTHONPATH=src LABLOOP_TORCH_THREADS=2 MUJOCO_GL=egl
P=${P:-.venv/bin/python}; J=${J:-8}; N=${N:-8}; S0=${S0:-10}  # fresh seeds S0..S0+N-1, unseen while choosing our settings
L=results/logs; mkdir -p $L
GOLD="deck=configs/deck_level4_gold.json;sim3d={misplace_sigma_mm: 1.5, perception: true};truth=configs/truth_prep.yaml;plr=true;constraints=configs/constraints_gold.yaml;prep=configs/prep_gold.json"
OURS="scheduler_cfg={deterministic: true, stability: 0, objective: {makespan: 1, flow: 10}}"  # flow-weighted: reads back early feed the next design; no stability penalty in an unattended lab
DISP="scheduler_cfg={deterministic: true}"
spec() {  # variant -> --set string: discovery baselines keep our scheduler, scheduling baselines keep our designer
  case $1 in
    base_ours_v3) echo "designer=throughput-bo;scheduler=repair;$OURS;$GOLD" ;;
    base_bo_v3) echo "designer=bo-ts;scheduler=repair;$OURS;$GOLD" ;;
    base_lhs_v3) echo "designer=lhs;scheduler=repair;$OURS;$GOLD" ;;
    base_random_v3) echo "designer=random;scheduler=repair;$OURS;$GOLD" ;;
    base_grid_v3) echo "designer=grid;scheduler=repair;$OURS;$GOLD" ;;
    base_fifo_v3) echo "designer=throughput-bo;scheduler=fifo;$DISP;$GOLD" ;;
    base_priority_v3) echo "designer=throughput-bo;scheduler=priority;$DISP;$GOLD" ;;
    base_cpsat_v3) echo "designer=throughput-bo;scheduler=cpsat;$OURS;$GOLD" ;;
  esac
}
ALL="base_ours_v3 base_bo_v3 base_lhs_v3 base_random_v3 base_grid_v3 base_fifo_v3 base_priority_v3 base_cpsat_v3"
for v in ${@:-$ALL}; do [ -f results/m9/level4/$v.json ] || echo $v; done |
  xargs -P $J -I{} bash -c "$(declare -f spec); GOLD='$GOLD'; OURS='$OURS'; DISP='$DISP'; $P scripts/run_level.py --workers 3 --config configs/level4.yaml --seeds $N --seed-start $S0 --out results/m9 --variant \"{}:\$(spec {})\" > $L/{}.txt 2>&1"
echo done > $L/BASELINES_V3_DONE
