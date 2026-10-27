#!/usr/bin/env bash
# Method baselines on the v3 lab (Results tab, Baselines panel). Every run knows all real limits and uses the
# gold labware and prep, so only the method differs; base_ours_v3 is ours (throughput-aware BO + CP-SAT repair).
#   bash scripts/run_baselines.sh [VARIANT ...]     # default: all; skips variants whose results exist
# LAB=agent: the web replay's lab instead (the agent's limits, labware and prep, seeds 0-2), every method with the
# same scripted data review (cleaning added at an assumed 15 min, as the Haiku review did); variants *_agentlab.
set -u
export PYTHONHASHSEED=0 PYTHONPATH=src LABLOOP_TORCH_THREADS=2 MUJOCO_GL=egl
P=${P:-.venv/bin/python}; J=${J:-8}; N=${N:-8}; S0=${S0:-10}; W=${W:-3}; OUT=${OUT:-results/m9}  # fresh seeds S0..S0+N-1, unseen while choosing our settings
L=results/logs; mkdir -p $L
GOLD="deck=configs/deck_level4_gold.json;sim3d={misplace_sigma_mm: 1.5, perception: true};truth=configs/truth_prep.yaml;plr=true;constraints=configs/constraints_gold.yaml;prep=configs/prep_gold.json"
OURS="scheduler_cfg={deterministic: true, stability: 0, objective: {makespan: 1, flow: 10}}"  # flow-weighted: reads back early feed the next design; no stability penalty in an unattended lab
DISP="scheduler_cfg={deterministic: true}"
SFX=v3
if [ "${LAB:-gold}" = agent ]; then
  SFX=agentlab; N=${N_AGENT:-3}; S0=${S0_AGENT:-0}
  GOLD="deck=configs/deck_level4_agentlw2.json;sim3d={misplace_sigma_mm: 1.5, perception: true};truth=configs/truth_prep.yaml;plr=true;constraints=results/constraints/open_haiku/constraints.json;prep=results/constraints/open_haiku/prep.json;review={data: true, sched: false, provider: scripted, hypotheses: results/constraints/open_haiku/summary.json, every_reads: 2, max_calls: 4}"
fi
spec() {  # variant -> --set string: discovery baselines keep our scheduler, scheduling baselines keep our designer
  case ${1%_*} in
    base_ours) echo "designer=throughput-bo;scheduler=repair;$OURS;$GOLD" ;;
    base_bo) echo "designer=bo-ts;scheduler=repair;$OURS;$GOLD" ;;
    base_lhs) echo "designer=lhs;scheduler=repair;$OURS;$GOLD" ;;
    base_random) echo "designer=random;scheduler=repair;$OURS;$GOLD" ;;
    base_grid) echo "designer=grid;scheduler=repair;$OURS;$GOLD" ;;
    base_fifo) echo "designer=throughput-bo;scheduler=fifo;$DISP;$GOLD" ;;
    base_priority) echo "designer=throughput-bo;scheduler=priority;$DISP;$GOLD" ;;
    base_cpsat) echo "designer=throughput-bo;scheduler=cpsat;$OURS;$GOLD" ;;
  esac
}
ALL=$(for k in ours bo lhs random grid fifo priority cpsat; do echo -n "base_${k}_$SFX "; done)
for v in ${@:-$ALL}; do [ -f $OUT/level4/$v.json ] || echo $v; done |
  xargs -P $J -I{} bash -c "$(declare -f spec); GOLD='$GOLD'; OURS='$OURS'; DISP='$DISP'; $P scripts/run_level.py --workers $W --config configs/level4.yaml --seeds $N --seed-start $S0 --out $OUT --variant \"{}:\$(spec {})\" > $L/{}.txt 2>&1"
echo done > $L/BASELINES_${SFX}_DONE_$(basename $OUT)
