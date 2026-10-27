#!/usr/bin/env bash
# Live-mode combo runs under v3 (same variants as `run_l4_det.sh combos_v3`), several variants at once.
#   bash scripts/run_combos_v3.sh [VARIANT ...]     # default: all 12; skips variants whose results exist
set -u
export PYTHONHASHSEED=0 PYTHONPATH=src LABLOOP_TORCH_THREADS=2 MUJOCO_GL=egl
P=${P:-.venv/bin/python}; J=${J:-6}
L=results/logs; mkdir -p $L
BASE="designer=throughput-bo;scheduler=repair;scheduler_cfg={deterministic: true, stability: 0, objective: {makespan: 1, flow: 10}};deck=configs/deck_level4_agentlw2.json;sim3d={misplace_sigma_mm: 1.5, perception: true};truth=configs/truth_prep.yaml;plr=true;prep=results/constraints/open_haiku/prep.json"
R="review={data: true, sched: false, provider: scripted, hypotheses: configs/combos/hypotheses_changeover.json, every_reads: 2, max_calls: 4}"
spec() {  # variant name -> its --set string
  case $1 in
    combo_000_v3) echo "$BASE" ;;
    combo_111_v3) echo "$BASE;constraints=configs/constraints_gold.yaml" ;;
    combo_000_rev_v3) echo "$BASE;$R" ;;
    combo_*_rev_v3) c=${1#combo_}; echo "$BASE;constraints=configs/combos/combo_${c%_rev_v3}.yaml;$R" ;;
    combo_*_v3) c=${1#combo_}; echo "$BASE;constraints=configs/combos/combo_${c%_v3}.yaml" ;;
  esac
}
ALL="combo_000_v3 combo_111_v3 combo_001_v3 combo_010_v3 combo_011_v3 combo_100_v3 combo_101_v3 combo_110_v3 combo_000_rev_v3 combo_010_rev_v3 combo_100_rev_v3 combo_110_rev_v3"
for v in ${@:-$ALL}; do [ -f results/m9/level4/$v.json ] || echo $v; done |
  xargs -P $J -I{} bash -c "$(declare -f spec); BASE='$BASE'; R='$R'; $P scripts/run_level.py --workers 3 --config configs/level4.yaml --seeds 3 --out results/m9 --variant \"{}:\$(spec {})\" > $L/combo_{}.txt 2>&1"
echo done > $L/L4_COMBOS_V3_DONE
