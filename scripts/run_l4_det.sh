#!/usr/bin/env bash
# Level 4, deterministic scheduling (M10): reruns the variants the web page compares.
#   bash scripts/run_l4_det.sh truth            # no limits (blind_det) and all real limits (gold_det)
#   bash scripts/run_l4_det.sh agent [RUN_DIR]  # the agent's pre-launch limits (agent_open_det)
#   bash scripts/run_l4_det.sh combos           # every partial combination of the real limits (configs/combos)
#   bash scripts/run_l4_det.sh branches         # combos without cleaning + a scripted data review that adds it
#   bash scripts/run_l4_det.sh labware          # all real limits + labware truth: fastest deck vs. right labware
set -u
export PYTHONHASHSEED=0 PYTHONPATH=src LABLOOP_TORCH_THREADS=2 MUJOCO_GL=egl
P=${P:-.venv/bin/python}
L=results/logs; mkdir -p $L
BASE="designer=throughput-bo;scheduler=repair;scheduler_cfg={deterministic: true};deck=configs/deck_level4.json;sim3d={misplace_sigma_mm: 1.5, perception: true};truth=configs/truth.yaml"
case "${1:-truth}" in
  truth) $P scripts/run_level.py --workers 6 --config configs/level4.yaml --seeds 3 --out results/m9 \
           --variant "blind_det:$BASE" --variant "gold_det:$BASE;constraints=configs/constraints_gold.yaml" > $L/l4_det_truth.txt 2>&1
         echo done > $L/L4_DET_TRUTH_DONE ;;
  agent) RUN=${2:-results/constraints/open_haiku}
         $P scripts/run_level.py --workers 3 --config configs/level4.yaml --seeds 3 --out results/m9 \
           --variant "agent_open_det:$BASE;constraints=$RUN/constraints.json" > $L/l4_det_agent.txt 2>&1
         echo done > $L/L4_DET_AGENT_DONE ;;
  combos) V=()
         for f in configs/combos/combo_*.yaml; do c=$(basename $f .yaml); V+=(--variant "${c}_det:$BASE;constraints=$f"); done
         $P scripts/run_level.py --workers 9 --config configs/level4.yaml --seeds 3 --out results/m9 "${V[@]}" > $L/l4_det_combos.txt 2>&1
         echo done > $L/L4_DET_COMBOS_DONE ;;
  branches) R="review={data: true, sched: false, provider: scripted, hypotheses: configs/combos/hypotheses_changeover.json, every_reads: 2, max_calls: 4}"
         V=(--variant "combo_000_rev_det:$BASE;$R")
         for c in 010 100 110; do V+=(--variant "combo_${c}_rev_det:$BASE;constraints=configs/combos/combo_${c}.yaml;$R"); done
         $P scripts/run_level.py --workers 6 --config configs/level4.yaml --seeds 3 --out results/m9 "${V[@]}" > $L/l4_det_branches.txt 2>&1
         echo done > $L/L4_DET_BRANCHES_DONE ;;
  labware) LW="${BASE/truth=configs\/truth.yaml/truth=configs/truth_labware.yaml};constraints=configs/constraints_gold.yaml"
         $P scripts/run_level.py --workers 6 --config configs/level4.yaml --seeds 3 --out results/m9            --variant "lw_fast_det:$LW" --variant "lw_right_det:${LW/deck_level4.json/deck_level4_labware.json}" > $L/l4_det_labware.txt 2>&1
         echo done > $L/L4_DET_LABWARE_DONE ;;
  lw_headline) LW="${BASE/truth=configs\/truth.yaml/truth=configs/truth_labware.yaml}"  # all real limits + labware = lw_right_det
         $P scripts/run_level.py --workers 6 --config configs/level4.yaml --seeds 3 --out results/m9 --variant "blind_lw_det:$LW"            --variant "agent_open_lw_det:${LW/deck_level4.json/deck_level4_agentlw.json};constraints=results/constraints/open_haiku/constraints.json" > $L/l4_det_lw_headline.txt 2>&1
         echo done > $L/L4_DET_LW_HEADLINE_DONE ;;
  fixcheck) LW="${BASE/truth=configs\/truth.yaml/truth=configs/truth_labware.yaml}"; CF="confirm={plates: 2, candidates: 8, replicates: 6}"
         AG="${LW/deck_level4.json/deck_level4_agentlw.json};constraints=results/constraints/open_haiku/constraints.json"
         R="review={data: true, sched: false, provider: scripted, hypotheses: results/constraints/open_haiku/summary.json, every_reads: 2, max_calls: 4}"
         $P scripts/run_level.py --workers 9 --config configs/level4.yaml --seeds 3 --out results/m9            --variant "lw_right_conf_det:${LW/deck_level4.json/deck_level4_labware.json};constraints=configs/constraints_gold.yaml;$CF"            --variant "agent_scr_q_lw_det:$AG;$R" --variant "agent_scr_qc_lw_det:$AG;$R;$CF" > $L/l4_det_fixcheck.txt 2>&1
         echo done > $L/L4_DET_FIXCHECK_DONE ;;
  v2) V2="${BASE/truth=configs\/truth.yaml/truth=configs/truth_prep.yaml}"  # labware + prep truth; level4 confirms
         $P scripts/run_level.py --workers 9 --config configs/level4.yaml --seeds 3 --out results/m9            --variant "blind_v2:$V2;prep=default"            --variant "agent_open_v2:${V2/deck_level4.json/deck_level4_agentlw.json};constraints=results/constraints/open_haiku/constraints.json;prep=results/constraints/open_haiku/prep.json"            --variant "gold_v2:${V2/deck_level4.json/deck_level4_labware.json};constraints=configs/constraints_gold.yaml;prep=configs/prep_gold.json" > $L/l4_v2.txt 2>&1
         echo done > $L/L4_V2_DONE ;;
  v3) V3="${BASE/truth=configs\/truth.yaml/truth=configs/truth_prep.yaml};plr=true"  # + pipetting truth, PyLabRobot in the loop
         $P scripts/run_level.py --workers 9 --config configs/level4.yaml --seeds 3 --out results/m9            --variant "blind_v3:$V3;prep=default"            --variant "agent_open_v3:${V3/deck_level4.json/deck_level4_agentlw2.json};constraints=results/constraints/open_haiku/constraints.json;prep=results/constraints/open_haiku/prep.json"            --variant "gold_v3:${V3/deck_level4.json/deck_level4_gold.json};constraints=configs/constraints_gold.yaml;prep=configs/prep_gold.json" > $L/l4_v3.txt 2>&1
         echo done > $L/L4_V3_DONE ;;
  combos_v3) V3="${BASE/truth=configs\/truth.yaml/truth=configs/truth_prep.yaml};plr=true"  # live mode under v3: agent labware + prep, each combination of real limits
         AG="${V3/deck_level4.json/deck_level4_agentlw2.json};prep=results/constraints/open_haiku/prep.json"
         R="review={data: true, sched: false, provider: scripted, hypotheses: configs/combos/hypotheses_changeover.json, every_reads: 2, max_calls: 4}"
         V=(--variant "combo_000_v3:$AG" --variant "combo_111_v3:$AG;constraints=configs/constraints_gold.yaml" --variant "combo_000_rev_v3:$AG;$R")
         for f in configs/combos/combo_*.yaml; do c=$(basename $f .yaml); V+=(--variant "${c}_v3:$AG;constraints=$f"); done
         for c in 010 100 110; do V+=(--variant "combo_${c}_rev_v3:$AG;constraints=configs/combos/combo_${c}.yaml;$R"); done
         $P scripts/run_level.py --workers ${W:-9} --config configs/level4.yaml --seeds 3 --out results/m9 "${V[@]}" > $L/l4_combos_v3.txt 2>&1
         echo done > $L/L4_COMBOS_V3_DONE ;;
esac
