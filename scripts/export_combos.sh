#!/usr/bin/env bash
# Live-mode replays (web/data/replay_combo_<EDC>[_rev].json) from the combos_v3 runs (scripts/run_l4_det.sh combos_v3).
#   bash scripts/export_combos.sh
set -eu
export PYTHONPATH=src
P=${P:-.venv/bin/python}
for c in 000 001 010 011 100 101 110 111 000_rev 010_rev 100_rev 110_rev; do
  $P scripts/export_loop.py --variant combo_${c}_v3 --seed 1 --name combo_$c --compare v3 \
    --deck configs/deck_level4_agentlw2.json --label "This combination"
done
