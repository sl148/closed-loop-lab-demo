#!/usr/bin/env bash
# Level 4 sweep (rerun after an out-of-memory crash; Thompson-sampling path evaluation is now chunked).
set -u
export LABLOOP_TORCH_THREADS=2
uv run python scripts/run_level.py --workers 3 --config configs/level4.yaml --seeds 5 --out results/level4 \
   --variant "bo-fifo:designer=bo-ts;scheduler=fifo" \
   --variant "bo-repair:designer=bo-ts;scheduler=repair" \
   --variant "tbo-repair:designer=throughput-bo;scheduler=repair" > results/logs/level4.txt 2>&1
uv run python scripts/export_web.py > results/logs/export.txt 2>&1
uv run python scripts/make_results.py >> results/logs/export.txt 2>&1
echo done > results/logs/L4_DONE
