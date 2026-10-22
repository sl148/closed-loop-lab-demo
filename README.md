# labloop: a closed-loop lab automation demo for drug-combination screening

A simulated self-driving lab that runs **design → plate layout → scheduling → dispense / incubate / read →
failure recovery → model update** for drug-combination synergy screening. The biology is an oracle fitted to
real screening data. The contribution is the automation layer: which experiments to run, where they go on
the plate, when each instrument runs, and how the lab recovers when something fails.

The web replay (`web/`) shows every result below, a recorded run you can scrub through, and the 3D workcell.

## What is in it

| Layer | Module | What it does |
|---|---|---|
| Oracle | `oracle/emulator.py`, `oracle/surface.py` | Single-agent Hill fits plus MuSyC pair surfaces fitted to the **O'Neil et al. 2016** screen (DrugComb v1.4 dose-level dump; 22,737 drug-pair x cell-line blocks). Pair synergy fades below the screened dose range, so the oracle only claims synergy the data supports. 3-4 drug combinations are Bliss plus summed pair terms (an assumption, stated). Effect deepens with exposure time (24/48/72 h). |
| Measurement | `oracle/noise.py`, `execution/reader.py` | Gaussian read noise, edge-well evaporation bias, pipetting CV; per-plate control normalization (region-wise when the layout puts vehicle controls in both the edge and the interior). |
| Experiment selection | `design/` | Grid, random, Latin hypercube, active learning (uncertainty sampling), BoTorch BO (batch Thompson sampling, qUCB, qLogNEI) on Bliss excess. |
| **B: throughput-aware BO** | `design/throughput.py` | Chooses which cell line and timepoint each plate serves by expected improvement per **hour-to-result**, where hours come from inserting the plate into the lab's live plan. Runs asynchronously with in-flight wells as fantasies. |
| Plate layout | `layout/layouts.py` | Sequential, random, and edge-aware + pass-minimizing (samples in interior wells, dose-sorted columns so each 8-channel pass shares sources). |
| Scheduling model | `schedule/model.py`, `schedule/validator.py` | Flexible job shop with min/max time lags, a cumulative incubator, alternative devices, downtime, pinned running ops. An independent validator checks every schedule. |
| Schedulers | `schedule/serial.py`, `schedule/cpsat.py` | FIFO, priority rules, and CP-SAT (optional intervals, NoOverlap / Cumulative). CP-SAT reaches the known optimum on ft06/ft10/la01/ta01/ta02 and Brandimarte mk01-03 (`scripts/bench_solvers.py`). |
| **A: online repair** | `schedule/repair.py` | On every new batch, overrun, failure or breakdown: frees only the affected neighborhood (new, invalidated, conflicting and near-term jobs), pins the rest to the previous plan, warm-starts CP-SAT with a stability term, and falls back to a wider window, a full solve, then the serial scheduler. |
| Execution | `execution/lab.py` | SimPy online lab: dispatches the live plan, samples duration noise and failures, detects plan-breaking overruns, calls the scheduler to replan. |
| PyLabRobot | `execution/plr_bridge.py` | Runs each plate's pipetting protocol on a simulated Hamilton STARlet (chatterbox backend) with tip and volume tracking; its pass count matches the duration model exactly. It caught two real protocol errors during development (reservoir running dry, tips running out). |
| Failure adapter | `failures/` | Standard failure events (tip pickup, liquid transfer, misplaced plate, device down, measurement, timing window) → recovery actions (retry, substitute, remeasure, reassign, human, invalidate, retest). |
| **C: LLM recovery policy** | `failures/policies/llm.py`, `agent/` | Claude proposes recovery actions as schema-validated structured output; the proposal is checked (actions, bounds, wells, devices) and any error or invalid proposal falls back to the rule policy. Decisions are logged and replay bit-identically with zero API calls. |
| 3D workcell | `sim3d/` | MuJoCo cell built from PyLabRobot deck coordinates: STARlet deck, 8-channel gantry, PF400-class SCARA, hotel / incubator / reader. Measures transfer and pipetting times, checks collisions and placement, sets the tip-pickup failure rate from contact physics, and detects misplaced plates with an overhead camera. |

## Levels

| Level | Scale | Uncertainty |
|---|---|---|
| 0 | 2 drugs, 1 cell line, 8x8 matrix, one 96-well plate per round | none; every protocol also run through PyLabRobot |
| 1 | 2 drugs, 2 cell lines x 3 timepoints, 4 plates per round, arm transport | none |
| 2 | 3 drugs x 3 cell lines x 3 timepoints (O'Neil), 384-well, 3 plates per round | 7% read noise, edge bias, pipetting error |
| 3 | Level 2 | + durations ±20%, tip / transfer / misplacement / measurement failures |
| 4 | 4 drugs x 4 cell lines x 3 timepoints (49k conditions), two concurrent campaigns, duplicated devices | + breakdowns, reagent expiry |

## Results

Filled in from `results/` by `scripts/export_web.py`; see `RESULTS.md` for the numbers behind each claim and the
web replay for the curves.

## Run

```
uv sync
uv run pytest                                            # unit + integration tests
uv run python scripts/run_level.py --config configs/level0.yaml --seeds 3
uv run python scripts/bench_solvers.py                   # solver sanity on JSSP / FJSP instances
bash scripts/run_all.sh                                  # every experiment (a few hours, 20 cores)
uv run python scripts/sim3d_calibrate.py                 # MuJoCo motion / tip / camera calibration
uv run python scripts/export_web.py                      # bundles for web/
cd web && python -m http.server 8765                     # local replay at http://127.0.0.1:8765
uv run python scripts/eval_recovery.py --provider scripted   # recovery-policy harness, offline
uv run python scripts/eval_recovery.py --provider live       # rules vs Claude (uses API credits)
```

Data: `data/oneil_blocks.csv.gz` is extracted by `scripts/extract_oneil.py` from the first ~280 MB of the DrugComb
v1.4 dump on Zenodo (record 18449193); the command is in the script's docstring.

Python 3.12 (`.python-version`). `link-mode = "copy"` is pinned because the repo lives on OneDrive.

## Honest limits

- The biology is an emulator. Pair surfaces come from 4x4 dose grids; ladder doses below the screened range are
  tapered, and multi-drug synergy beyond pairs is not modeled unless planted.
- A 384-well combination plate takes ~6 h on a single 8-channel head in this model, which makes the liquid
  handler the bottleneck the schedulers fight over. A real lab would often use an acoustic dispenser for this
  step; the scheduling gains depend on that bottleneck.
- Failure probabilities are configured, except tip pickup, which is calibrated from the MuJoCo contact model.
- CP-SAT runs use wall-clock limits and 8 workers, so runs with CP-SAT are not bit-reproducible; FIFO and priority
  runs are.
