# labloop: a self-driving lab for drug-combination screening (simulated)

A closed-loop lab that decides which drug combinations to test, lays them out on plates, plans the deck and
pipetting, schedules every instrument, runs the plates, recovers from failures, and learns from the reads,
with no human in the loop. The biology is a response model fitted to real screening data (O'Neil et al. 2016).
The contribution is the automation: experiment selection, scheduling, deck and protocol design, and recovery.

In plain words: the lab hunts for two or three cancer drugs that work better together than apart. Every test
takes days in an incubator with limited room, so *what* to test next and *when* each machine runs both matter.

**Live demo:** https://sl148.github.io/closed-loop-lab-demo/ (recorded runs; the page's Live mode reruns the
agent with your own Claude when opened inside claude.ai).

## Agentic loop

Nobody tells the lab its physical limits. Before launch a Claude agent (Haiku) reads the lab's document
archive (assay protocol, compound datasheets, cell-culture guide, consumables catalog), brainstorms which
timing limits might exist, and proposes each one only with a verbatim quote that contains the number. It also
picks the labware (plate, tips, dispense modes) and plans the liquid prep (stocks, dilution series, cell
reservoirs), checked on a simulated Hamilton STARlet in PyLabRobot. During the run, statistics on the lab's
own reads call the agent back: a data review adds a limit the documents never stated and quarantines the
plates it spoiled; a schedule review re-checks costly assumed values. Bayesian optimization picks the plates,
CP-SAT repair schedules the robots, and every plate and dilution batch is pipetted on the simulated STARlet.

Level 4 with hidden limits (read equilibration, drug-dilution shelf life, cell-line changeover), liquid prep,
labware and pipetting truth; 3 seeds, deterministic scheduling:

| Run | Synergy missed (regret) | True best pairs found | Wrong values kept | Invalid plates |
|---|---|---|---|---|
| No limits | 0.74 | 0 / 6 | 58 | 34 |
| Agent, before launch | 0.29 | 1 / 6 | 453 | 12 |
| Agent + in-run reviews | 0.27 | 2 / 6 | 90 | 12 |
| All real limits (gold) | 0.09 | 5 / 6 | 3 | 3 |

- Before launch the agent grounds 2 of the 3 real limits (drug dilutions within 1 day, Cayman datasheet;
  a wait before reading, TB288 p. 7) and misses cell-line cleaning, which no document quantifies.
- The data review then flags plates read after an uncleaned cell-line switch, adds a cleaning step as a
  soft, assumed limit, and drops those plates: wrong values kept fall from 453 to 90 (all 3 seeds).
- The remaining gap to gold comes from values, not missing limits: the agent's 10 min pre-read wait (the
  protocol's room-temperature step is 30 min) and its slower labware choice (drug tips used fresh, 118 vs
  36 min per plate). With the agent's labware and prep but the gold limit values, regret is 0.08 (4 / 6).
- Agent cost per campaign: about $0.13 (Haiku): limits $0.09, labware $0.005, prep $0.03, reviews $0.01.

## Ours vs. method baselines

Same lab (all real limits known, gold labware and prep), so only the method differs; 8 fresh seeds (20-27),
none used while choosing our settings (`scripts/run_baselines.sh`, `scripts/export_baselines.py`; the web
page's Results tab toggles each baseline on and off). "Holds 90%" is when the mean of the reported best
pair's true synergy reaches 90% of the best possible and stays there.

| Method | Holds 90% of best pair | Synergy found | Best pair found | Invalid plates |
|---|---|---|---|---|
| **Ours** (throughput-aware BO + CP-SAT repair) | **162 h** | 97% | 15/16 | 0.6 |
| Plain BO (our scheduler) | 272 h | 97% | 16/16 | 0.3 |
| Latin hypercube | 282 h | 96% | 15/16 | 0.9 |
| Random | 270 h | 93% | 10/16 | 0.1 |
| Dose grid | never | 81% | 8/16 | 0.1 |
| FIFO dispatch (our optimizer) | 287 h | 98% | 15/16 | 0.9 |
| Priority rules (our optimizer) | 252 h | 92% | 13/16 | 5.5 |
| Full CP-SAT re-solve (our optimizer) | never | 92% | 13/16 | 0.6 |

Ours reaches a good answer 90-125 h sooner than every baseline; final quality ties the best of them (within
about 1%). Getting there took fixes the baselines exposed: plans now cost a cell-line switch's reservoir
swap, keep slack before the drug-dilution deadline, and the throughput-aware optimizer cools its cost term
over the budget (Lee et al. 2020), since the best pair always sat at the slow 72 h read.

## Headline results

| Question | Result |
|---|---|
| Does online repair scale? (A) | At 64 plates in flight, repair re-plans within **5.4%** of the best schedule and moves **32 ops**; a 1 s full re-solve lands **35.8%** off and moves **244**. |
| Does throughput-aware BO help? (B) | Choosing each plate's cell line and read time by expected gain per hour of lab time finishes the 15-plate campaign in **248 h vs 416 h** for BO-then-schedule, at similar final regret. |
| Can an LLM run failure recovery? (C) | Claude (schema-validated, rule fallback) recovers in **8.3 vs 11.5 min** per failure, accepts **0** corrupt wells, needs **0** fallbacks, and replays bit-identically with no API calls. |
| Does deck design matter? | Optimizing carrier placement, tip types and tip reuse cuts liquid-handler time from **94 to 14 min** per 384-well plate, and the async campaign from **161 to 150 h**. |
| Is the solver sound? | CP-SAT reaches the known optimum on ft06, ft10, la01, ta01, ta02 and Brandimarte mk01-03. |
| Is the motion physical? | Every arm route clears the STARlet housing in MuJoCo (0 collisions, 0.0 mm placement error); arm and gantry never share the deck. |

Numbers are means over seeds; `RESULTS.md` has every table with spreads, and the web replay has the curves.

## How the loop works

```
choose experiments -> lay out plate -> plan deck + pipetting -> schedule instruments
        ^                                                              |
        |                                                              v
  update model  <-  read plate  <-  incubate 24-72 h  <-  dispense  (failures -> recover -> re-plan)
```

| Layer | Where | What it does |
|---|---|---|
| Response model | `oracle/` | Hill + MuSyC pair surfaces fitted to the O'Neil screen (DrugComb v1.4); time-dependent effect; read noise, edge bias, pipetting CV. |
| Experiment selection | `design/` | Grid, random, LHS, active learning, BoTorch BO (Thompson sampling, qEI), and **throughput-aware async BO** (B). |
| Plate layout | `layout/` | Edge-aware placement with dose-sorted columns so each 8-channel pass shares a source. |
| Deck + protocol | `protocol/` | Carrier rails (kept clear of the waste block), tip types, tip reuse by non-contact dispensing; per-plate pass planner timed with gantry kinematics; deck optimizer over 25k layouts. |
| Scheduling | `schedule/` | Flexible job shop with time lags, a cumulative incubator, downtime, and **deck zones** (load to unload holds the liquid handler). FIFO, priority, CP-SAT, and **online repair** (A, LNS around the disruption, warm-started, stability-penalized). An independent validator checks every plan. |
| Execution | `execution/` | SimPy lab: dispatches the plan, samples overruns and failures, re-plans; tracks tips and swaps used tip boxes per rack (used box to waste bin, new box from hotel). Every protocol also runs on a simulated Hamilton STARlet in **PyLabRobot** with tip and volume tracking. |
| Recovery | `failures/`, `agent/` | Typed failure events to recovery actions; rule policy and **Claude policy** (C) with logged, replayable decisions. |
| 3D workcell | `sim3d/` | MuJoCo STARlet (housing, gantry, waste block), PF400-class SCARA, hotel, incubator, reader, tip-box hotel and bin. Fold-and-stage arm motions; calibrates move times, tip-pickup failure, and camera detection of misplaced plates. |
| Replay site | `web/` | Static console: 3D workcell with camera views, pipetting panel, Gantt, a Discovery panel (best recipe so far vs best possible), and a plain-language explainer. |

## Levels

| Level | Scale | Uncertainty |
|---|---|---|
| 0 | 2 drugs, 1 cell line, 96-well | none |
| 1 | 2 drugs, 2 cell lines x 3 read times, arm transport | none |
| 2 | 3 drugs x 3 cell lines x 3 read times (O'Neil), 384-well | read noise, edge bias, pipetting error |
| 3 | Level 2 | + duration variance, tip / transfer / misplacement / measurement failures |
| 4 | 4 drugs x 4 cell lines, two concurrent campaigns, duplicated devices | + breakdowns, reagent expiry |

## Run

```
uv sync
uv run pytest                                                   # 54 tests
uv run python scripts/run_level.py --config configs/level0.yaml --seeds 3
uv run python scripts/bench_solvers.py                          # solver check on JSSP / FJSP
uv run python scripts/optimize_deck.py --config configs/level2.yaml
uv run python scripts/sim3d_calibrate.py                        # MuJoCo motion / tip / camera calibration
bash scripts/run_all.sh                                         # all experiments (hours; use a server)
bash scripts/run_console.sh                                     # the web Console's run (GP belief + camera logs)
uv run python scripts/export_web.py && uv run python scripts/make_results.py
cd web && python -m http.server 8765                            # replay at http://127.0.0.1:8765
uv run python scripts/eval_recovery.py --provider replay        # recovery policy, no API calls
```

Python 3.12 via uv (`link-mode = "copy"` because the repo lives on OneDrive). Data: `data/oneil_blocks.csv.gz`
is extracted by `scripts/extract_oneil.py` from the DrugComb v1.4 dump on Zenodo.

## Honest limits

- The agent's archive is 7 documents; the hidden limits are designed so the documents ground two of three.
- Live mode replays recorded lab runs: the browser reruns the agent, the deck search and the camera check,
  then plays the recorded run made with the limits the live agent found.
- The biology is an emulator: pair surfaces come from 4x4 dose grids, and synergy beyond pairs is additive.
- Failure rates are configured, except tip pickup, which comes from the MuJoCo contact model.
- No liquid physics: PyLabRobot tracks volumes; dispensing accuracy is a noise term.
- A tip-box swap replaces every empty or partly used rack at once.
- Batch MC acquisitions (qEI, qUCB) are impractical at ~300-point plates (about 1.4 h per seed vs minutes for
  Thompson sampling); qUCB was not run to completion.
- CP-SAT runs in experiments A-C use wall-clock limits, so they are not bit-reproducible; the agentic Level 4
  runs use deterministic mode (one worker, deterministic time limit) and replay bit-identically.
- Arm motion is a fixed set of taught routes checked for collisions in MuJoCo, not an online motion planner.
- In campaigns, whether the camera flags a misplaced plate is sampled from the detection rate measured in
  MuJoCo; the web viewer redraws the logged offset (x10) rather than a rendered camera frame.
