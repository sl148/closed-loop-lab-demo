"""Export experiment results into the static web replay's data bundles (web/data/).



    uv run python scripts/export_web.py



Reads results/ (sweeps from scripts/run_all.sh, sim3d calibration) and writes:

  web/data/results.json         seed-aggregated curves and metric tables per experiment

  web/data/replay_<name>.json   one full run per scenario: ops, failures, replans, plates, progress

  web/data/cell.json            3D cell geometry + downsampled episode + calibration curves

Bundle schema version: labloop-web/1. Missing experiments are skipped.

"""



from __future__ import annotations



import json

from pathlib import Path



import numpy as np



from labloop.metrics import run_metrics, summarize



RES = Path("results")

OUT = Path("web/data")

SCHEMA = "labloop-web/1"

KINDS = ["transfer", "dispense", "incubate", "read", "tip_swap"]



EXPERIMENTS = {

    "m2": {"dir": "m2/level2", "title": "Experiment selection (Level 2)", "x": "plates",

           "note": "Level 2 / 15 plates / 10 seeds / scheduler fixed",

           "labels": {"designer=grid": "Grid", "designer=random": "Random", "designer=lhs": "Latin hypercube",

                      "designer=al": "Active learning", "designer=bo-ts": "BO (Thompson)",

                      "designer=bo-qucb": "BO (qUCB)", "designer=bo-qei": "BO (qEI)"}},

    "m2_layout": {"dir": "m2_layout/level2", "title": "Plate layout (Level 2)", "x": "plates",

                  "note": "Level 2 / BO fixed / 10 seeds",

                  "labels": {"layout=sequential": "Sequential", "layout=random": "Random",

                             "layout=optimized": "Edge-aware + pass-minimizing"}},

    "m3": {"dir": "m3/level3", "title": "Scheduling under execution uncertainty (Level 3, sync)", "x": "t",

           "note": "Level 3 / failures / designer fixed / 10 seeds",

           "labels": {"scheduler=fifo": "FIFO", "scheduler=priority-lwr": "Priority (LWR)",

                      "scheduler=cpsat": "CP-SAT from scratch", "scheduler=repair": "Online repair"}},

    "m3_async": {"dir": "m3_async/level3", "title": "Scheduling with continuous arrivals (Level 3, async)", "x": "t",

                 "note": "Level 3 async / 6 plates in flight / 10 seeds",

                 "labels": {"scheduler=fifo": "FIFO", "scheduler=priority-lwr": "Priority (LWR)",

                            "scheduler=cpsat": "CP-SAT from scratch", "scheduler=repair": "Online repair"}},

    "m4": {"dir": "m4/level2", "title": "Closing the loop: when to run which plate (Level 2)", "x": "t",

           "note": "Level 2 / 15 plates / 10 seeds",

           "labels": {"bo-sync-fifo": "BO rounds + FIFO", "bo-sync-cpsat": "BO rounds + CP-SAT",

                      "bo-async-repair": "Async BO + repair", "tbo-async-repair": "Throughput-aware BO + repair"}},

    "level4_sim3d": {"dir": "level4_sim3d/level4", "title": "Level 4 on the calibrated workcell", "x": "t",

                     "note": "Level 4 / MuJoCo-calibrated / 5 seeds",

                     "labels": {"camera": "Camera detection", "always-detected": "Perfect detection"}},

    "level4": {"dir": "level4/level4", "title": "Everything at once (Level 4)", "x": "t",

               "note": "4 drugs x 4 cell lines / 2 campaigns / breakdowns / 5 seeds",

               "labels": {"bo-fifo": "BO + FIFO", "bo-repair": "BO + repair", "tbo-repair": "Throughput-aware BO + repair"}},

}



REPLAYS = {
    "console": ("console/level4/console.json", 0,
                "Level 4 on the optimized, MuJoCo-calibrated deck: two campaigns, camera checks, repair"),
    "m8": ("m8/level2/optimized-deck.json", 0, "Level 2 on the optimized deck: tip reuse, tip resupply by the arm"),
}





def clean(x):

    """NaN/inf -> None so the bundles are valid JSON."""

    if isinstance(x, dict):

        return {k: clean(v) for k, v in x.items()}

    if isinstance(x, list):

        return [clean(v) for v in x]

    if isinstance(x, float) and not np.isfinite(x):

        return None

    return x





def dump(obj) -> str:

    return json.dumps(clean(obj), separators=(",", ":"), allow_nan=False)





def regret_frac_curve(run: dict, x: str, grid: np.ndarray) -> np.ndarray:

    """Fraction of the best achievable synergy still missed by the recommended combination,

    averaged over campaigns, as a step function of time (h) or plates read."""

    curves = []

    for c in run["campaigns"].values():

        opt = c["optimum"]["excess"]

        xs = [0.0] + [(p["t"] / 60 if x == "t" else p["plates_read"]) for p in c["progress"]]

        ys = [1.0] + [(p["regret"] / opt if p["regret"] is not None else 1.0) for p in c["progress"]]

        idx = np.searchsorted(np.asarray(xs), grid, side="right") - 1

        curves.append(np.clip(np.asarray(ys)[idx], 0, 1))

    return np.mean(curves, axis=0)





def experiment(key: str, spec: dict) -> dict | None:

    d = RES / spec["dir"]

    files = sorted(d.glob("*.json")) if d.exists() else []

    files = [f for f in files if f.name != "summary.json"]

    if not files:

        return None

    variants = []

    data = {f.stem: json.loads(f.read_text()) for f in files}

    order = [k for k in spec["labels"] if k in data] + [k for k in data if k not in spec["labels"]]

    xmax = max((p["t"] / 60 if spec["x"] == "t" else p["plates_read"])

               for runs in data.values() for r in runs for c in r["campaigns"].values() for p in c["progress"])

    grid = np.linspace(0, xmax, 80)

    for k in order:

        runs = data[k]

        ys = np.stack([regret_frac_curve(r, spec["x"], grid) for r in runs])

        s = summarize(runs)

        variants.append({

            "key": k, "label": spec["labels"].get(k, k), "seeds": len(runs),

            "curve": {"median": np.round(np.median(ys, 0), 4).tolist(),

                      "lo": np.round(np.percentile(ys, 25, 0), 4).tolist(),

                      "hi": np.round(np.percentile(ys, 75, 0), 4).tolist()},

            "metrics": s,

        })

        if key.startswith(("m3", "level4")):

            rs = [x for r in runs for x in r["reschedules"]]

            variants[-1]["resolve_ms"] = sorted(round(1000 * x["time_s"], 2) for x in rs)

    out = {"title": spec["title"], "note": spec["note"], "x": spec["x"],
           "grid": np.round(grid, 3).tolist(), "variants": variants}
    if key == "m3_async":
        sh = RES / "m3_shadow/level3/scheduler=repair.json"
        if sh.exists():
            rs = [x for r in json.loads(sh.read_text()) for x in r["reschedules"] if "shadow_obj" in x]
            gaps = [100 * (x["obj"] - x["shadow_obj"]) / x["shadow_obj"] for x in rs if x["shadow_obj"] > 0]
            speed = [x["shadow_time_s"] / max(x["time_s"], 1e-4) for x in rs]
            out["shadow"] = {"n": len(rs), "gap_pct_median": round(float(np.median(gaps)), 3),
                             "gap_pct_p95": round(float(np.percentile(gaps, 95)), 3),
                             "gap_pct_max": round(float(np.max(gaps)), 3),
                             "speedup_median": round(float(np.median(speed)), 1),
                             "repair_ms_median": round(1000 * float(np.median([x["time_s"] for x in rs])), 1),
                             "full_s_median": round(float(np.median([x["shadow_time_s"] for x in rs])), 2)}
    return out


def short(content) -> str:
    if isinstance(content, str):
        return content
    return "|".join(f"{d:g}" for d in content["doses"])


def _trace_plate(args):
    """Run one plate's pipetting protocol through PyLabRobot with tracing (worker process):
    the planned protocol on the run's deck when it had one, else the fresh-tip baseline."""
    pid, p, deck = args
    from labloop.execution.plr_bridge import PLRBridge
    from labloop.types import Condition, PlateMap, PlateSpec

    pm = PlateMap(pid, PlateSpec(p["rows"], p["cols"]), p["cell_line"], p["timepoint_h"], p["campaign"])
    for name, c in p["wells"].items():
        r, col = ord(name[0]) - 65, int(name[1:]) - 1
        pm.wells[(r, col)] = c if isinstance(c, str) else Condition.from_dict(c)
    n_drugs = next((len(c["doses"]) for c in p["wells"].values() if not isinstance(c, str)), 0)
    if deck:
        from labloop.protocol.deck import DeckDesign
        from labloop.protocol.planner import plan_plate

        rep = PLRBridge(n_drugs).run_protocol(pm, plan_plate(pm, DeckDesign(**deck)))
    else:
        rep = PLRBridge(n_drugs, trace=True).run(pm)
    return pid, {"passes": rep.trace, "tips": rep.tips_used, "swaps": rep.rack_swaps, "refills": rep.refills,
                 "bad": rep.wells_bad}


def protocol_traces(run: dict, workers: int = 4) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_trace_plate, [(k, v, run.get("deck")) for k, v in run["plates"].items()]))


def drug_names(run: dict) -> list[str]:
    import yaml

    cfg = yaml.safe_load((Path("configs") / f"{run['name']}.yaml").read_text())
    return cfg["oracle"].get("drugs") or [d["name"] for d in next(iter(cfg["oracle"]["cell_lines"].values()))["drugs"]]


def op_route(o: dict) -> list:
    """[route, tip-carrier site] for the animation: transfers ("mv_lh_rackwaste_s1" -> "lh>rackwaste", 1)
    and LH CO-RE tip-box moves ("core_out_s3" -> "core_out", 3); [] for other ops."""
    import re

    name = o["op"].split(".")[1]
    m = re.search(r"_s(\d+)$", name)
    site = int(m.group(1)) if m else None
    if o["kind"] == "transfer":
        return [">".join(name[3:].split("_")[:2]), site]
    if name.startswith("core_"):
        return [name[:m.start()] if m else name, site]
    return []


def discovery(run: dict, campaign: str, read_t: dict, ladders: list | None = None) -> dict:
    """What the Discovery panel draws for one campaign: every tested dose combination with
    the plate read time and its true Bliss excess (simulator ground truth; the lab sees a
    noisy version), indexed onto per-drug dose levels."""
    import yaml

    from labloop.oracle.surface import build_oracle

    cfg = yaml.safe_load(Path(f"configs/{run['name']}.yaml").read_text())
    oracle = build_oracle(cfg["oracle"])
    seen = {}
    for pid, pl in run["plates"].items():
        if pl["campaign"] != campaign or pid not in read_t:
            continue
        for content in pl["wells"].values():
            if isinstance(content, str):
                continue
            key = (pl["cell_line"], pl["timepoint_h"], tuple(content["doses"]))
            if key not in seen or read_t[pid] < seen[key]:
                seen[key] = read_t[pid]
    opt = run["campaigns"][campaign]["optimum"]["condition"]
    n = len(opt["doses"])
    levels = [sorted({k[2][i] for k in seen} | {opt["doses"][i], 0.0}) for i in range(n)]
    for i, lad in enumerate(ladders or []):
        levels[i] = list(lad)
    idx = lambda doses: [levels[i].index(d) if d in levels[i] else min(range(len(levels[i])), key=lambda j: abs(levels[i][j] - d))
                         for i, d in enumerate(doses)]
    tested = sorted([t, line, tp, *idx(d), round(oracle.true_excess(line, d, tp), 4)] for (line, tp, d), t in seen.items())
    return {"levels": levels, "tested": tested, "opt_idx": idx(opt["doses"])}


def replay(run: dict, title: str, seed: int) -> dict:
    ops = sorted(run["schedule"], key=lambda o: (o["start"], o["op"]))
    plates = {}
    for pid, p in run["plates"].items():
        reads = {r["well"]: [r["viability"], int(r["valid"])] for r in p["reads"].values()} if p["reads"] else {}
        plates[pid] = {"campaign": p["campaign"], "cell_line": p["cell_line"], "timepoint_h": p["timepoint_h"],
                       "rows": p["rows"], "cols": p["cols"], "status": p["status"],
                       "wells": {w: short(c) for w, c in p["wells"].items()}, "reads": reads}
    failures = [{"t": e["t"], "type": e["event"]["type"], "plate": e["event"].get("plate"),
                 "resource": e["event"].get("resource"), "wells": e["event"].get("wells", []),
                 "attempt": e["event"].get("attempt", 1), "actions": [a["type"] + (f" {a['minutes']}m" if a["minutes"] else "")
                                                                       for a in e["decision"]["actions"]],
                 "rationale": e["decision"]["rationale"], "policy": e["decision"]["policy"]}
                for e in run["events"] if e["kind"] == "failure"]
    progress = {}
    read_t = {e["plate"]: e["t"] for e in run["events"] if e["kind"] == "read"}
    for name, c in run["campaigns"].items():
        opt = c["optimum"]["excess"]
        progress[name] = {"optimum": c["optimum"], "designer": c["designer"], "mode": c["mode"],
                          "points": [{"t": p["t"], "plates": p["plates_read"],
                                      "regret_frac": round(p["regret"] / opt, 4) if p["regret"] is not None else 1.0,
                                      "best": p["best"], "best_est": p["best_est"], "best_true": p.get("best_true")}
                                     for p in c["progress"]],
                          "discovery": discovery(run, name, read_t, c["belief"][0]["ladders"] if c.get("belief") else None),
                          "belief": [{"t": b["t"], "groups": b["groups"]} for b in c.get("belief", [])],
                          "decisions": c.get("group_decisions", [])[:200]}
    return {
        "schema": SCHEMA, "title": title, "seed": seed,
        "meta": {"scenario": run["name"], "scheduler": run["scheduler"], "policy": run["policy"],
                 "makespan": run["makespan"], "resources": run["resources"], "metrics": run_metrics(run)},
        "kinds": KINDS,
        "ops": [[o["job"], KINDS.index(o["kind"]), o["resource"], o["start"], o["end"], *op_route(o)] for o in ops],
        "failures": failures,
        "camera": [{k: e[k] for k in ("t", "plate", "dx_mm", "dy_mm", "offset_mm", "detected")} | {"route": op_route({"op": e["op"], "kind": "transfer"})[0]}
                   for e in run["events"] if e["kind"] == "camera_check"],
        "replans": [{"t": r["t"], "reasons": r["reasons"], "solver": r["solver"], "ms": round(1000 * r["time_s"], 1),
                     "moved": r["moved"], "free": r["n_free"], "ops": r["n_ops"]} for r in run["reschedules"]],
        "reads": [{"t": e["t"], "plate": e["plate"], "valid": e["n_valid"], "n": e["n"]}
                  for e in run["events"] if e["kind"] == "read"],
        "submits": {pid: e["t"] for e in run["events"] if e["kind"] == "submit" for pid in e["plates"]},
        "plates": plates, "progress": progress,
        "drugs": drug_names(run),
        "deck": run.get("deck"),
        "tips": [{k: v for k, v in e.items() if k != "left"} for e in run["events"]
                 if e["kind"] in ("tip_order", "tip_refill", "tip_mid_swap", "tip_swap")],
        "protocols": protocol_traces(run),
    }


def recovery() -> dict | None:
    """Rules vs Claude recovery (M6): summary plus one real rationale per failure type."""
    for prov in ("replay", "live"):
        p = RES / f"m6/summary_{prov}.json"
        if p.exists():
            break
    else:
        return None
    summary = json.loads(p.read_text())
    examples, counts = {}, {}
    for f in sorted((RES / "m6").glob("run_*_seed*.json")):
        if "scripted" in f.name:
            continue
        for e in json.loads(f.read_text()):
            if not e["decision"]["policy"].startswith("llm") or "->rules" in e["decision"]["policy"]:
                continue
            t = e["event"]["type"]
            acts = " + ".join(a["type"] for a in e["decision"]["actions"])
            counts.setdefault(t, {}).setdefault(acts, 0)
            counts[t][acts] += 1
            examples.setdefault(t, {"actions": acts, "rationale": e["decision"]["rationale"]})
    return {"summary": summary, "examples": examples, "counts": counts, "source": prov}


def cell() -> dict | None:
    d = RES / "sim3d"
    if not (d / "episode.json").exists():
        return None
    ep = json.loads((d / "episode.json").read_text())
    frames = ep["frames"][::2]
    from labloop.sim3d.cell import cell_spec

    spec = cell_spec()
    stations = {k: {"xy": list(v.center), "z": v.plate_z} for k, v in spec.stations.items()}
    return {
        "schema": SCHEMA, "geometry": json.loads((d / "cell_geometry.json").read_text()), "stations": stations,
        "frames": [{"t": f["t"], "phase": f["phase"], "b": f["bodies"]} for f in frames],
        "timing": ep["meta"]["timing_s"], "collisions": ep["meta"]["collisions"],
        "motion": json.loads((d / "motion_costs.json").read_text()),
        "tips": json.loads((d / "tips.json").read_text()),
        "perception": json.loads((d / "perception.json").read_text()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"schema": SCHEMA, "experiments": {}}
    bench = RES / "bench_solvers.json"
    if bench.exists():
        results["bench"] = json.loads(bench.read_text())
    for key, spec in EXPERIMENTS.items():
        e = experiment(key, spec)
        if e:
            results["experiments"][key] = e
            print("experiment", key, [v["key"] for v in e["variants"]])
    (OUT / "results.json").write_text(dump(results))
    index = []
    for name, (path, seed, title) in REPLAYS.items():
        p = RES / path
        if not p.exists():
            continue
        run = json.loads(p.read_text())[seed]
        (OUT / f"replay_{name}.json").write_text(dump(replay(run, title, seed)))
        index.append({"name": name, "title": title})
        print("replay", name)
        break
    results["replays"] = index
    results["m6"] = recovery()
    (OUT / "results.json").write_text(dump(results))
    c = cell()
    if c:
        (OUT / "cell.json").write_text(dump(c))
        print("cell", len(c["frames"]), "frames")
    for f in sorted(OUT.glob("*.json")):
        print(f"{f.name:28} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
