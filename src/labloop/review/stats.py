"""Which schedule factors go with bad data? (data-side review, M10)

Everything here works on what a real lab could observe: the plate reads, the executed schedule,
and the events it logged. Hidden-limit bookkeeping (truth events, corrupt-well counts) is never
read, so a flag here is evidence a person could check.

1. `plate_scores`: per plate, `over_frac` is the share of drug wells reading above the untreated
   controls by more than TAIL_K control SDs. Drugs only lower viability, and synergy lowers it more,
   so such wells point at something added to them (cells carried over from another line) or at
   measurement trouble. `ctrl_noise` is the spread of the plate's untreated controls against the
   lab's typical plate: a whole-plate problem (reading a plate that is still warm) raises it.
   (`plate_residuals` keeps a per-plate fit of the dose design for inspection; its misfit near
   synergy is too large for a contamination score.)
2. `plate_factors`: what happened to each plate, from the schedule (a cell-line switch on its liquid
   handler without a clean, minutes out of the incubator before the read, read lateness, drug age).
3. `check_hypotheses`: the agent's own unproven hypotheses (it suspected a limit but found no number)
   name a factor, a score and a direction before any data is seen, so only they are tested,
   Holm-corrected over that small family. This is the review's main test. Two methods:
   - "history" (default, for over_frac): each plate is anomalous if it has at least MIN_WELLS drug wells
     above the controls; are the exposed plates anomalous more often than the lab's historical rate p0
     (one-sided binomial over plates, so a cluster of bad wells on one plate counts once)? When at
     least 3 unexposed plates have been read, exposed plates must also be worse than them. Fires early
     (a handful of exposed plates) because it needs no comparison group.
   - "contrast": a one-sided Freedman-Lane permutation test, score ~ factor + cell line + timepoint.
     Needs plates on both sides, so it fires late when most plates are exposed.
4. `associate`: the same test over every (score, factor) pair, two-sided and Holm-corrected, for
   limits nobody hypothesized. Much weaker (16 tests), so it only catches large effects.
The permutation RNG is the review's own, never the lab's.
"""

from __future__ import annotations

import math

import numpy as np

SCORES = ("over_frac", "ctrl_noise")
MIN_WELLS = 2  # a plate with this many implausible wells counts as anomalous ("history" method)
P0 = 0.06  # historical share of anomalous plates in a clean lab (Level 4 runs with every limit kept)
TAIL_K = 4.0  # a drug well this many control SDs above the untreated controls is implausible
MAD = 1.4826


def _design(doses: np.ndarray) -> np.ndarray:
    """Intercept, one-hot dose level per drug (level 0 = absent is the baseline), pairwise log-dose terms."""
    cols = [np.ones(len(doses))]
    lv = []
    for j in range(doses.shape[1]):
        levels = np.unique(doses[:, j])
        idx = np.searchsorted(levels, doses[:, j])
        cols += [(idx == k).astype(float) for k in range(1, len(levels))]
        lv.append(idx - idx.mean())
    for a in range(len(lv)):
        for b in range(a + 1, len(lv)):
            cols.append(lv[a] * lv[b])
    return np.column_stack(cols)


def _huber(X: np.ndarray, y: np.ndarray, c: float = 1.345, iters: int = 30) -> np.ndarray:
    """Huber regression by iteratively reweighted least squares; returns residuals."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = MAD * np.median(np.abs(r - np.median(r))) or 1e-6
        w = np.minimum(1.0, c / np.maximum(np.abs(r / s), 1e-12))
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.allclose(new, beta, atol=1e-8):
            beta = new
            break
        beta = new
    return y - X @ beta


def plate_residuals(plate: dict) -> tuple[np.ndarray, dict]:
    """Residuals of log viability on the plate's own design (valid condition wells only), and controls."""
    doses, y, ctrl = [], [], {"NEG": [], "POS": []}
    for well, content in plate["wells"].items():
        rd = plate["reads"].get(well)
        if not rd or not rd.get("valid", True):
            continue
        if isinstance(content, str):
            ctrl.setdefault(content, []).append(rd["value"])
            continue
        doses.append(content["doses"])
        y.append(math.log(max(rd["viability"], 1e-4)))
    if len(y) < 40:
        return np.array([]), ctrl
    return _huber(_design(np.array(doses)), np.array(y)), ctrl


def plate_scores(run: dict, plates: list[str] | None = None) -> dict[str, dict]:
    """Anomaly scores per read plate. Viability is already relative to the untreated (NEG) controls;
    the control SD used is the larger of the plate's own and the lab's median over the same plates."""
    res = {}
    for pid in plates or list(run["plates"]):
        plate = run["plates"][pid]
        if not plate.get("reads"):
            continue
        v, neg = [], []
        for well, content in plate["wells"].items():
            rd = plate["reads"].get(well)
            if not rd or not rd.get("valid", True):
                continue
            if content == "NEG":
                neg.append(rd["viability"])
            elif not isinstance(content, str) and not _edge(well, plate["rows"], plate["cols"]):
                v.append(rd["viability"])  # edge wells carry the plate's edge bias
        if len(v) < 40 or len(neg) < 2:
            continue
        res[pid] = {"v": np.array(v), "neg_sd": float(np.std(neg, ddof=1))}
    if not res:
        return {}
    lab = float(np.median([x["neg_sd"] for x in res.values()]))
    out = {}
    for pid, x in res.items():
        sd = max(x["neg_sd"], lab)
        over = int(np.sum(x["v"] > 1 + TAIL_K * sd))
        out[pid] = {"over_frac": over / len(x["v"]), "over_n": over, "ctrl_noise": x["neg_sd"] / lab,
                    "neg_sd": x["neg_sd"], "max_viability": float(x["v"].max()), "n_wells": int(len(x["v"]))}
    return out


def _edge(well: str, rows: int, cols: int) -> bool:
    r, c = ord(well[0]) - 65, int(well[1:]) - 1
    return r in (0, rows - 1) or c in (0, cols - 1)


def plate_factors(run: dict) -> dict[str, dict]:
    """What happened to each plate, from the executed schedule and logged events (no hidden limits)."""
    rows = {r["op"]: r for r in run["schedule"]}
    release = {p: e["t"] for e in run["events"] if e["kind"] == "submit" for p in e["plates"]}
    cleaned = {e["plate"] for e in run["events"] if e["kind"] == "clean"}
    fails: dict[str, int] = {}
    for e in run["events"]:
        if e["kind"] == "failure":
            p = (e.get("event") or {}).get("plate")
            if p:
                fails[p] = fails.get(p, 0) + 1
    disp = sorted((r["start"], r["resource"], r["job"]) for r in run["schedule"]
                  if r["op"].endswith(".dispense") and r["job"] in run["plates"])
    prev_line: dict[str, str] = {}
    before: dict[str, str | None] = {}
    for _, lh, pid in disp:
        before[pid] = prev_line.get(lh)
        prev_line[lh] = run["plates"][pid]["cell_line"]
    out = {}
    for pid, plate in run["plates"].items():
        op = lambda s: rows.get(f"{pid}.{s}")
        d, inc, mv, rd = op("dispense"), op("incubate"), op("mv_incubator_reader"), op("read")
        if not (d and inc and mv and rd):
            continue
        switch = before.get(pid) not in (None, plate["cell_line"])
        out[pid] = {"lh_switch_uncleaned": float(switch and pid not in cleaned),
                    "lh_switch_cleaned": float(switch and pid in cleaned),
                    "eq_min": float(rd["start"] - mv["start"]),
                    "read_late_min": float(inc["end"] - inc["start"] - plate["timepoint_h"] * 60),
                    "drug_age_min": float(d["start"] - release.get(pid, d["start"])),
                    "rest_min": float(inc["start"] - d["start"]),
                    "lh2": float(d["resource"] == "lh2"),
                    "n_obs_failures": float(fails.get(pid, 0)),
                    "cell_line": plate["cell_line"], "timepoint_h": plate["timepoint_h"],
                    "lh": d["resource"], "prev_line": before.get(pid), "read_t": rd["start"]}
    return out


FACTORS = ("lh_switch_uncleaned", "lh_switch_cleaned", "eq_min", "read_late_min", "drug_age_min", "rest_min",
           "lh2", "n_obs_failures")
BINARY = {"lh_switch_uncleaned", "lh_switch_cleaned", "lh2"}


def _covariates(meta: list[dict]) -> np.ndarray:
    cols = [np.ones(len(meta))]
    for key in ("cell_line", "timepoint_h"):
        levels = sorted({m[key] for m in meta})
        cols += [np.array([m[key] == lv for m in meta], float) for lv in levels[1:]]
    Z = np.column_stack(cols)
    return Z[:, _independent(Z)]


def _independent(Z: np.ndarray) -> list[int]:
    keep: list[int] = []
    for j in range(Z.shape[1]):
        if np.linalg.matrix_rank(Z[:, keep + [j]]) > len(keep):
            keep.append(j)
    return keep


def freedman_lane(y: np.ndarray, x: np.ndarray, Z: np.ndarray, B: int, rng) -> tuple[float, float]:
    """Two-sided permutation p for x in y ~ x + Z, permuting residuals of the reduced model y ~ Z.
    Returns (p, effect) where effect is x's coefficient."""
    M = np.eye(len(y)) - Z @ np.linalg.pinv(Z)  # residual maker for Z
    xr = M @ x
    sxx = float(xr @ xr)
    if sxx < 1e-12:
        return 1.0, 0.0
    e = M @ y
    def tstat(er: np.ndarray) -> np.ndarray:  # er: (k, n) residualized responses
        b = er @ xr / sxx
        res = er - np.outer(b, xr)
        dof = max(1, len(y) - Z.shape[1] - 1)
        se = np.sqrt((res ** 2).sum(1) / dof / sxx)
        return b / np.maximum(se, 1e-12), b
    t0, b0 = tstat(e[None])
    perms = np.array([rng.permutation(len(y)) for _ in range(B)])
    tb, _ = tstat((M @ e[perms].T).T)
    p = (1 + int(np.sum(np.abs(tb) >= abs(t0[0])))) / (B + 1)
    return p, float(b0[0])


def associate(scores: dict[str, dict], factors: dict[str, dict], B: int = 4999, seed: int = 0,
              alpha: float = 0.01, min_arm: int = 5, min_n: int = 12,
              min_effect: dict | None = None) -> list[dict]:
    """Every (score, factor) test, Holm-corrected, sorted by p; `flag` marks the ones to act on."""
    min_effect = min_effect or {"over_frac": 0.002, "ctrl_noise": 0.3}
    pids = [p for p in scores if p in factors]
    if len(pids) < min_n:
        return []
    meta = [factors[p] for p in pids]
    Z = _covariates(meta)
    rng = np.random.default_rng(seed)
    tests = []
    for f in FACTORS:
        x = np.array([m[f] for m in meta], float)
        if np.ptp(x) == 0:
            continue
        arms = (int((x > 0).sum()), int((x == 0).sum())) if f in BINARY else (len(x), len(x))
        for s in SCORES:
            y = np.array([scores[p][s] for p in pids], float)
            p, eff = freedman_lane(y, x, Z, B, rng)
            if f not in BINARY:  # per SD of the factor, comparable across units
                eff *= float(x.std())
            tests.append({"score": s, "factor": f, "p": p, "effect": eff, "n": len(pids), "arms": arms,
                          "binary": f in BINARY})
    m, run_max = len(tests), 0.0
    for i, t in enumerate(sorted(tests, key=lambda t: t["p"])):  # Holm step-down, kept monotone
        run_max = max(run_max, min(1.0, t["p"] * (m - i)))
        t["p_holm"] = run_max
        t["flag"] = (t["p_holm"] < alpha and t["effect"] >= min_effect[t["score"]]
                     and min(t["arms"]) >= (min_arm if t["binary"] else min_n))
    return sorted(tests, key=lambda t: t["p"])


# What a hypothesis would show in the data: (factor, score, sign of the expected effect). Only limits
# whose breach leaves a mark these scores can see are testable; the rest stay with the documents.
def hypothesis_test(h: dict) -> tuple[str, str, int] | None:
    kind, steps = h.get("kind"), set(h.get("steps") or [])
    if kind == "changeover" and "dispense" in steps:
        return "lh_switch_uncleaned", "over_frac", 1  # carried-over cells read above the controls
    if kind == "min_between" and "read" in steps:
        return "eq_min", "ctrl_noise", -1  # a plate read too soon after the incubator is noisier
    return None


def _history_test(scores: dict, factors: dict, f: str, p0: float, min_exposed: int = 3) -> dict:
    from scipy.stats import binomtest

    pids = [p for p in scores if p in factors]
    bad = {p: int(scores[p].get("over_n", 0) >= MIN_WELLS) for p in pids}
    e = [p for p in pids if factors[p][f] > 0]
    u = [p for p in pids if factors[p][f] == 0]
    row = {"arms": (len(e), len(u)), "exposed_anomalous": sum(bad[p] for p in e),
           "unexposed_anomalous": sum(bad[p] for p in u), "p0": p0}
    if len(e) < min_exposed:
        return {**row, "p": 1.0, "effect": 0.0, "testable": False}
    rate_e = row["exposed_anomalous"] / len(e)
    rate_u = row["unexposed_anomalous"] / len(u) if u else None
    p = binomtest(row["exposed_anomalous"], len(e), p0, alternative="greater").pvalue
    worse = len(u) < 3 or rate_e > rate_u  # attribution guard once there is a comparison group
    return {**row, "p": float(p) if worse else 1.0, "effect": rate_e - (rate_u if rate_u is not None else p0),
            "rate_exposed": rate_e, "rate_unexposed": rate_u, "testable": True}


def check_hypotheses(scores: dict[str, dict], factors: dict[str, dict], hyps: list[dict], B: int = 1999,
                     seed: int = 0, alpha: float = 0.05, min_arm: int = 5, min_n: int = 12,
                     method: str = "history", p0: float = P0) -> list[dict]:
    """One-sided test per testable hypothesis, Holm-corrected over the family."""
    pids = [p for p in scores if p in factors]
    meta = [factors[p] for p in pids]
    rng = np.random.default_rng(seed)
    out = []
    for h in hyps:
        spec = hypothesis_test(h)
        if not spec:
            continue
        f, s, sign = spec
        if method == "history" and s == "over_frac":
            out.append({"hypothesis": h["id"], "factor": f, "score": s, "sign": sign, "n": len(pids), "method": "history",
                        **_history_test(scores, factors, f, p0)})
            continue
        if len(pids) < min_n:
            continue
        x = np.array([m[f] for m in meta], float)
        arms = (int((x > 0).sum()), int((x == 0).sum())) if f in BINARY else (len(x), len(x))
        row = {"hypothesis": h["id"], "factor": f, "score": s, "sign": sign, "n": len(pids), "arms": arms}
        if np.ptp(x) == 0:  # e.g. every switch was cleaned: nothing to compare
            out.append({**row, "p": 1.0, "effect": 0.0, "testable": False})
            continue
        y = np.array([scores[p][s] for p in pids], float)
        p2, eff = freedman_lane(y, x, _covariates(meta), B, rng)
        p1 = p2 / 2 if eff * sign > 0 else 1 - p2 / 2
        out.append({**row, "p": p1, "effect": eff if f in BINARY else eff * float(x.std()), "testable": True})
    m, run_max = len(out), 0.0
    for i, t in enumerate(sorted(out, key=lambda t: t["p"])):
        run_max = max(run_max, min(1.0, t["p"] * (m - i)))
        t["p_holm"] = run_max
        need = 0 if t.get("method") == "history" else (min_arm if t["factor"] in BINARY else min_n)
        t["flag"] = t["testable"] and t["p_holm"] < alpha and min(t["arms"]) >= need
    return sorted(out, key=lambda t: t["p"])
