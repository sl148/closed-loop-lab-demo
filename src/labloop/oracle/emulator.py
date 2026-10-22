"""Data-driven oracle from the O'Neil et al. 2016 combination screen (via DrugComb).

- Single agents: Hill curve per (drug, cell line), pooled over the single-agent rows of
  every block that drug appears in.
- Pairs: MuSyC fit per (pair, cell line) (synergy package); the pair term is MuSyC's own
  Bliss excess E(d1,0)E(0,d2)/E0^2 - E(d1,d2)/E0, which is exactly zero on the single-agent
  axes and smooth in log dose, so it extrapolates from O'Neil's 4x4 grid to our 7-step ladders.
  It is tabulated on a log-dose grid once and interpolated afterwards.
- 3-4 drug combinations: Bliss product of the singles minus the sum of the pair terms (no
  higher-order synergy beyond what is planted explicitly in the config). This is an
  assumption of the emulator, not a property of the data.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

from labloop.oracle.surface import CombinationSurface, HillDrug, Oracle, SynergyPocket

DATA = Path(__file__).parents[3] / "data" / "oneil_blocks.csv.gz"
GRID_N = 48


@lru_cache(maxsize=1)
def load_blocks(path: str = str(DATA)) -> pd.DataFrame:
    return pd.read_csv(path)


def _pair_rows(df: pd.DataFrame, a: str, b: str, line: str) -> tuple[pd.DataFrame, bool]:
    """Rows for (a, b) in `line`, oriented so conc_r is drug a. Returns (rows, found)."""
    sub = df[(df.drug_row == a) & (df.drug_col == b) & (df.cell_line_name == line)]
    if len(sub):
        return sub, True
    sub = df[(df.drug_row == b) & (df.drug_col == a) & (df.cell_line_name == line)]
    if len(sub):
        sub = sub.rename(columns={"conc_r": "conc_c", "conc_c": "conc_r"})
        return sub, True
    return sub, False


def _hill_v(d, ec50, h, emax):
    d = np.asarray(d, float)
    return 1 - emax * d**h / (d**h + ec50**h)


def fit_single(df: pd.DataFrame, drug: str, line: str) -> HillDrug:
    rows = pd.concat([
        df[(df.drug_row == drug) & (df.cell_line_name == line) & (df.conc_c == 0) & (df.conc_r > 0)]
        .assign(d=lambda x: x.conc_r),
        df[(df.drug_col == drug) & (df.cell_line_name == line) & (df.conc_r == 0) & (df.conc_c > 0)]
        .assign(d=lambda x: x.conc_c),
    ])
    d, v = rows["d"].to_numpy(float), rows["viability"].to_numpy(float)
    dmax = d.max()
    try:
        (ec50, h, emax), _ = curve_fit(_hill_v, d, v, p0=(np.median(d), 1.0, max(0.05, 1 - v.min())),
                                       bounds=([dmax * 1e-3, 0.3, 0.0], [dmax * 30, 5.0, 1.0]), maxfev=20000)
    except RuntimeError:
        ec50, h, emax = dmax, 1.0, float(np.clip(1 - v.min(), 0, 1))
    return HillDrug(drug, float(ec50), float(h), float(emax))


def max_conc(df: pd.DataFrame, drug: str) -> float:
    return float(max(df.loc[df.drug_row == drug, "conc_r"].max(skipna=True) if (df.drug_row == drug).any() else 0,
                     df.loc[df.drug_col == drug, "conc_c"].max(skipna=True) if (df.drug_col == drug).any() else 0))


class PairTerm:
    """MuSyC-internal Bliss excess, tabulated in log dose."""

    def __init__(self, model, dmax_a: float, dmax_b: float, dmin_a: float, dmin_b: float):
        la = np.linspace(np.log10(dmax_a) - 4, np.log10(dmax_a) + 0.5, GRID_N)
        lb = np.linspace(np.log10(dmax_b) - 4, np.log10(dmax_b) + 0.5, GRID_N)
        A, B = np.meshgrid(10**la, 10**lb, indexing="ij")
        e0 = float(model.E0)
        e_ab = np.asarray(model.E(A.ravel(), B.ravel()), float).reshape(A.shape) / e0
        e_a = np.asarray(model.E(10**la, np.zeros_like(la)), float) / e0
        e_b = np.asarray(model.E(np.zeros_like(lb), 10**lb), float) / e0
        table = e_a[:, None] * e_b[None, :] - e_ab
        self.interp = RegularGridInterpolator((la, lb), np.nan_to_num(table), bounds_error=False, fill_value=None)
        self.lo = (la[0], lb[0])
        self.hi = (la[-1], lb[-1])
        # Taper anchor: lowest screened dose. MuSyC's potency terms can extrapolate strong
        # synergy far below the data; the oracle only claims what the screen supports.
        self.measured_lo = (np.log10(dmin_a), np.log10(dmin_b))

    def __call__(self, da, db):
        da, db = np.asarray(da, float), np.asarray(db, float)
        pos = (da > 0) & (db > 0)
        la = np.clip(np.log10(np.where(pos, da, 1.0)), self.lo[0], self.hi[0])
        lb = np.clip(np.log10(np.where(pos, db, 1.0)), self.lo[1], self.hi[1])
        vals = self.interp(np.stack([la, lb], axis=-1)).reshape(np.shape(la))
        # Below the fitted range synergy must vanish with the dose: fade over one decade.
        raw_a = np.log10(np.where(pos, da, 1.0))
        raw_b = np.log10(np.where(pos, db, 1.0))
        taper = (np.clip(1 - (self.measured_lo[0] - raw_a), 0, 1)
                 * np.clip(1 - (self.measured_lo[1] - raw_b), 0, 1))
        return np.where(pos, vals * taper, 0.0)


def fit_pair(df: pd.DataFrame, a: str, b: str, line: str):
    from synergy.combination import MuSyC

    rows, found = _pair_rows(df, a, b, line)
    if not found:
        raise KeyError(f"no O'Neil block for {a} x {b} in {line}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MuSyC(E0_bounds=(0.8, 1.2), E1_bounds=(0, 1.2), E2_bounds=(0, 1.2), E3_bounds=(0, 1.2))
        try:
            m.fit(rows.conc_r.to_numpy(float), rows.conc_c.to_numpy(float), rows.viability.to_numpy(float),
                  max_nfev=50000)
        except (RuntimeError, ValueError, TypeError):
            return None
    return m


class InterpPairTerm:
    """Fallback when MuSyC does not converge: the block's measured Bliss excess, smoothed
    with 2x2 window means and bilinearly interpolated in log dose (zero on the axes)."""

    def __init__(self, rows: pd.DataFrame):
        piv = rows.pivot_table(index="conc_r", columns="conc_c", values="viability")
        va, vb = piv.loc[:, 0].drop(0), piv.loc[0].drop(0)
        exc = np.outer(va.values, vb.values) - piv.drop(index=0, columns=0).values
        pad = np.pad(exc, 1, mode="edge")
        smooth = np.mean([pad[i:i + exc.shape[0], j:j + exc.shape[1]] for i in range(3) for j in range(3)], axis=0)
        la, lb = np.log10(va.index.to_numpy(float)), np.log10(vb.index.to_numpy(float))
        self.interp = RegularGridInterpolator((la, lb), smooth, bounds_error=False, fill_value=None)
        self.lo, self.hi = (la[0], lb[0]), (la[-1], lb[-1])
        self.measured_lo = self.lo

    __call__ = PairTerm.__call__


def build_oneil_oracle(cfg: dict) -> Oracle:
    df = load_blocks()
    drugs, lines = cfg["drugs"], cfg["cell_lines"]
    surfaces = {}
    for line in lines:
        singles = tuple(fit_single(df, d, line) for d in drugs)
        terms = {}
        for i, j in combinations(range(len(drugs)), 2):
            m = fit_pair(df, drugs[i], drugs[j], line)
            rows = _pair_rows(df, drugs[i], drugs[j], line)[0]
            if m is not None:
                terms[(i, j)] = PairTerm(m, max_conc(df, drugs[i]), max_conc(df, drugs[j]),
                                         rows.loc[rows.conc_r > 0, "conc_r"].min(), rows.loc[rows.conc_c > 0, "conc_c"].min())
            else:
                terms[(i, j)] = InterpPairTerm(rows)
        pockets = tuple(SynergyPocket(tuple(p["drugs"]), tuple(p["center_log10"]), p["width_log10"], p["strength"])
                        for p in cfg.get("pockets", {}).get(line, []))
        surfaces[line] = CombinationSurface(singles, pockets, terms)
    return Oracle(surfaces, cfg.get("t_ref_h", 48.0), cfg.get("time_exponent", 0.5))


def measured_excess(df: pd.DataFrame) -> pd.DataFrame:
    """Per (pair, cell line): max observed Bliss excess on the 4x4 grid (from the block's own
    single-agent rows), used to pick panels with a real synergy pocket."""
    out = []
    for (a, b, line), g in df.groupby(["drug_row", "drug_col", "cell_line_name"]):
        piv = g.pivot_table(index="conc_r", columns="conc_c", values="viability")
        if 0 not in piv.index or 0 not in piv.columns:
            continue
        va, vb = piv.loc[:, 0].drop(0), piv.loc[0].drop(0)
        combo = piv.drop(index=0, columns=0)
        exc = np.outer(va.values, vb.values) - combo.values
        # A real pocket is supported by its neighbors: peak of 2x2-window means.
        win = [np.nanmean(exc[i:i + 2, j:j + 2]) for i in range(exc.shape[0] - 1) for j in range(exc.shape[1] - 1)]
        out.append({"a": a, "b": b, "line": line, "max_excess": float(np.nanmax(exc)),
                    "peak_smooth": float(np.nanmax(win)) if win else np.nan,
                    "mean_excess": float(np.nanmean(exc))})
    return pd.DataFrame(out)
