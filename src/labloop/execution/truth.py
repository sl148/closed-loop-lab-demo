"""Physical limits of the lab that the scheduler is not told about (M9).

A real lab has limits that live in manuals, datasheets and protocols rather than in the
scheduler: how long seeded cells may sit outside the incubator, how long drug dilutions keep
their potency at room temperature, that the biology keeps going while a plate waits for the
reader, and that switching cell lines on a liquid handler without cleaning carries cells over.
The emulator applies them whether or not the plan respects them. The constraint agent's job is
to find them in the document archive so the scheduler can.

Effects are what a real lab would see: drug potency loss, a late read and carry-over bias the
data silently; stressed cells add well-to-well variation, which a plate QC (Z' factor on the
controls) can catch.

Labware has the same kind of limits: a clear-walled plate lets luminescence leak into its
neighbours, a black one loses signal, and a tip used below its validated volume dispenses
less, and less evenly, than asked. They apply to the deck the run uses (protocol/deck.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Truth:
    # Seeded cells on the deck at room temperature: limit from dispense start to incubator entry,
    # and the extra relative well-to-well variation per minute over it.
    out_of_incubator_min: int | None = None
    out_of_incubator_noise_per_min: float = 0.003
    # Drug dilutions are prepared when the plate is released; past the stable period at room
    # temperature their potency halves every `drug_halflife_min`.
    drug_rt_stable_min: int | None = None
    drug_halflife_min: float = 720.0
    # A luminescent viability read needs the plate equilibrated to room temperature first: from
    # leaving the incubator to the read; shorter waits add temperature-gradient variation.
    read_equilibration_min: int | None = None
    read_equilibration_noise: float = 0.15  # extra relative variation at zero equilibration
    # Reads respond to the actual incubation time, not the nominal timepoint.
    actual_timepoint: bool = False
    # A cell-line switch on a liquid handler with no cleaning in between contaminates wells.
    changeover_wells: int = 0
    # Plate QC: a plate whose controls give Z' below this is rejected and re-run.
    qc_z_min: float | None = None
    # Labware. Per assay plate wall: "crosstalk" = share of the neighbours' mean signal a well
    # picks up; "extra_noise" = relative variation from lost signal.
    plates: dict[str, dict[str, float]] = field(default_factory=dict)
    # A tip used below its validated volume (deck.TIPS min_ul): drug doses come out scaled by
    # 1 + under_range_bias with relative spread under_range_cv; for cell seeding the spread
    # shows up as extra well-to-well variation.
    under_range_cv: float = 0.0
    under_range_bias: float = 0.0
    # A reused tip that touches liquid already in the well (surface dispensing into a wet well) carries
    # that well's contents to the next ones: this many wells per plate get the wrong mix.
    carryover_wells: int = 0
    # Liquid prep (execution/supplies.py). A stock made above the drug's DMSO solubility does not fully
    # dissolve: every dose of it comes out scaled by solubility / stock. Cells lose `dmso_kill_per_pct`
    # of their viability per % of DMSO in the well above `dmso_max_pct`.
    stock_solubility: bool = False
    dmso_max_pct: float | None = None
    dmso_kill_per_pct: float = 0.25
    sources: dict[str, str] = field(default_factory=dict)  # limit -> archive passage

    @property
    def on(self) -> bool:
        return any((self.out_of_incubator_min is not None, self.drug_rt_stable_min is not None,
                    self.read_equilibration_min is not None, self.actual_timepoint, self.changeover_wells > 0,
                    self.qc_z_min is not None, self.labware_on, self.stock_solubility, self.dmso_max_pct is not None))

    def prep_effects(self, plan, drugs: list[str], tops: dict[str, float]) -> dict:
        """Read-time effects of the prep plan, the same for every plate of the run."""
        if plan is None or not (self.stock_solubility or self.dmso_max_pct is not None):
            return {}
        from labloop.execution.supplies import DRUGS

        out: dict = {}
        scale = [1.0] * len(drugs)
        if self.stock_solubility:
            scale = [min(1.0, DRUGS[d][1] / plan.stock_mg_per_ml(d)) for d in drugs]
        if self.under_range_bias:  # a dilution step outside its liquid class: that drug's doses all come out off
            from labloop.execution.prep_protocol import out_of_range

            bad = out_of_range(plan.protocol(drugs, tops, 7))
            scale = [s * (1 + self.under_range_bias) if d in bad else s for s, d in zip(scale, drugs)]
            if bad:
                out["prep_under_range"] = sorted(bad)
        if any(s < 1 for s in scale):
            out["drug_scale"] = scale
        if self.dmso_max_pct is not None:
            dm = plan.dmso_pct(drugs, tops)
            if sum(dm["top_pct"]) > self.dmso_max_pct:  # some well can go over: the reader works out which
                out["dmso"] = {**dm, "max": self.dmso_max_pct, "kill_per_pct": self.dmso_kill_per_pct}
        return out

    @property
    def labware_on(self) -> bool:
        return bool(self.plates) or self.under_range_cv > 0 or self.under_range_bias != 0 or self.carryover_wells > 0

    def labware_effects(self, deck: dict | None, n_wells: int) -> dict:
        """Read-time effects of the deck's labware for a plate of `n_wells` wells."""
        if not deck or not self.labware_on:
            return {}
        from labloop.protocol.deck import TIPS
        from labloop.protocol.planner import CELLS_UL, DRUG_UL

        size, out = (384 if n_wells > 96 else 96), {}
        fx = self.plates.get(deck.get("assay_plate", "clear"), {})
        if fx.get("crosstalk"):
            out["crosstalk"] = fx["crosstalk"]
        if fx.get("extra_noise"):
            out["extra_noise"] = fx["extra_noise"]
        from labloop.protocol.pipetting import in_range

        d = {"drug_tip": "50uL", "cell_tip": "1000uL", "drug_mode": "jet", "cell_mode": "jet", **deck}
        if not in_range(DRUG_UL[size], d["drug_tip"], "medium", d["drug_mode"]):  # Hamilton liquid class range
            out["tip_under_range"] = {"drug": f"{d['drug_tip']} {d['drug_mode']}"}
            out["dose_cv"], out["dose_bias"] = self.under_range_cv, self.under_range_bias
        if not in_range(CELLS_UL[size], d["cell_tip"], "medium", d["cell_mode"]):
            out.setdefault("tip_under_range", {})["cells"] = f"{d['cell_tip']} {d['cell_mode']}"
            out["extra_noise"] = out.get("extra_noise", 0.0) + self.under_range_cv
        return out

    def carryover(self, deck: dict | None) -> str | None:
        """Which reagent's reused tips touch liquid already in the wells, if any."""
        if not deck or not self.carryover_wells:
            return None
        reuse = lambda k: deck.get(f"{k}_reuse") if deck.get(f"{k}_reuse") is not None else deck.get("tip_reuse", True)
        second = "cells" if deck.get("drugs_first") else "drug"  # added on top of the first reagent
        mode = deck.get(f"{'cell' if second == 'cells' else 'drug'}_mode", "jet")
        return second if mode == "surface" and reuse("cell" if second == "cells" else "drug") else None

    def plate_effects(self, t_release: int, t_dispense: int, t_incubate: int, incubated_min: int,
                      t_out: int | None = None, t_read: int | None = None) -> dict:
        """Read-time effects for one plate from its executed times (minutes); t_out = leaves the incubator."""
        out: dict = {}
        if self.read_equilibration_min is not None and t_out is not None and t_read is not None:
            short = self.read_equilibration_min - (t_read - t_out)
            if short > 0:
                out["read_equilibration_short_min"] = short
                out["extra_noise"] = out.get("extra_noise", 0.0) + self.read_equilibration_noise * short / self.read_equilibration_min
        if self.out_of_incubator_min is not None:
            over = t_incubate - t_dispense - self.out_of_incubator_min
            if over > 0:
                out["out_of_incubator_over_min"] = over
                out["extra_noise"] = out.get("extra_noise", 0.0) + self.out_of_incubator_noise_per_min * over
        if self.drug_rt_stable_min is not None:
            over = t_dispense - t_release - self.drug_rt_stable_min
            if over > 0:
                out["drug_rt_over_min"] = over
                out["dose_scale"] = 0.5 ** (over / self.drug_halflife_min)
        if self.actual_timepoint:
            out["timepoint_h"] = incubated_min / 60
        return out


def z_factor(pos: list[float], neg: list[float]) -> float | None:
    """Z' = 1 - 3 (sd_pos + sd_neg) / |mean_neg - mean_pos| (Zhang et al. 1999)."""
    import numpy as np

    if len(pos) < 2 or len(neg) < 2:
        return None
    gap = abs(float(np.mean(neg)) - float(np.mean(pos)))
    if gap < 1e-9:
        return float("-inf")
    return 1 - 3 * (float(np.std(pos, ddof=1)) + float(np.std(neg, ddof=1))) / gap


def build_truth(cfg: dict | str | None) -> Truth:
    if not cfg:
        return Truth()
    if isinstance(cfg, str):
        cfg = yaml.safe_load(Path(cfg).read_text())
    return Truth(**cfg)
