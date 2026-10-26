"""Liquid supplies on the liquid-handler deck: drug dilution plates and cell reservoirs.

Nothing refills by itself. A prep plan (agent/prep.py, or `PrepPlan.default()`) says how the lab
makes its liquids from what it has on hand:
- stocks: each drug's powder dissolved in DMSO, once, at the start;
- batches: the LH makes one dilution plate per drug (the dose ladder, 3-fold) from the stock, in
  DMSO or in medium, `ul_per_well` per well; a batch is remade when a plate would run it short,
  or when it would be older than `max_age_min` at the plate's dispense;
- cells: one reservoir per cell line, `cell_ml` per fill, made at the cell-prep station; the arm
  swaps the reservoir on the deck when the next plate is another cell line or this one runs low.

`Supplies` keeps that state per liquid handler and the lab's inventory ledger. The lab
(execution/lab.py) asks it before a plate enters the deck and orders the jobs it needs.
Numbers per drug come from the Cayman product sheets in data/kb (formula weight, DMSO solubility).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Cayman product information (data/kb/manifest.yaml): formula weight (g/mol), solubility in DMSO (mg/ml)
DRUGS = {"MK-4827": (492.6, 30.0), "SORAFENIB": (464.8, 20.0), "ERLOTINIB": (393.4, 25.0), "VINBLASTINE": (909.1, 10.0)}
CELL_UL, DRUG_UL = 16.0, 2.0  # per well (protocol/planner.py, 384-well)
FINAL_UL = CELL_UL + 4 * DRUG_UL  # nominal well volume after cells and four drug additions
WORK_FACTOR = FINAL_UL / DRUG_UL  # a working dilution is the final dose times this
DEAD_UL = 20.0  # left in a dilution-plate well that the tips cannot reach
CELL_DEAD_ML = 1.0  # left in a cell trough that the tips cannot reach
PREP_MIN_PER_DRUG = 6  # LH: stock to top working dilution, 3-fold series down the column, mixing
TIPS_PER_DRUG = 8  # one fresh tip per transfer in the series


@dataclass
class PrepPlan:
    stocks: dict[str, dict] = field(default_factory=dict)  # drug -> {"mg", "dmso_ul"}
    series_solvent: dict[str, str] = field(default_factory=dict)  # drug -> "dmso" | "medium"
    ul_per_well: float = 300.0
    max_age_min: int | None = None  # None: remade only when it runs out
    cell_ml: float = 30.0
    source: str = "default"
    # How each batch is pipetted (execution/prep_protocol.py): intermediate dilution factors per drug
    # before the top column, and tip + mode per transfer role (fill, stock, intermediate, top, series).
    dilution: dict[str, list[float]] = field(default_factory=dict)
    pipetting: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def default(cls, drugs: list[str]) -> "PrepPlan":
        """What the lab does unasked: 10 mM stocks from 1 mg, big batches, remade only when empty."""
        return cls({d: {"mg": 1.0, "dmso_ul": round(1.0 / DRUGS[d][0] * 1e6 / 10.0)} for d in drugs},
                   {d: "medium" for d in drugs}, 300.0, None, 30.0, "default")

    @classmethod
    def from_dict(cls, d: dict) -> "PrepPlan":
        return cls(d.get("stocks", {}), d.get("series_solvent", {}), float(d.get("ul_per_well", 300)),
                   d.get("max_age_min"), float(d.get("cell_ml", 30)), d.get("source", "agent"),
                   {k: [float(x) for x in v] for k, v in (d.get("dilution") or {}).items()}, d.get("pipetting") or {})

    def to_dict(self) -> dict:
        return {"stocks": self.stocks, "series_solvent": self.series_solvent, "ul_per_well": self.ul_per_well,
                "max_age_min": self.max_age_min, "cell_ml": self.cell_ml, "source": self.source,
                "dilution": self.dilution, "pipetting": self.pipetting}

    # ------------------------------------------------------------------ arithmetic
    def stock_mM(self, drug: str) -> float:
        s = self.stocks[drug]
        return s["mg"] / DRUGS[drug][0] * 1e6 / s["dmso_ul"]

    def stock_mg_per_ml(self, drug: str) -> float:
        s = self.stocks[drug]
        return s["mg"] / s["dmso_ul"] * 1e3

    def top_working_uM(self, top_dose_uM: float) -> float:
        return top_dose_uM * WORK_FACTOR

    def dmso_final_pct(self, tops: dict[str, float]) -> float:
        """Worst well: every drug at its top dose. A dilution series in medium keeps the top's DMSO
        share at the top and less below; one in DMSO carries it at every dose."""
        return sum(self.top_working_uM(t) / (self.stock_mM(d) * 1e3) * 100 / WORK_FACTOR for d, t in tops.items())

    def stock_ul_per_batch(self, drug: str, top_dose_uM: float, n_doses: int = 7) -> float:
        """Stock drawn for one batch: every transfer out of the stock in the dilution protocol (all 8 rows)."""
        from labloop.execution.prep_protocol import build

        return sum(t.ul * t.channels * t.repeats for t in build(self, [drug], {drug: top_dose_uM}, n_doses) if t.src == "stock")

    def dmso_pct(self, drugs: list[str], tops: dict[str, float]) -> dict:
        """Per drug: the DMSO share it adds to a well at its top dose (%), and whether lower doses carry
        the same (series in DMSO) or less (series in medium). The reader turns this into each well's DMSO."""
        return {"top_pct": [self.top_working_uM(tops[d]) / (self.stock_mM(d) * 1e3) * 100 / WORK_FACTOR for d in drugs],
                "flat": [self.series_solvent.get(d, "medium") == "dmso" for d in drugs],
                "tops": [tops[d] for d in drugs]}

    def protocol(self, drugs: list[str], tops: dict[str, float], n_doses: int):
        from labloop.execution.prep_protocol import build

        return build(self, drugs, tops, n_doses)

    def prep_minutes(self, n_drugs: int, tops: dict[str, float] | None = None, n_doses: int = 7) -> int:
        if tops:
            from labloop.execution.prep_protocol import minutes

            return minutes(self.protocol(list(tops), tops, n_doses))
        return PREP_MIN_PER_DRUG * n_drugs


@dataclass
class Batch:
    id: str
    made: int
    left: dict[tuple[int, float], float]  # (drug index, dose) -> ul per well left


class Supplies:
    def __init__(self, plan: PrepPlan, drugs: list[str], ladders: list[list[float]], inventory: dict):
        self.plan, self.drugs, self.ladders = plan, drugs, [[x for x in l if x > 0] for l in ladders]
        self.inv = {k: dict(v) if isinstance(v, dict) else v for k, v in inventory.items()}
        self.stock_ul = {d: 0.0 for d in drugs}
        self.batch: dict[str, Batch | None] = {}
        self.cells: dict[str, dict] = {}  # lh -> {"line", "left_ml"}
        self.cell_left: dict[str, float] = {}  # cell line -> ml left in its reservoir (off deck too)
        self.ledger: list[dict] = []
        self.n_batches = 0

    # ------------------------------------------------------------------ what a plate takes
    def need(self, pm) -> dict:
        # Channel k always draws from row k of the source column and serves its own plate rows (k, or 2k and
        # 2k + 1 on a 384-well plate): a source well must cover its busiest channel, not the average.
        per: dict[tuple[int, float], dict[int, int]] = {}
        for (r, _c), content in pm.wells.items():
            if isinstance(content, str):
                continue
            ch = r // 2 if pm.spec.rows > 8 else r
            for i, d in enumerate(content.doses):
                if d > 0:
                    per.setdefault((i, d), {})[ch] = per.setdefault((i, d), {}).get(ch, 0) + 1
        # each of 8 channels draws from its own row of the source column
        wells = sum(1 for w in pm.wells.values() if w != "EMPTY")  # controls get cells too
        return {"drug": {k: max(v.values()) * DRUG_UL for k, v in per.items()},
                "cells_ml": wells * CELL_UL / 1000, "line": pm.cell_line}

    def drug_ok(self, lh: str, pm, at: int) -> bool:
        b = self.batch.get(lh)
        if b is None:
            return False
        if self.plan.max_age_min is not None and at - b.made > self.plan.max_age_min:
            return False
        return all(b.left.get(k, 0) - DEAD_UL >= v for k, v in self.need(pm)["drug"].items())

    def cells_ok(self, lh: str, pm) -> bool:
        c = self.cells.get(lh)
        return bool(c) and c["line"] == pm.cell_line and c["left_ml"] - CELL_DEAD_ML >= self.need(pm)["cells_ml"]

    def consume(self, lh: str, pm) -> dict:
        n = self.need(pm)
        b = self.batch.get(lh)
        if b:
            for k, v in n["drug"].items():
                b.left[k] = b.left.get(k, 0) - v
        c = self.cells.get(lh)
        if c and c["line"] == pm.cell_line:
            c["left_ml"] -= n["cells_ml"]
            self.cell_left[c["line"]] = c["left_ml"]
        return n

    # ------------------------------------------------------------------ making things
    def _take(self, item: str, n: float, t: int, ref: str) -> None:
        self.inv[item] = self.inv.get(item, 0) - n
        self.ledger.append({"t": t, "item": item, "delta": -n, "ref": ref, "left": self.inv[item]})

    def make_stocks(self, t: int) -> None:
        for d in self.drugs:
            s = self.plan.stocks[d]
            self.inv.setdefault("mg", {})[d] = self.inv.get("mg", {}).get(d, 0) - s["mg"]
            self.ledger.append({"t": t, "item": f"{d} powder (mg)", "delta": -s["mg"], "ref": "stocks", "left": self.inv["mg"][d]})
            self._take("DMSO (ul)", s["dmso_ul"], t, "stocks")
            self.stock_ul[d] += s["dmso_ul"]

    def batch_ul(self, pm) -> float:
        """Per-dose volume a batch needs so this plate can draw from it: the plan's volume, or more when the
        plate's busiest channel takes more of one dose than that leaves above the dead volume."""
        return max([self.plan.ul_per_well] + [v + DEAD_UL for v in self.need(pm)["drug"].values()])

    def make_batch(self, lh: str, t: int, tops: dict[str, float], job: str, ul: float | None = None) -> Batch:
        """`ul`: a larger per-dose volume than the plan's, for a plate the plan's batch cannot serve
        (the dilution series is scaled up; stock and solvent scale with it)."""
        if not any(self.stock_ul.values()):
            self.make_stocks(t)
        self.n_batches += 1
        ul = max(ul or 0.0, self.plan.ul_per_well)
        k = ul / self.plan.ul_per_well
        for i, d in enumerate(self.drugs):
            draw = self.plan.stock_ul_per_batch(d, tops[d]) * k
            self.stock_ul[d] -= draw
            if self.stock_ul[d] < 0:  # out of stock: dissolve more powder if there is any
                self.ledger.append({"t": t, "item": f"{d} stock (ul)", "delta": 0, "ref": "stock_out", "left": self.stock_ul[d]})
                self.make_stock_one(d, t)
            solvent = "DMSO (ul)" if self.plan.series_solvent.get(d, "medium") == "dmso" else "medium (ul)"
            self._take(solvent, ul * len(self.ladders[i]), t, job)
            self._take("SRC-DIL plates", 1, t, job)
        b = Batch(job, t, {(i, x): ul for i, lad in enumerate(self.ladders) for x in lad})
        self.batch[lh] = b
        return b

    def make_stock_one(self, d: str, t: int) -> None:
        s = self.plan.stocks[d]
        self.inv.setdefault("mg", {})[d] = self.inv.get("mg", {}).get(d, 0) - s["mg"]
        self.ledger.append({"t": t, "item": f"{d} powder (mg)", "delta": -s["mg"], "ref": "restock", "left": self.inv["mg"][d]})
        self._take("DMSO (ul)", s["dmso_ul"], t, "restock")
        self.stock_ul[d] += s["dmso_ul"]

    def fill_cells(self, lh: str, line: str, t: int, job: str) -> None:
        """The arm seats `line`'s reservoir: the one already made if it has enough, else a fresh fill."""
        left = self.cell_left.get(line, 0.0)
        if left - CELL_DEAD_ML < 384 * CELL_UL / 1000:  # cannot serve another plate: a fresh reservoir
            self._take("RES-CELL reservoirs", 1, t, job)
            self._take("medium (ul)", self.plan.cell_ml * 1000, t, job)
            left = self.plan.cell_ml
            self.cell_left[line] = left
        self.cells[lh] = {"line": line, "left_ml": left}
