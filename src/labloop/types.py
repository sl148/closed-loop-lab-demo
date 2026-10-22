"""Domain types shared by design, layout, execution, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

CONTROL_NEG = "NEG"  # vehicle only, viability ~1
CONTROL_POS = "POS"  # kill control, viability ~0
EMPTY = "EMPTY"


@dataclass(frozen=True, order=True)
class Condition:
    cell_line: str
    doses: tuple[float, ...]  # one per drug (uM), 0 = absent
    timepoint_h: int

    @property
    def active(self) -> tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.doses) if d > 0)

    @property
    def is_single(self) -> bool:
        return len(self.active) == 1

    @property
    def is_combo(self) -> bool:
        return len(self.active) >= 2

    def single(self, i: int) -> Condition:
        doses = tuple(d if k == i else 0.0 for k, d in enumerate(self.doses))
        return Condition(self.cell_line, doses, self.timepoint_h)

    def group(self) -> tuple[str, int]:
        return self.cell_line, self.timepoint_h

    def to_dict(self) -> dict:
        return {"cell_line": self.cell_line, "doses": list(self.doses), "timepoint_h": self.timepoint_h}

    @staticmethod
    def from_dict(d: dict) -> Condition:
        return Condition(d["cell_line"], tuple(d["doses"]), d["timepoint_h"])


@dataclass(frozen=True)
class PlateSpec:
    rows: int = 8
    cols: int = 12

    @property
    def n_wells(self) -> int:
        return self.rows * self.cols

    def well_name(self, r: int, c: int) -> str:
        return f"{chr(ord('A') + r)}{c + 1}"

    def is_edge(self, r: int, c: int) -> bool:
        return r in (0, self.rows - 1) or c in (0, self.cols - 1)

    def channel_group(self, r: int, c: int) -> tuple[int, int]:
        """Wells one 8-channel head pass can address together: a 96-well column, or every
        other row of a 384-well column."""
        stride = max(1, self.rows // 8)
        return c, r % stride


@dataclass
class PlateMap:
    plate_id: str
    spec: PlateSpec
    cell_line: str
    timepoint_h: int
    campaign: str = ""
    # (row, col) -> Condition or a control label
    wells: dict[tuple[int, int], Condition | str] = field(default_factory=dict)

    def conditions(self) -> list[Condition]:
        return [w for w in self.wells.values() if isinstance(w, Condition)]


@dataclass(frozen=True)
class WellRead:
    plate_id: str
    well: str
    content: Condition | str
    is_edge: bool
    value: float  # raw reader value
    viability: float  # normalized to plate controls
    t_read_min: int
    valid: bool = True
