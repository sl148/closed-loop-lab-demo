"""Constraint sets for every combination of the lab's three real limits (live mode, M11).

    PYTHONPATH=src python scripts/make_combos.py      # writes configs/combos/combo_<E><D><C>.yaml

E = read equilibration, D = drug (reagent) stability, C = cell-line changeover; 1 = the limit is known
(gold value), 0 = missed. 000 is the blind run and 111 the gold run, so only the other six are written.
The web page's live mode plays the recorded run whose combination matches what the live agent found.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

ORDER = ["read_equilibration", "reagent_stability", "changeover"]


def main() -> None:
    gold = yaml.safe_load(Path("configs/constraints_gold.yaml").read_text())
    by_id = {c["id"]: c for c in gold["constraints"]}
    out = Path("configs/combos")
    out.mkdir(parents=True, exist_ok=True)
    for bits in product("01", repeat=3):
        code = "".join(bits)
        if code in ("000", "111"):
            continue
        keep = [by_id[k] for k, b in zip(ORDER, bits) if b == "1"]
        (out / f"combo_{code}.yaml").write_text(yaml.safe_dump({"task": gold["task"], "constraints": keep}, sort_keys=False))
        print(code, [c["id"] for c in keep])


if __name__ == "__main__":
    main()
