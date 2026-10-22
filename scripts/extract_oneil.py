"""Extract the O'Neil 2016 screen from the DrugComb v1.4 dose-level dump.

Source: Zenodo record 18449193, drugcomb_data_v1.4.csv (2 GB). O'Neil is the first ~280 MB,
so only that byte range is needed:
    curl -r 0-285000000 -o drugcomb_head.csv \
      https://zenodo.org/api/records/18449193/files/drugcomb_data_v1.4.csv/content

Writes data/oneil_blocks.csv.gz: replicate-averaged viability per
(drug_row, drug_col, cell_line, conc_r, conc_c), concentrations in uM.

    uv run python scripts/extract_oneil.py <path to drugcomb_head.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COLS = ["block_id", "conc_r", "conc_c", "inhibition", "drug_row", "drug_col", "conc_r_unit", "conc_c_unit",
        "cell_line_name", "study_name"]


def main(src: str) -> None:
    parts = []
    reader = pd.read_csv(src, usecols=COLS, chunksize=500_000, on_bad_lines="skip", engine="c")
    for chunk in reader:
        sub = chunk[chunk["study_name"] == "ONEIL"]
        parts.append(sub)
        if len(sub) == 0 and len(parts) > 2:
            break  # O'Neil is contiguous at the start of the file
    df = pd.concat(parts)
    assert set(df["conc_r_unit"].dropna()) <= {"uM"} and set(df["conc_c_unit"].dropna()) <= {"uM"}
    df["viability"] = 1.0 - df["inhibition"] / 100.0
    agg = (df.groupby(["drug_row", "drug_col", "cell_line_name", "conc_r", "conc_c"])
             .agg(viability=("viability", "mean"), sd=("viability", "std"), n=("viability", "size"))
             .reset_index())
    out = Path(__file__).parents[1] / "data" / "oneil_blocks.csv.gz"
    agg.to_csv(out, index=False, float_format="%.5g")
    print(f"rows={len(df)} blocks={df['block_id'].nunique()} triplets="
          f"{agg.groupby(['drug_row', 'drug_col', 'cell_line_name']).ngroups} drugs="
          f"{len(set(agg.drug_row) | set(agg.drug_col))} cell_lines={agg.cell_line_name.nunique()} -> {out}")
    print("replicate sd (median):", round(float(agg["sd"].median()), 4))


if __name__ == "__main__":
    main(sys.argv[1])
