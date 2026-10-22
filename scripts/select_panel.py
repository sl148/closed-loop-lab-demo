"""Pick O'Neil drug/cell-line panels for Levels 2 and 4: every drug pair must be screened in
every chosen cell line, and the panel should contain a real but localized synergy pocket
(high max Bliss excess, low mean excess). Prints the panels; configs pin them explicitly.

    uv run python scripts/select_panel.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from labloop.oracle.emulator import load_blocks, max_conc, measured_excess


def main() -> None:
    df = load_blocks()
    cache = Path(__file__).parents[1] / "data" / "oneil_excess.csv"
    if cache.exists():
        ex = pd.read_csv(cache)
    else:
        ex = measured_excess(df)
        ex.to_csv(cache, index=False, float_format="%.5g")
    ex["pair"] = [tuple(sorted(x)) for x in zip(ex.a, ex.b)]
    # Localized pocket score: a neighbor-supported peak, weak average.
    ex["score"] = ex.peak_smooth - 2 * ex.mean_excess.clip(lower=0)
    score = {(r.pair, r.line): r.score for r in ex.itertuples()}
    peak = {(r.pair, r.line): r.peak_smooth for r in ex.itertuples()}
    lines_of = ex.groupby("pair")["line"].apply(set).to_dict()
    seeds = [p for p in ex.sort_values("score", ascending=False).drop_duplicates("pair").pair.head(15)]
    drugs_all = sorted({d for p in lines_of for d in p})
    for n_drugs, n_lines in ((3, 3), (4, 4)):
        best = None
        for a, b in seeds:
            for extra in combinations([d for d in drugs_all if d not in (a, b)], n_drugs - 2):
                drugs = (a, b, *extra)
                pairs = [tuple(sorted(p)) for p in combinations(drugs, 2)]
                if any(p not in lines_of for p in pairs):
                    continue
                common = set.intersection(*(lines_of[p] for p in pairs))
                if len(common) < n_lines:
                    continue
                per_line = {l: max(score[(p, l)] for p in pairs) for l in common}
                lines = sorted(common, key=lambda l: -per_line[l])[:n_lines]
                total = sum(per_line[l] for l in lines)
                if best is None or total > best[0]:
                    best = (total, drugs, lines, pairs)
        total, drugs, lines, pairs = best
        print(f"{n_drugs} drugs x {n_lines} lines  score={total:.3f}")
        print("  drugs:", list(drugs), " max conc:", [max_conc(df, d) for d in drugs])
        print("  cell_lines:", list(lines))
        for l in lines:
            print("   ", l, {f"{p[0]}+{p[1]}": round(peak[(p, l)], 3) for p in pairs})


if __name__ == "__main__":
    main()
