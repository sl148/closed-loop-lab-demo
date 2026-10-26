"""The lab's records for one run, as linked tables (a small LIMS): what was made, from what, where, when.

Built from a run's replay bundle (scripts/export_web.py `replay` plus the M10 `loop` section), so the
web page and the workbook show the same rows.

Tables (every row has `t`, minutes from launch, when it came to exist, so a viewer can replay them):
- plates:    one row per assay plate; links to its liquid handler, incubator, reader, limits version
- wells:     one row per well with contents (doses per drug or control) and the reader's viability
- labware:   assay plates, dilution plates, the cell trough, tip racks; where each sits and when it is used
- inventory: a ledger of every consumption (tips, plates, cells, drug dilutions, detection reagent),
             linked to the plate or rack that used it; sum it up to any time for stock on hand
- events:    what the lab saw: failures and recoveries, camera flags, cleaning, QC fails, reviews,
             limit changes, the pre-launch deck check
- limits:    every constraint version the lab ran under, with its source
- truth:     sim only, never seen by the lab: wells corrupted by the hidden limits, per plate

Volumes come from the pipetting protocol (per pass: wells x uL per well). Detection reagent is added
at the read, equal to the medium volume (CellTiter-Glo TB288: reagent volume = medium volume).
"""

from __future__ import annotations

from collections import defaultdict

TABLES = ("plates", "wells", "labware", "inventory", "events", "limits", "truth")


def _well(i: int, cols: int) -> str:
    return f"{chr(65 + i // cols)}{i % cols + 1}"


def tables(b: dict, raw: dict | None = None) -> dict[str, list[dict]]:
    """`raw`: plate -> well -> the reader's raw signal (the bundle keeps only viability)."""
    kinds, drugs, loop = b["kinds"], [d.lower() for d in b.get("drugs", [])], b.get("loop", {})
    ops_by = defaultdict(list)
    for o in b["ops"]:
        ops_by[o[0]].append(o)
    read_at = {r["plate"]: r for r in b["reads"]}
    version = loop.get("plate_version", {})
    out: dict[str, list[dict]] = {k: [] for k in TABLES}
    ledger = out["inventory"]

    # plates, and what each consumed
    for pid in sorted(b["plates"], key=lambda p: (b["submits"].get(p, 0), p)):
        p, ops = b["plates"][pid], sorted(ops_by.get(pid, []), key=lambda o: o[3])
        of = lambda k: next((o for o in ops if kinds[o[1]] == k), None)
        d, inc, rd, load = of("dispense"), of("incubate"), of("read"), of("transfer")
        r = read_at.get(pid)
        out["plates"].append({
            "t": b["submits"].get(pid, ops[0][3] if ops else 0), "plate": pid, "campaign": p["campaign"], "cell_line": p["cell_line"],
            "timepoint_h": p["timepoint_h"], "format": f"{p['rows'] * p['cols']}-well",
            "liquid_handler": d[2] if d else None, "dispensed": d[3] if d else None, "incubator": inc[2] if inc else None,
            "incubated": [inc[3], inc[4]] if inc else None, "reader": rd[2] if rd else None, "read": r["t"] if r else None,
            "valid_wells": r["valid"] if r else None, "wells": r["n"] if r else None,
            "status": p.get("status"), "limits_version": version.get(pid, 0),
            "tips": (b.get("protocols", {}).get(pid) or {}).get("tips")})
        if load:
            ledger.append({"t": load[3], "item": "assay plate, 384-well", "delta": -1, "unit": "plate", "ref": pid})
        proto = b.get("protocols", {}).get(pid)
        if proto and d:
            vol = defaultdict(float)
            for ps in proto["passes"]:
                vol[ps[0]] += len(ps[3]) * ps[4]
            for what, v in sorted(vol.items()):
                item = f"cells, {p['cell_line']}" if what == 0 else f"{drugs[what - 1] if what - 1 < len(drugs) else what} dilution"
                ledger.append({"t": d[4], "item": item, "delta": -round(v, 1), "unit": "uL", "ref": pid})
            ledger.append({"t": d[4], "item": f"tips, {proto['passes'][0][10] if len(proto['passes'][0]) > 10 else '300uL'}",
                           "delta": -proto.get("tips", 0), "unit": "tip", "ref": pid})
            if rd:
                cells = vol.get(0, 0.0)
                ledger.append({"t": rd[3], "item": "detection reagent (CellTiter-Glo)", "delta": -round(cells, 1), "unit": "uL", "ref": pid})
        # wells: contents and the read
        reads = p.get("reads") or {}
        for w, c in p["wells"].items():
            row = {"t": r["t"] if r else None, "plate": pid, "well": w,
                   "content": "vehicle control" if c == "NEG" else "kill control" if c == "POS" else "drugs"}
            if c not in ("NEG", "POS"):
                for k, dose in enumerate(str(c).split("|")):
                    row[f"{drugs[k] if k < len(drugs) else k} uM"] = float(dose)
            if w in reads:
                if raw and w in raw.get(pid, {}):
                    row["raw signal"] = raw[pid][w]
                row["viability"], row["valid"] = reads[w][0], bool(reads[w][1])
            out["wells"].append(row)

    # labware
    deck = b.get("deck", {})
    for pid, p in b["plates"].items():
        ops = sorted(ops_by.get(pid, []), key=lambda o: o[3])
        if ops:
            out["labware"].append({"t": ops[0][3], "labware": pid, "type": "assay plate, 384-well", "contents": p["cell_line"],
                                   "where": f"{ops[0][5].split('>')[0] if ops[0][5] else 'hotel'}", "first_used": ops[0][3], "last_used": ops[-1][4]})
    for k, name in enumerate(drugs):
        out["labware"].append({"t": 0, "labware": f"DIL-{name.upper()[:4]}", "type": "dilution plate, 96-well", "contents": f"{name} dose ladder",
                               "where": f"sources carrier, site {k + 2} (rail {deck.get('source_rail')})", "first_used": None, "last_used": None})
    out["labware"].append({"t": 0, "labware": "TROUGH-CELLS", "type": "reservoir" if deck.get("cell_source") == "trough" else "reservoir plate",
                           "contents": "cell suspension", "where": f"sources carrier, site 1 (rail {deck.get('source_rail')})", "first_used": None, "last_used": None})
    racks = {}
    for e in b.get("tips", []):
        if e["kind"] == "tip_order":
            racks[e["job"]] = {"t": e["t"], "labware": e["job"], "type": f"tip rack, {deck.get('drug_tip', '300uL')}", "contents": "96 tips",
                               "where": f"rack hotel -> {e['lh']} tip carrier (rail {deck.get('tip_rail')})", "first_used": None, "last_used": None}
        elif e["kind"] == "tip_swap" and e["job"] in racks:
            racks[e["job"]]["first_used"] = e["t"]
            ledger.append({"t": e["t"], "item": f"tips, {deck.get('drug_tip', '300uL')}", "delta": 96, "unit": "tip", "ref": e["job"]})
    out["labware"] += list(racks.values())
    for lw in out["labware"]:
        used = [x["t"] for x in ledger if x["ref"] == lw["labware"]]
        if lw["first_used"] is None and lw["type"].startswith(("dilution", "reservoir")):
            drug = lw["contents"].split(" dose")[0]
            used = [x["t"] for x in ledger if (drug in x["item"] if "dose" in lw["contents"] else x["item"].startswith("cells"))]
            lw["first_used"], lw["last_used"] = (min(used), max(used)) if used else (None, None)

    # events the lab saw
    for f in b.get("failures", []):
        out["events"].append({"t": f["t"], "kind": "failure", "what": f["type"].replace("_", " "), "ref": f.get("plate") or f.get("resource"),
                              "detail": ", ".join(f.get("actions", []))})
    for c in b.get("camera", []):
        if c.get("detected"):
            out["events"].append({"t": c["t"], "kind": "camera", "what": "plate off its nest", "ref": c["plate"], "detail": "re-placed"})
    for e in b.get("events", []):
        if e.get("hidden"):
            continue
        if e["kind"] == "clean":
            out["events"].append({"t": e["t"], "kind": "clean", "what": f"cell line switch {e['prev']} -> {e['line']}", "ref": e["plate"], "detail": e["lh"]})
        elif e["kind"] == "qc_fail":
            out["events"].append({"t": e["t"], "kind": "qc", "what": f"Z' {e['z']} < {e['z_min']}", "ref": e["plate"], "detail": "plate invalid"})
        elif e["kind"] == "review":
            out["events"].append({"t": e["t"], "kind": "agent review", "what": e.get("review"), "ref": e.get("key"), "detail": e.get("note")})
        elif e["kind"] == "constraints_update":
            out["events"].append({"t": e["t"], "kind": "limits", "what": f"version {e['version']}", "ref": None, "detail": e.get("reason")})
    dc = loop.get("deck_check")
    if dc:
        for k, p in enumerate(dc["passes"]):
            bad = [r for r in p if not r["ok"]]
            out["events"].append({"t": 0, "kind": "deck check", "what": f"pass {k + 1}: {len(p) - len(bad)}/{len(p)} sites as designed",
                                  "ref": None, "detail": "; ".join(f"{r['id']}: {r['issue']}" for r in bad) or "launch"})
    out["events"].sort(key=lambda e: e["t"])

    # limits, every version
    for h in loop.get("constraint_history", []):
        for c in h.get("specs", []):
            src = c.get("source") or {}
            out["limits"].append({"t": h.get("t", 0), "version": h.get("version", 0), "limit": c.get("id"), "kind": c.get("kind"),
                                  "minutes": c.get("minutes"), "hard": c.get("hard"), "source": f"{src.get('doc')} p.{src.get('page')}" if src.get("page") else src.get("doc"),
                                  "assumed": bool(c.get("value_assumed")), "evidence": c.get("evidence", "doc"), "why": h.get("reason")})

    # sim only: wells the hidden limits corrupted, and what got through
    for e in b.get("events", []):
        if e["kind"] == "truth":
            out["truth"].append({"t": e["t"], "plate": e["plate"], "limit": e.get("limit"), "wells_corrupted": e.get("wells"),
                                 "kept": None, "detail": e.get("prev") or e.get("lh")})
    kept = {r["plate"]: r.get("corrupt", 0) for r in b["reads"]}
    for row in out["truth"]:
        row["kept"] = kept.get(row["plate"])  # corrupted values that passed QC (plate read, not invalidated)
    for e in (b.get("supplies_ledger") or []):  # liquids the lab made: powder, DMSO, medium, dilution plates, reservoirs
        unit = e["item"][e["item"].rfind("(") + 1:-1] if e["item"].endswith(")") else "each"
        ledger.append({"t": e["t"], "item": e["item"].split(" (")[0], "delta": e["delta"], "unit": unit, "ref": e["ref"]})
    ledger.sort(key=lambda x: x["t"])
    return out


def write_xlsx(tabs: dict[str, list[dict]], path) -> None:
    """One sheet per table, a header row, minutes converted to hours."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    wb.remove(wb.active)
    for name in TABLES:
        rows = tabs.get(name, [])
        ws = wb.create_sheet(name)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        hours = {"t", "dispensed", "read", "first_used", "last_used"}
        ws.append([f"{c} (h)" if c in hours else c for c in cols])
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if c in hours and isinstance(v, (int, float)):
                    v = round(v / 60, 2)
                elif isinstance(v, list):
                    v = " - ".join(f"{x / 60:.2f} h" for x in v) if c == "incubated" else ", ".join(map(str, v))
                vals.append(v)
            ws.append(vals)
        ws.freeze_panes = "A2"
    wb.save(path)
