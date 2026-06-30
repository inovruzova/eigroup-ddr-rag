"""
Check column-name consistency across extracted DDR JSONs — registry style.

Reports table
-------------
The report-metadata block is a flat list of "Label: value" pairs and is read
from the page TEXT, anchored on the known label set (learned from the files
where the block extracted cleanly). This is robust to layouts where
find_tables() misses the block (e.g. some older reports), because the text
always contains the labels. Anchoring on known labels avoids being fooled by
colons inside time values (00:00, 13:51).

Other sections
--------------
Named by their document heading. Orientation is decided by whether column 0
holds data values (horizontal) or only labels (vertical/record). Drilling
Fluid's group labels (Solids / Viscometer tests / Filtration tests) prefix
their member fields; the rows of a record that splits across a page break are
accumulated in order before prefixing, so the prefix carries across the split.

Noise handling
--------------
Tables with <=1 non-empty cell (the title banner, stranded one-cell headings,
empty stubs) are skipped. Horizontal continuation tables (a section's data
overflowing onto the next page) are skipped for column purposes — their first
row is data, not a header.

Outputs:
  - inconsistencies.txt    only the column mismatches, with missing/extra diff
  - schema_reference.json   {table_name: reference columns}, ready for DDL

Usage:
    python consistency_check.py [/path/to/jsons]
"""

import sys
import re
import json
from datetime import datetime
from pathlib import Path

GROUP_LABELS = {"solids", "viscometer tests", "filtration tests"}


def norm(name):
    if name is None:
        return ""
    s = str(name).replace("\n", " ")
    s = re.sub(r"\([^)]*\)", "", s)
    s = " ".join(s.split())
    return s.strip().strip(":").strip().lower()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", norm(name)).strip("_")


def non_empty(cell):
    return cell is not None and str(cell).strip() != ""


def count_nonempty(rows):
    return sum(non_empty(c) for r in rows for c in r)


def looks_like_value(cell):
    if not non_empty(cell):
        return False
    s = str(cell).strip()
    return bool(re.match(r"^[+-]?\d*\.?\d+$", s)
                or re.match(r"^\d{1,2}:\d{2}$", s)
                or re.match(r"^\d{4}-\d{2}-\d{2}", s))


# --- column type inference (deterministic, evidence-driven) ----------------
# Storage types are decided by scanning every non-empty value in a column.
# A column earns a numeric/date type only if EVERY value conforms (100%);
# any exception keeps it TEXT, which makes the non-conforming columns the
# data-quality review list rather than a silent coercion.
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?$")
# A leading zero (01, 007, -09) means the width/format is meaningful — an
# identifier or code, not a quantity — so the column stays TEXT. Does NOT
# fire on "0" or on decimals like "0.5" (the 0 there is a real integer part).
_LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")


def is_code_name(name):
    # Codes are opaque identifiers you never do arithmetic on. All-digit codes
    # (e.g. an IADC code "437") would otherwise type as INTEGER, so guard them
    # by name. The leading-zero rule already catches padded codes structurally;
    # this is the backstop for codes that happen not to be padded.
    return name == "iadc_code" or name.endswith("_code")


def infer_column_type(name, values):
    """Return one of INTEGER, REAL, DATE, TEXT for a column, given its name and
    the set of non-empty values seen across the whole corpus."""
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not vals:
        return "TEXT"
    if is_code_name(name):
        return "TEXT"
    if any(_LEADING_ZERO_RE.match(v) for v in vals):
        return "TEXT"
    if all(_INT_RE.match(v) for v in vals):
        return "INTEGER"
    if all(_FLOAT_RE.match(v) for v in vals):
        return "REAL"
    if all(_DATE_RE.match(v) for v in vals):
        return "DATE"
    return "TEXT"


def is_horizontal(rows):
    # primary signal: data values in column 0 below the header row
    for r in rows[1:]:
        if r and looks_like_value(r[0]):
            return True
    # fallback for a header-only table (its data rows are on the next page, so
    # there are no data values anywhere yet): if row 0 is a multi-cell label
    # header, it's horizontal. A real vertical record always has at least one
    # data value somewhere, so it never reaches this branch.
    if not any(looks_like_value(c) for r in rows for c in r):
        row0 = [c for c in (rows[0] if rows else []) if non_empty(c)]
        if len(row0) > 1:
            return True
    return False


def is_banner(rows):
    cells = [norm(c) for r in rows for c in r if non_empty(c)]
    if not cells:
        return True
    return all(c.replace(" ", "") == "summaryreport" or c.startswith("period")
               for c in cells)


def ordered_unique(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def record_columns(rows):
    """Columns of a key:value record, with group-label prefixing.

    Run over the FULL set of a record's rows (including continuation rows from
    a page split) so the active group prefix carries across the split.
    """
    cols, prefix = [], None
    for r in rows:
        if not r or not non_empty(r[0]):
            continue
        nkey = norm(r[0])
        val = r[1] if len(r) > 1 else None
        # A group label is value-less and has no unit parens; this separates
        # the label "Solids" from the member field "Solids ()".
        if nkey in GROUP_LABELS and not non_empty(val) and "(" not in str(r[0]):
            prefix = slug(nkey)
            continue
        c = slug(r[0])
        if not c:
            continue
        cols.append(f"{prefix}_{c}" if prefix else c)
    return cols


# ---------------------------------------------------------------------------
# Reports-from-text
# ---------------------------------------------------------------------------
def gather_report_stems(all_data):
    """Learn the report label stems from the colon key:value blocks, any file."""
    stems, seen = [], set()
    for data in all_data:
        for pg in data["pages"]:
            for table in pg["tables"]:
                if table.get("kind", "table") != "table" or table.get("heading"):
                    continue
                rows = table.get("rows") or []
                keys = [r[0] for r in rows if r and non_empty(r[0])]
                if not keys:
                    continue
                colon = sum(str(k).rstrip().endswith(":")
                            for k in keys) / len(keys)
                if colon > 0.6:
                    for k in keys:
                        s = norm(k)
                        if s and s not in seen:
                            seen.add(s)
                            stems.append(s)
    return stems


def label_regex(stem):
    words = r"\s+".join(re.escape(w) for w in stem.split())
    return re.compile(r"\b" + words + r"\s*(?:\([^)]*\))?\s*:", re.IGNORECASE)


def reports_from_text(data, stems):
    text = data["pages"][0]["text"] if data["pages"] else ""
    return [slug(s) for s in stems if label_regex(s).search(text or "")]


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------
def extract_tables(data, report_stems):
    record_rows, horiz, text_cols, unrecognized = {}, {}, {}, []

    for pg in data["pages"]:
        for ti, table in enumerate(pg["tables"]):
            kind = table.get("kind", "table")
            heading = table.get("heading")

            if kind == "text":
                if heading:
                    text_cols[slug(heading)] = ["content"]
                continue

            rows = table.get("rows") or []
            if count_nonempty(rows) <= 1 or is_banner(rows):
                continue  # title, stranded one-cell heading, empty stub

            if not is_horizontal(rows):
                keys = [r[0] for r in rows if r and non_empty(r[0])]
                if not keys:
                    continue
                if heading:
                    record_rows.setdefault(slug(heading), []).extend(rows)
                else:
                    colon = sum(str(k).rstrip().endswith(":")
                                for k in keys) / len(keys)
                    if colon > 0.6:
                        continue  # report metadata: handled from text
                    unrecognized.append((pg["page_number"], ti, heading, rows[0]))
            else:
                if table.get("continuation"):
                    continue  # data continuation: row0 is data, not a header
                if heading:
                    horiz.setdefault(slug(heading),
                                     [slug(c) for c in rows[0]])
                else:
                    unrecognized.append((pg["page_number"], ti, heading, rows[0]))

    tables = {}
    rc = reports_from_text(data, report_stems)
    if rc:
        tables["reports"] = ordered_unique(rc)
    for name, rws in record_rows.items():
        tables[name] = ordered_unique(record_columns(rws))
    for name, cols in horiz.items():
        tables.setdefault(name, ordered_unique(cols))
    tables.update(text_cols)
    return tables, unrecognized


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    json_paths = sorted(folder.glob("*.extracted.json"))
    if not json_paths:
        print(f"No *.extracted.json files found in {folder.resolve()}")
        sys.exit(1)

    all_data = [json.loads(p.read_text()) for p in json_paths]
    report_stems = gather_report_stems(all_data)

    # registry[table] = {"order": union of cols (first-seen), "count": {col:n},
    #                    "files": how many files contain this table}
    registry, unrecognized_all = {}, []
    for path, data in zip(json_paths, all_data):
        tables, unrec = extract_tables(data, report_stems)
        for name, cols in tables.items():
            reg = registry.setdefault(name, {"order": [], "count": {},
                                             "files": 0})
            reg["files"] += 1
            for c in cols:
                if c not in reg["count"]:
                    reg["order"].append(c)
                reg["count"][c] = reg["count"].get(c, 0) + 1
        for page_no, ti, heading, preview in unrec:
            unrecognized_all.append((path.name, page_no, ti, heading, preview))
        print(f"{path.name}: {len(tables)} section(s)")

    # ---- schema_reference.json: the UNION of columns per table ----
    schema = {name: reg["order"] for name, reg in registry.items()}
    Path("./ddr_extraction/schema_reference.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False))

    # ---- schema_report.txt: universal vs partial columns per table ----
    lines = ["SCHEMA REPORT (union of all columns seen)",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Files scanned: {len(json_paths)}",
             "Each table's schema is the union of every column seen.",
             "Partial columns (present in only some files) are listed so a real",
             "variant (e.g. tvd vs mtvd, in many files) is distinguishable from",
             "an extraction artifact (in just one or two).", ""]
    partial_total = 0
    for name, reg in registry.items():
        partial = [(c, reg["count"][c]) for c in reg["order"]
                   if reg["count"][c] < reg["files"]]
        lines += ["=" * 60,
                  f"TABLE: {name}   ({reg['files']} file(s), "
                  f"{len(reg['order'])} columns)",
                  "=" * 60]
        if not partial:
            lines.append("  all columns present in every file")
        else:
            partial_total += len(partial)
            lines.append("  PARTIAL columns (col: files_with_it / "
                         "files_with_table):")
            for c, n in sorted(partial, key=lambda x: x[1]):
                lines.append(f"    {c}: {n}/{reg['files']}")
        lines.append("")

    if unrecognized_all:
        lines += ["=" * 60, "UNRECOGNIZED (orphaned tables — review)", "=" * 60]
        for fname, page_no, ti, heading, preview in unrecognized_all:
            lines.append(f"  {fname}  page {page_no} table {ti}: {preview}")
        lines.append("")

    Path("./ddr_extraction/schema_report.txt").write_text("\n".join(lines))

    print("\n" + "-" * 50)
    print(f"Tables: {len(registry)}")
    print(f"Partial (variant) columns across all tables: {partial_total}")
    print(f"Unrecognized: {len(unrecognized_all)}")
    print("Wrote: schema_reference.json (union schema), schema_report.txt")


if __name__ == "__main__":
    main()