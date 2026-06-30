"""
Build a SQLite database from the extracted DDR JSONs.

Simplest deployable store: one file (ddr.db), no server, works directly with
Streamlit via sqlite3 + pandas. Every column is TEXT so values are stored
verbatim (-999.99, placeholder dates, empty units) with no type coercion.

Schema (from schema_reference.json):
  reports                : one row per report. Surrogate report_id PK, plus
                           wellbore_id + period (from page text) and the metadata
                           columns. UNIQUE(wellbore_id, report_number).
  <every other section>  : child table, id PK + report_id FK -> reports.report_id.
                           Text sections (summaries) become a 'content' column.

Usage:
    python build_database.py                 # scans current folder
    python build_database.py /path/to/jsons  # scans that folder
"""

import sys
import re
import json
import sqlite3
from pathlib import Path

from ddr_extraction.verify_ddr_table import (
    norm, slug, non_empty, count_nonempty, looks_like_value,
    is_horizontal, is_banner, GROUP_LABELS, gather_report_stems, label_regex,
    infer_column_type,
)

DB_NAME = "ddr.db"


def q(ident):
    return '"' + ident.replace('"', "") + '"'


def parse_wellbore_period(text):
    text = text or ""
    wb = re.search(r"Wellbore:\s*(.+?)\s+Period:", text)
    wellbore_id = wb.group(1).strip() if wb else None
    pe = re.search(r"Period:\s*(\S.*?\S)\s+-\s+(\S.*?\S)(?:\n|$)", text)
    if pe:
        return wellbore_id, pe.group(1).strip(), pe.group(2).strip()
    return wellbore_id, None, None


def parse_reports_values(text, stems):
    text = text or ""
    matches = []
    for stem in stems:
        m = label_regex(stem).search(text)
        if m:
            matches.append((m.start(), m.end(), slug(stem)))
    matches.sort()
    out = {}
    for i, (s, e, name) in enumerate(matches):
        bound = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        nl = text.find("\n", e)
        if nl != -1:
            bound = min(bound, nl)
        val = text[e:bound].strip()
        out[name] = val or None
    return out


def record_samples(rows):
    n_samples = max((len(r) for r in rows), default=1) - 1
    n_samples = max(n_samples, 1)
    samples = [dict() for _ in range(n_samples)]
    prefix = None
    for r in rows:
        if not r or not non_empty(r[0]):
            continue
        nkey = norm(r[0])
        valless = len(r) < 2 or not non_empty(r[1])
        if nkey in GROUP_LABELS and valless and "(" not in str(r[0]):
            prefix = slug(nkey)
            continue
        col = slug(r[0])
        if not col:
            continue
        if prefix:
            col = f"{prefix}_{col}"
        for i in range(n_samples):
            samples[i][col] = r[i + 1] if i + 1 < len(r) else None
    return [s for s in samples if any(non_empty(v) for v in s.values())]


def horizontal_rows(recs):
    header = None
    for rec in recs:
        rows = rec.get("rows") or []
        if (rows and len([c for c in rows[0] if non_empty(c)]) >= 2
                and not any(looks_like_value(c) for c in rows[0])):
            header = [slug(c) for c in rows[0]]
            break
    if header is None:
        first = recs[0].get("rows") or [[]]
        header = [slug(c) for c in first[0]]

    out = []
    for rec in recs:
        for r in rec.get("rows") or []:
            if not any(non_empty(c) for c in r):
                continue
            if [slug(c) for c in r] == header:
                continue
            row = {header[i]: (r[i] if i < len(r) else None)
                   for i in range(len(header)) if header[i]}
            if any(non_empty(v) for v in row.values()):
                out.append(row)
    return out


def section_rows(data):
    groups, texts = {}, {}
    for pg in data["pages"]:
        for rec in pg["tables"]:
            if rec.get("kind") == "text":
                if rec.get("heading"):
                    texts.setdefault(slug(rec["heading"]), []).append(
                        rec.get("text") or "")
                continue
            heading = rec.get("heading")
            if not heading:
                continue
            rows = rec.get("rows") or []
            if is_banner(rows) or count_nonempty(rows) <= 1:
                continue
            groups.setdefault(slug(heading), []).append(rec)

    out = {}
    for name, contents in texts.items():
        joined = "\n".join(c for c in contents if c and c.strip()).strip()
        out[name] = [{"content": joined}] if joined else []
    for name, recs in groups.items():
        all_rows = [r for rec in recs for r in rec.get("rows") or []]
        out[name] = (horizontal_rows(recs) if is_horizontal(all_rows)
                     else record_samples(all_rows))
    return out


# DATE has no native SQLite type — it is stored as ISO TEXT (which sorts
# correctly and works with date()/strftime()). The semantic type is recorded
# in column_types.json; the storage affinity used in the DDL is this map.
STORAGE = {"INTEGER": "INTEGER", "REAL": "REAL", "DATE": "TEXT", "TEXT": "TEXT"}


def gather_column_values(all_data, stems, schema):
    """First pass: collect the distinct non-empty values seen for every column,
    using the same extraction the loader uses, so types are inferred from
    exactly the values that will be stored."""
    from collections import defaultdict
    schema_set = {t: set(cols) for t, cols in schema.items()}
    vals = {t: defaultdict(set) for t in schema}
    for data in all_data:
        text0 = data["pages"][0]["text"] if data["pages"] else ""
        rv = parse_reports_values(text0, stems)
        for c in schema["reports"]:
            v = rv.get(c)
            if v is not None and str(v).strip():
                vals["reports"][c].add(str(v).strip())
        for name, rowdicts in section_rows(data).items():
            if name not in schema or name == "reports":
                continue
            for rd in rowdicts:
                for c, v in rd.items():
                    if c in schema_set[name] and v is not None and str(v).strip():
                        vals[name][c].add(str(v).strip())
    return vals


def infer_types(all_data, stems, schema):
    vals = gather_column_values(all_data, stems, schema)
    types = {}
    for table, cols in schema.items():
        types[table] = {c: infer_column_type(c, vals[table].get(c, ()))
                        for c in cols}
    return types


def create_tables(conn, schema, types):
    cur = conn.cursor()
    for table, cols in schema.items():
        coldefs = [f"{q(c)} {STORAGE[types[table].get(c, 'TEXT')]}" for c in cols]
        if table == "reports":
            # wellbore_id/period are derived (parsed from text), not inferred:
            # wellbore_id is an identifier (TEXT); periods are ISO dates (TEXT).
            defs = ["report_id INTEGER PRIMARY KEY AUTOINCREMENT",
                    "wellbore_id TEXT", "period_start TEXT", "period_end TEXT"]
            defs += coldefs
            # No natural-key constraint: report_number restarts per drilling
            # campaign, so (wellbore_id, report_number) is not unique. The
            # surrogate report_id is the only identity we can defend.
        else:
            defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT", "report_id INTEGER"]
            defs += coldefs
            defs.append("FOREIGN KEY(report_id) REFERENCES reports(report_id)")
        cur.execute(f"DROP TABLE IF EXISTS {q(table)}")
        cur.execute(f"CREATE TABLE {q(table)} ({', '.join(defs)})")
    conn.commit()



def load_file(conn, data, stems, schema):
    cur = conn.cursor()
    text0 = data["pages"][0]["text"] if data["pages"] else ""
    rvals = parse_reports_values(text0, stems)
    wellbore_id, ps, pe = parse_wellbore_period(text0)

    rcols = ["wellbore_id", "period_start", "period_end"] + schema["reports"]
    rrow = {"wellbore_id": wellbore_id, "period_start": ps, "period_end": pe}
    for c in schema["reports"]:
        rrow[c] = rvals.get(c)
    cur.execute(
        f"INSERT INTO reports ({', '.join(q(c) for c in rcols)}) "
        f"VALUES ({', '.join('?' for _ in rcols)})",
        [rrow[c] for c in rcols],
    )
    report_id = cur.lastrowid

    for name, rowdicts in section_rows(data).items():
        if name not in schema or name == "reports":
            continue
        valid = set(schema[name])
        for rd in rowdicts:
            cols = [c for c in rd if c in valid]
            if not cols:
                continue
            allcols = ["report_id"] + cols
            cur.execute(
                f"INSERT INTO {q(name)} ({', '.join(q(c) for c in allcols)}) "
                f"VALUES ({', '.join('?' for _ in allcols)})",
                [report_id] + [rd[c] for c in cols],
            )
    conn.commit()


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./ddr_extraction/parsed_pdfs_coordinates")
    schema_path = Path("./ddr_extraction/schema_reference.json")
    if not schema_path.exists():
        print("schema_reference.json not found — run verify_ddr_table.py first.")
        sys.exit(1)
    schema = json.loads(schema_path.read_text())

    json_paths = sorted(folder.glob("*.extracted.json"))
    if not json_paths:
        print(f"No *.extracted.json files found in {folder.resolve()}")
        sys.exit(1)
    all_data = [json.loads(p.read_text()) for p in json_paths]
    stems = gather_report_stems(all_data)

    types = infer_types(all_data, stems, schema)
    Path("./ddr_extraction/column_types.json").write_text(json.dumps(types, indent=2))
    counts = {}
    for cols in types.values():
        for tp in cols.values():
            counts[tp] = counts.get(tp, 0) + 1
    print("Inferred column types: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    conn = sqlite3.connect(DB_NAME)
    create_tables(conn, schema, types)
    loaded, failed = 0, 0
    for p, data in zip(json_paths, all_data):
        try:
            load_file(conn, data, stems, schema)
            loaded += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED {p.name}: {e}")
    conn.close()
    print(f"\nLoaded {loaded} report(s) into {DB_NAME}"
          + (f", {failed} failed" if failed else ""))


if __name__ == "__main__":
    main()