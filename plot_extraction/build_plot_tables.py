"""
Load the pressure-plot extractions into the existing ddr.db.

Companion to build_database.py. The plot tables are an ADDITIVE island in the
same single-file DB: they share no key with the DDR reports (DDR wellbores look
like '15/9-19 A'; plot wells look like 'Well_05'), so DDR<->plot never joins.

Within the plot island the two plot kinds DO share a subject well_label:
formation 'Well_05_pressure_profile.png' and time-series 'pressure_time_plot_05'
both describe subject Well_05. The 4 series inside a time-series plot are OFFSET
(reference) wells local to that plot, stored as offset_label 'Offset_1'..'Offset_4'.

Folder layout produced by the two extractors (both write out/<png_name>/):

  formation plot  -> measured_points.csv   (depth_ft, pressure_psi)
                     reference_lines.csv    (depth_ft, MIN_psi, BASE_psi, MAX_psi)
                     extraction_summary.csv (quantity, value)   # virgin pressure
  time-series     -> points.csv            (well, date, pressure_psi)
                     extraction_summary.csv (well, n_points, ...) # ignored: derived

Discriminator: measured_points.csv present -> formation; points.csv -> time-series.

REAL for measured numbers (OpenCV output: no sentinels to preserve, and the
chatbot wants numeric comparisons); TEXT for labels/dates/provenance. No views,
no aggregates -- faithful storage of the extracted rows, nothing computed.

Idempotent: a plot whose source_file is already loaded is skipped, so re-running
on the full corpus and single-file Streamlit re-uploads are both safe.

Usage:
    python build_plot_tables.py                 # scans ./out
    python build_plot_tables.py /path/to/out    # scans that folder
    python build_plot_tables.py /path/to/out ddr.db
"""

import sys
import re
import sqlite3
from pathlib import Path

import pandas as pd

DB_NAME = "ddr.db"


def q(ident):
    return '"' + ident.replace('"', "") + '"'


# --------------------------------------------------------------------- schema
def create_tables(conn):
    """CREATE IF NOT EXISTS so this is additive and safe to re-run. It never
    touches the DDR tables, and build_database.py's DROP loop never touches
    these (their names aren't in schema_reference.json)."""
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS formation_pressure_plots (
            plot_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file         TEXT,
            well_label          TEXT,
            virgin_pressure_psi REAL
        );
        CREATE TABLE IF NOT EXISTS formation_measured_points (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plot_id      INTEGER,
            depth_ft     REAL,
            pressure_psi REAL,
            FOREIGN KEY(plot_id) REFERENCES formation_pressure_plots(plot_id)
        );
        CREATE TABLE IF NOT EXISTS formation_reference_lines (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            plot_id   INTEGER,
            depth_ft  REAL,
            min_psi   REAL,
            base_psi  REAL,
            max_psi   REAL,
            FOREIGN KEY(plot_id) REFERENCES formation_pressure_plots(plot_id)
        );
        CREATE TABLE IF NOT EXISTS timeseries_pressure_plots (
            ts_plot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            well_label  TEXT          -- subject well, e.g. 'Well_05'
        );
        CREATE TABLE IF NOT EXISTS timeseries_pressure_points (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_plot_id   INTEGER,
            offset_label TEXT,        -- reference well, local to the plot: 'Offset_1'..'Offset_4'
            obs_month    TEXT,
            pressure_psi REAL,
            FOREIGN KEY(ts_plot_id) REFERENCES timeseries_pressure_plots(ts_plot_id)
        );
        """
    )
    conn.commit()


# ---------------------------------------------------------------- csv helpers
def read_rows(path):
    """Read a CSV into a list of dicts, converting blank/NaN cells to None so
    SQLite stores a real NULL (not a float nan). Missing/empty file -> []."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    df = df.where(pd.notnull(df), None)
    return df.to_dict("records")


def scalar(path, key_col, key, val_col):
    """Pull one value from a (key, value)-shaped summary CSV. None if absent."""
    for r in read_rows(path):
        if str(r.get(key_col)) == key:
            v = r.get(val_col)
            return None if v is None else float(v)
    return None


_WELL = re.compile(r"(Well[_\- ]?\d+)", re.IGNORECASE)


def well_label_from_name(name):
    """Formation plots don't carry the well label in any CSV -- recover it
    best-effort from the PNG filename; None if the name doesn't encode it."""
    m = _WELL.search(name)
    return m.group(1) if m else None


# -------------------------------------------------------------- kind dispatch
def detect_kind(folder):
    if (folder / "measured_points.csv").exists():
        return "formation"
    if (folder / "points.csv").exists():
        return "timeseries"
    return None


def already_loaded(conn, table, source_file):
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM {q(table)} WHERE source_file = ? LIMIT 1",
                (source_file,))
    return cur.fetchone() is not None


# ------------------------------------------------------------------- loaders
def load_formation_plot(conn, folder):
    """folder name == source PNG filename. Reusable as the per-plot unit for
    a Streamlit single upload (point it at one extracted folder)."""
    cur = conn.cursor()
    source_file = folder.name
    if already_loaded(conn, "formation_pressure_plots", source_file):
        return "skipped"

    vp = scalar(folder / "extraction_summary.csv",
                "quantity", "virgin_pressure_psi", "value")
    cur.execute(
        "INSERT INTO formation_pressure_plots "
        "(source_file, well_label, virgin_pressure_psi) VALUES (?, ?, ?)",
        (source_file, well_label_from_name(source_file), vp),
    )
    plot_id = cur.lastrowid

    pts = read_rows(folder / "measured_points.csv")
    cur.executemany(
        "INSERT INTO formation_measured_points "
        "(plot_id, depth_ft, pressure_psi) VALUES (?, ?, ?)",
        [(plot_id, r.get("depth_ft"), r.get("pressure_psi")) for r in pts],
    )

    refs = read_rows(folder / "reference_lines.csv")
    cur.executemany(
        "INSERT INTO formation_reference_lines "
        "(plot_id, depth_ft, min_psi, base_psi, max_psi) VALUES (?, ?, ?, ?, ?)",
        [(plot_id, r.get("depth_ft"), r.get("MIN_psi"),
          r.get("BASE_psi"), r.get("MAX_psi")) for r in refs],
    )
    conn.commit()
    return "loaded"


def subject_well_from_ts_name(name):
    """'pressure_time_plot_05.png' -> 'Well_05' (the plot's subject well)."""
    m = re.search(r"(\d+)", name)
    return f"Well_{int(m.group(1)):02d}" if m else None


def offset_label_from_series(series):
    """Local series name 'Well_03' -> 'Offset_3' (reference well in this plot)."""
    m = re.search(r"(\d+)", str(series or ""))
    return f"Offset_{int(m.group(1))}" if m else series


def load_timeseries_plot(conn, folder):
    cur = conn.cursor()
    source_file = folder.name
    if already_loaded(conn, "timeseries_pressure_plots", source_file):
        return "skipped"

    cur.execute(
        "INSERT INTO timeseries_pressure_plots (source_file, well_label) "
        "VALUES (?, ?)",
        (source_file, subject_well_from_ts_name(source_file)),
    )
    ts_plot_id = cur.lastrowid

    pts = read_rows(folder / "points.csv")
    cur.executemany(
        "INSERT INTO timeseries_pressure_points "
        "(ts_plot_id, offset_label, obs_month, pressure_psi) VALUES (?, ?, ?, ?)",
        [(ts_plot_id, offset_label_from_series(r.get("well")),
          r.get("date"), r.get("pressure_psi")) for r in pts],
    )
    conn.commit()
    return "loaded"


def load_plot_folder(conn, folder):
    kind = detect_kind(folder)
    if kind == "formation":
        return load_formation_plot(conn, folder)
    if kind == "timeseries":
        return load_timeseries_plot(conn, folder)
    return "unknown"


# ---------------------------------------------------------------------- main
def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./out")
    db_name = sys.argv[2] if len(sys.argv) > 2 else DB_NAME

    if not out_dir.is_dir():
        print(f"Output folder not found: {out_dir.resolve()}")
        sys.exit(1)
    folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
    if not folders:
        print(f"No extraction subfolders in {out_dir.resolve()}")
        sys.exit(1)

    conn = sqlite3.connect(db_name)
    create_tables(conn)
    counts = {"loaded": 0, "skipped": 0, "unknown": 0, "failed": 0}
    for folder in folders:
        try:
            result = load_plot_folder(conn, folder)
            counts[result] += 1
            if result == "unknown":
                print(f"  SKIP {folder.name}: no recognised plot CSVs")
        except Exception as e:
            counts["failed"] += 1
            print(f"  FAILED {folder.name}: {e}")
    conn.close()

    print(f"\nPlots into {db_name}: {counts['loaded']} loaded, "
          f"{counts['skipped']} already present, "
          f"{counts['unknown']} unrecognised, {counts['failed']} failed")


if __name__ == "__main__":
    main()