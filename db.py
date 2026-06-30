"""
db.py - the one place that touches ddr.db.

Opened READ-ONLY: the chatbot can never write to the corpus, no matter what
the LLM produces. A fresh connection per call keeps things thread-safe and
stupid simple (sqlite connections are cheap).
"""
import os
import sqlite3

DB_PATH = os.environ.get("DDR_DB_PATH", "ddr.db")

# The five plot tables live in a separate namespace from the DDR tables.
# Nothing joins across this boundary (see data_dictionary.GLOBAL_NOTES).
PLOT_TABLES = {
    "formation_pressure_plots", "formation_measured_points",
    "formation_reference_lines", "timeseries_pressure_plots",
    "timeseries_pressure_points",
}


def connect(db_path=None):
    path = db_path or DB_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Database not found: {path}. Run build_database.py / "
            f"build_plot_tables.py first, or set DDR_DB_PATH.")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def list_tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    return [r[0] for r in rows]


def table_columns(conn, table):
    """[(name, affinity), ...] from PRAGMA table_info."""
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [(r[1], (r[2] or "TEXT")) for r in rows]


def schema_text(conn):
    """Compact one-line-per-table schema, used to ground Text2SQL."""
    out = []
    for t in list_tables(conn):
        cols = ", ".join(f"{n} {a}" for n, a in table_columns(conn, t))
        out.append(f"{t}({cols})")
    return "\n".join(out)


def run_select(conn, sql, limit=200):
    """Execute an (already-validated) SELECT and return (columns, rows)."""
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [list(r) for r in cur.fetchmany(limit)]
    return cols, rows


def rows_as_dicts(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]