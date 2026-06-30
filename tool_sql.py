"""
tool_sql.py - construct_sql: questions answerable directly from columns.

NL question -> the LLM writes one SQLite SELECT (grounded in the data
dictionary) -> we validate it is read-only -> execute on the read-only
connection -> return rows. The data dictionary is what makes the SQL correct
(sentinels, the no-join rule, opaque column names).
"""
import re

import db
import data_dictionary
import llm

ROW_LIMIT = 200          # hard cap fetched from the DB
ROWS_TO_LLM = 50         # how many we hand to the synthesizer

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|truncate|begin|commit)\b", re.I)

_SYSTEM = (
    "You translate questions about drilling reports into a single SQLite "
    "SELECT. Use ONLY the tables/columns given. Obey every rule. Output JSON: "
    '{"sql": "<one SELECT>", "rationale": "<one sentence>"}. '
    "If the question cannot be answered from these tables, return "
    '{"sql": "", "rationale": "why not"}.'
)


def validate_sql(sql):
    """Defence in depth (the connection is already read-only). Raises on
    anything that is not a single SELECT/CTE."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        raise ValueError("empty query")
    if ";" in s:
        raise ValueError("only a single statement is allowed")
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("only SELECT queries are allowed")
    if _FORBIDDEN.search(s):
        raise ValueError("query contains a forbidden keyword")
    return s


def construct_sql(question):
    conn = db.connect()
    try:
        dictionary = data_dictionary.dictionary_text(conn)
        prompt = (
            f"{dictionary}\n\n"
            f"Question: {question}\n\n"
            "Write the SQL now as JSON."
        )
        try:
            out = llm.chat_json(_SYSTEM, prompt)
        except Exception as e:
            return {"ok": False, "tool": "sql", "error": f"LLM error: {e}",
                    "sql": None, "rows": []}

        raw_sql = (out.get("sql") or "").strip()
        rationale = out.get("rationale", "")
        if not raw_sql:
            return {"ok": False, "tool": "sql",
                    "error": rationale or "no SQL produced",
                    "sql": None, "rows": []}

        try:
            sql = validate_sql(raw_sql)
        except ValueError as e:
            return {"ok": False, "tool": "sql", "error": f"unsafe SQL: {e}",
                    "sql": raw_sql, "rows": []}

        try:
            cols, rows = db.run_select(conn, sql, limit=ROW_LIMIT)
        except Exception as e:
            return {"ok": False, "tool": "sql", "error": f"SQL failed: {e}",
                    "sql": sql, "rationale": rationale, "rows": []}

        truncated = len(rows) >= ROW_LIMIT
        return {
            "ok": True, "tool": "sql", "sql": sql, "rationale": rationale,
            "columns": cols, "rows": rows[:ROWS_TO_LLM],
            "row_count": len(rows), "truncated": truncated,
        }
    finally:
        conn.close()