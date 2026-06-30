"""
tool_semantic.py - semantic_search: questions over free text.

Hybrid retrieval over the short, acronym-dense free-text fields (activity
summaries, operation remarks, lithology, ...). BM25 catches exact jargon
('whipstock', '9 5/8"', 'TIH'); dense vectors catch paraphrase. The two
rankings are fused with Reciprocal Rank Fusion (RRF).

Indexing unit = one ROW (one text field of one report). report_id /
wellbore_id / period travel with it as metadata, so every hit maps back to a
report.

Dense vectors are PRECOMPUTED offline by build_embeddings.py and loaded from
disk (embeddings.npy + embeddings_docs.json). At runtime we only ever encode
the single user query - never the corpus - so the heavy work never lands on the
Streamlit thread. If the precomputed file is missing we run BM25-only (still
responsive) and tell the user to run build_embeddings.py.
"""
import json
import os
import re

import numpy as np

import db

EMB_PATH = os.environ.get("DDR_EMB_PATH", "embeddings.npy")
DOCS_PATH = os.environ.get("DDR_EMB_DOCS_PATH", "embeddings_docs.json")

# text fields worth indexing, as (table, column). Empty / 'None' rows dropped.
TEXT_SOURCES = [
    ("summary_of_activities", "content"),
    ("summary_of_planned_activities", "content"),
    ("operations", "main_sub_activity"),
    ("operations", "remark"),
    ("equipment_failure_information", "remark"),
    ("survey_station", "comment"),
    ("lithology_information", "lithology_description"),
    ("stratigraphic_information", "description"),
]

_index = None            # cached singleton built on first search


# ----------------------------------------------------------------- tokenizer
# English function words + generic query framing. Removed before BM25 scoring
# so a rare stopword (e.g. 'a' in 'took a kick') can't dominate via a high IDF.
# Domain tokens (9, 5, c1, tih, ...) are NOT in here and are kept.
_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "into", "over", "out", "up", "off", "and", "or", "but", "as",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "we", "i", "you", "he", "she", "they",
    "do", "does", "did", "has", "have", "had", "will", "would", "can", "any",
    "which", "what", "where", "when", "who", "how", "show", "list", "find",
    "give", "tell", "about", "mention", "mentioning", "mentioned", "report",
    "reports",
}


def _tok(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _STOP]


# ------------------------------------------------------------------- gather
def gather_docs():
    """Pull every non-empty text field as a doc with report metadata."""
    conn = db.connect()
    try:
        tables = set(db.list_tables(conn))
        docs = []
        for table, col in TEXT_SOURCES:
            if table not in tables:
                continue
            cols = {c for c, _ in db.table_columns(conn, table)}
            if col not in cols or "report_id" not in cols:
                continue
            sql = (
                f'SELECT t."{col}" AS text, r.report_id, r.wellbore_id, '
                f'r.period_start, r.period_end '
                f'FROM "{table}" t JOIN reports r ON t.report_id = r.report_id '
                f'WHERE t."{col}" IS NOT NULL AND TRIM(t."{col}") != "" '
                f'AND LOWER(TRIM(t."{col}")) != "none"'
            )
            for row in db.rows_as_dicts(conn, sql):
                docs.append({
                    "text": row["text"], "table": table, "column": col,
                    "report_id": row["report_id"],
                    "wellbore_id": row["wellbore_id"],
                    "period_start": row["period_start"],
                    "period_end": row["period_end"],
                })
        return docs
    finally:
        conn.close()


# --------------------------------------------------------------- build index
def _load_precomputed():
    """Load (docs, embeddings) written by build_embeddings.py, or None.

    The docs come from the artifact (not a fresh DB query) so row i of the
    embeddings always lines up with docs[i]. A count mismatch is a hard error -
    we don't guess."""
    if not (os.path.exists(EMB_PATH) and os.path.exists(DOCS_PATH)):
        return None
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    embeddings = np.load(EMB_PATH)
    if embeddings.shape[0] != len(docs):
        raise RuntimeError(
            f"embedding/doc mismatch: {embeddings.shape[0]} vs {len(docs)}. "
            f"Re-run build_embeddings.py.")
    return docs, embeddings


def _build_index():
    pre = _load_precomputed()
    if pre is not None:
        docs, embeddings = pre
    else:
        print("[semantic] no precomputed embeddings found "
              f"({EMB_PATH}); run build_embeddings.py for dense retrieval. "
              "Falling back to BM25-only.")
        docs = gather_docs()
        embeddings = None

    bm25 = None
    if docs:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([_tok(d["text"]) for d in docs])

    return {"docs": docs, "bm25": bm25, "embeddings": embeddings}


def get_index(force=False):
    global _index
    if _index is None or force:
        _index = _build_index()
    return _index


# ------------------------------------------------------------- RRF (pure fn)
def rrf(rank_lists, k=60, top_n=5):
    """Reciprocal Rank Fusion. rank_lists: list of ordered doc-index lists
    (best first). Returns fused doc indices, best first."""
    scores = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_n]


# ---------------------------------------------------------------- search
def semantic_search(question, top_n=5, pool=30):
    idx = get_index()
    docs = idx["docs"]
    if not docs:
        return {"ok": True, "tool": "semantic", "hits": [],
                "mode": "empty", "note": "no free-text rows indexed"}

    rank_lists = []
    mode = []

    if idx["bm25"] is not None:
        scores = idx["bm25"].get_scores(_tok(question))
        order = list(np.argsort(scores)[::-1][:pool])
        rank_lists.append([int(i) for i in order])
        mode.append("bm25")

    if idx["embeddings"] is not None:
        import embedder
        q = embedder.encode([question])[0]          # encode ONLY the query
        sims = idx["embeddings"] @ q
        order = list(np.argsort(sims)[::-1][:pool])
        rank_lists.append([int(i) for i in order])
        mode.append("vector")

    if not rank_lists:
        return {"ok": False, "tool": "semantic",
                "error": "no retriever available", "hits": []}

    fused = rrf(rank_lists, top_n=top_n) if len(rank_lists) > 1 else rank_lists[0][:top_n]
    hits = []
    for i in fused:
        d = docs[i]
        hits.append({
            "text": d["text"], "table": d["table"], "column": d["column"],
            "report_id": d["report_id"], "wellbore_id": d["wellbore_id"],
            "period_start": d["period_start"], "period_end": d["period_end"],
        })
    return {"ok": True, "tool": "semantic", "hits": hits,
            "mode": "+".join(mode), "n_docs": len(docs)}
