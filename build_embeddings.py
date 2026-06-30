"""
build_embeddings.py - offline, one-time (re-run when the DB changes).

Encodes every free-text doc ONCE and writes the vectors next to the docs they
came from. At runtime tool_semantic loads these and only ever encodes the
single user query - so no 25k-doc encode happens on the Streamlit thread, which
was what froze the machine.

Run:  python build_embeddings.py
Outputs (next to app.py):
  embeddings.npy        float32 (N, 384), L2-normalized, row i  <-> docs[i]
  embeddings_docs.json  the N docs in the SAME order (text + report metadata)

Re-run this whenever you re-extract DDRs or rebuild ddr.db.
"""
import json
import time

import numpy as np

import embedder
import tool_semantic

EMB_PATH = "embeddings.npy"
DOCS_PATH = "embeddings_docs.json"


def main():
    docs = tool_semantic.gather_docs()
    if not docs:
        raise SystemExit("No free-text rows found in the DB - nothing to embed. "
                         "Build ddr.db first (build_database.py).")

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} docs with fastembed (all-MiniLM-L6-v2)...")
    t0 = time.time()
    emb = embedder.encode(texts)
    dt = time.time() - t0

    if emb.shape[0] != len(docs):
        raise RuntimeError(
            f"embedding count {emb.shape[0]} != doc count {len(docs)}")

    np.save(EMB_PATH, emb)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print(f"done in {dt:.1f}s -> {EMB_PATH} {emb.shape} ({emb.nbytes/1e6:.1f} MB), "
          f"{DOCS_PATH} ({len(docs)} docs)")


if __name__ == "__main__":
    main()