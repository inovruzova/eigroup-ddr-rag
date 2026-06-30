"""
embedder.py - the one place that turns text into dense vectors.

Shared by the offline build step (build_embeddings.py) and the runtime query
path (tool_semantic.py) so the corpus vectors and the query vector always come
from the SAME model. Mixing models would silently wreck cosine similarity.

Backend: fastembed (ONNX runtime, no torch). ~10x lighter to install and load
than sentence-transformers/torch, which matters on Streamlit Community Cloud
(1 GB RAM) and is what removes the "loading weights" lag. Model is
all-MiniLM-L6-v2 (384-dim); output rows are L2-normalized so cosine similarity
is a plain dot product.
"""
import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

_model = None


def _load():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=_MODEL_NAME)


def encode(texts):
    """Encode a list of strings -> (N, DIM) float32, L2-normalized rows."""
    _load()
    vecs = np.asarray(list(_model.embed(list(texts))), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
