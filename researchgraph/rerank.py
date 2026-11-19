"""Cross-encoder reranker (B3+). Runs locally on CPU or Apple-silicon GPU (MPS).

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters, ~90 MB): fast enough to
score 30-45 query/passage pairs in well under a second. Override with RG_RERANKER=<HF model id>,
e.g. BAAI/bge-reranker-base for a larger, slower model.
"""
from __future__ import annotations

import os

MODEL = os.environ.get("RG_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder
        # CPU by default: the 22M-parameter model is fast enough there, and the Apple GPU (MPS)
        # shares memory with the rest of the system and ran out during long evaluation runs.
        device = os.environ.get("RG_RERANK_DEVICE", "cpu")
        _model = CrossEncoder(MODEL, device=device, max_length=512)
    return _model


def scores(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    return [float(s) for s in _load().predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)]
