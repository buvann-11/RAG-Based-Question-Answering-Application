from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Iterable, List

import numpy as np

from .config import EMBEDDING_MODEL_NAME

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class Embedder:
    def __init__(self) -> None:
        self._model = None
        if SentenceTransformer is not None:
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self._model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    local_files_only=True,
                )
            except Exception:
                self._model = None

    @property
    def uses_fallback(self) -> bool:
        return self._model is None

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.zeros((0, 384), dtype=float)
        if self._model is not None:
            vectors = self._model.encode(values, normalize_embeddings=True)
            return np.asarray(vectors, dtype=float)
        return np.vstack([self._fallback_vector(text) for text in values])

    def _fallback_vector(self, text: str) -> np.ndarray:
        # Deterministic local embedding approximation when the transformer
        # model is not available. It preserves offline operability.
        vector = np.zeros(384, dtype=float)
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 2):
                bucket = digest[index] % vector.size
                vector[bucket] += ((digest[index + 1] / 255.0) - 0.5)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def serialize_vector(vector: np.ndarray) -> str:
    return ",".join(f"{value:.8f}" for value in vector.tolist())


def deserialize_vector(raw_value: str) -> np.ndarray:
    if not raw_value:
        return np.zeros(384, dtype=float)
    values: List[float] = [float(item) for item in raw_value.split(",")]
    return np.asarray(values, dtype=float)
