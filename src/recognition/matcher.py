from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_db import FaceDatabase


@dataclass(slots=True)
class MatchResult:
    name: str | None
    similarity: float
    is_match: bool


class FaceMatcher:
    def __init__(self, similarity_threshold: float) -> None:
        self._threshold = similarity_threshold

    def match(self, embedding: np.ndarray, face_db: FaceDatabase) -> MatchResult:
        if face_db.is_empty:
            return MatchResult(name=None, similarity=0.0, is_match=False)

        vector = embedding.astype(np.float32)
        scores = face_db.embeddings @ vector
        index = int(np.argmax(scores))
        score = float(scores[index])
        name = str(face_db.names[index])
        return MatchResult(name=name, similarity=score, is_match=score >= self._threshold)
