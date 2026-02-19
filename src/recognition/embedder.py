from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray


class FaceEmbedder:
    def __init__(self) -> None:
        self._app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1)

    def detect_faces(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        faces = self._app.get(frame_bgr)
        detected: list[DetectedFace] = []
        for face in faces:
            if face.normed_embedding is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            detected.append(DetectedFace((x1, y1, x2, y2), embedding))
        return detected

    def embedding_from_image_path(self, image_path: str) -> np.ndarray | None:
        image = cv2.imread(image_path)
        if image is None:
            return None
        faces = self.detect_faces(image)
        if not faces:
            return None
        return faces[0].embedding
