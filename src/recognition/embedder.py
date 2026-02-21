from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from loguru import logger


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray


class FaceEmbedder:
    def __init__(self) -> None:
        providers = _select_onnx_providers()
        logger.info("Face embedder ONNX providers: {}", providers)
        self._app = FaceAnalysis(providers=providers)
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


def _select_onnx_providers() -> list[str]:
    available = set(ort.get_available_providers())
    preferred_env = os.environ.get("FACE_ONNX_PROVIDERS", "").strip()
    if preferred_env:
        requested = [p.strip() for p in preferred_env.split(",") if p.strip()]
        selected = [p for p in requested if p in available]
        if selected:
            if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
                selected.append("CPUExecutionProvider")
            return selected

    preferred_order = [
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [p for p in preferred_order if p in available]
    if not selected:
        return ["CPUExecutionProvider"]
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected
