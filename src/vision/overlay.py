from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class FaceOverlay:
    bbox: tuple[int, int, int, int]
    label: str
    similarity: float | None
    is_match: bool


def draw_overlays(frame: np.ndarray, overlays: Iterable[FaceOverlay]) -> np.ndarray:
    output = frame.copy()
    for overlay in overlays:
        x1, y1, x2, y2 = overlay.bbox
        color = (0, 200, 0) if overlay.is_match else (0, 120, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        score_text = ""
        if overlay.similarity is not None:
            score_text = f" ({overlay.similarity:.2f})"
        text = f"{overlay.label}{score_text}"
        cv2.putText(
            output,
            text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def draw_status(frame: np.ndarray, status: str) -> np.ndarray:
    output = frame.copy()
    cv2.putText(
        output,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
