from __future__ import annotations

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class CameraStream:
    def __init__(self, index: int, width: int, height: int) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Could not open camera index {self._index}")
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to read frame from camera")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
