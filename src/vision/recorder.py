from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np


class FrameRecorder:
    def __init__(self, max_frames: int = 300) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=max_frames)

    def push(self, frame: np.ndarray) -> None:
        self._buffer.append(frame.copy())

    def save(self, output_path: Path, fps: float = 20.0) -> None:
        if not self._buffer:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first = self._buffer[0]
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            for frame in self._buffer:
                writer.write(frame)
        finally:
            writer.release()
