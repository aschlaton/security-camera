from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    window_name: str = "Security Camera"
    recognition_every_n_frames: int = 3


@dataclass(slots=True)
class RecognitionConfig:
    people_dir: Path = Path("people")
    cache_path: Path = Path(".cache/face_db.npz")
    similarity_threshold: float = 0.45
    required_consecutive_matches: int = 3
    cooldown_seconds: float = 20.0
    person_absent_timeout_seconds: float = 6.0


@dataclass(slots=True)
class AudioConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    chunk_frames: int = 1024
    mic_queue_max_chunks: int = 64
    speaker_queue_max_chunks: int = 256


@dataclass(slots=True)
class RealtimeConfig:
    model: str = "gpt-realtime"
    openai_api_key: str | None = None


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig
    recognition: RecognitionConfig
    audio: AudioConfig
    realtime: RealtimeConfig


def load_config() -> AppConfig:
    api_key = os.environ.get("OPENAI_API_KEY")
    return AppConfig(
        camera=CameraConfig(),
        recognition=RecognitionConfig(),
        audio=AudioConfig(),
        realtime=RealtimeConfig(openai_api_key=api_key),
    )
