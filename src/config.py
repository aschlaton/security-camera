from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REALTIME_DISABLED = False


@dataclass(slots=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    window_name: str = "Security Camera"
    recognition_every_n_frames: int = 8
    recognition_max_width: int = 640


@dataclass(slots=True)
class RecognitionConfig:
    people_dir: Path = Path("people")
    cache_path: Path = Path(".cache/face_db.npz")
    initial_similarity_threshold: float = 0.50
    initial_required_frames: int = 3
    session_similarity_threshold: float = 0.40
    session_missed_frames: int = 12
    cooldown_seconds: float = 20.0


@dataclass(slots=True)
class AudioConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    chunk_frames: int = 1024
    mic_queue_max_chunks: int = 64
    speaker_queue_max_chunks: int = 256


@dataclass(slots=True)
class RealtimeConfig:
    openai_api_key: str
    model: str = "gpt-realtime"
    disabled: bool = False
    vad_threshold: float = 0.4 # higher = more sensitive
    vad_silence_duration_ms: int = 700 # how long silence before turn ends
    vad_prefix_padding_ms: int = 300 # how much audio to keep before detected speech


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig
    recognition: RecognitionConfig
    audio: AudioConfig
    realtime: RealtimeConfig


def load_config() -> AppConfig:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not REALTIME_DISABLED and not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Put it in .env and load it before running."
        )
    return AppConfig(
        camera=CameraConfig(),
        recognition=RecognitionConfig(),
        audio=AudioConfig(),
        realtime=RealtimeConfig(openai_api_key=api_key or "", disabled=REALTIME_DISABLED),
    )
