from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import dataclass

from src.config import AudioConfig, RealtimeConfig

from .audio_io import AudioIO
from .client import RealtimeClient
from .prompts import build_instructions


@dataclass(slots=True)
class RealtimeSessionRunner:
    realtime_config: RealtimeConfig
    audio_config: AudioConfig

    def __post_init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._error: Exception | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Exception | None:
        return self._error

    def start(self, person_name: str, person_prompt: str | None = None) -> None:
        if self.is_running:
            raise RuntimeError("Realtime session is already running")
        self._stop_event.clear()
        self._done_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run_thread,
            args=(person_name, person_prompt),
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout_seconds: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_seconds)
            self._thread = None

    def _run_thread(self, person_name: str, person_prompt: str | None) -> None:
        try:
            asyncio.run(self._run_session(person_name, person_prompt))
        except Exception as exc:
            self._error = exc
        finally:
            self._done_event.set()

    async def _run_session(self, person_name: str, person_prompt: str | None) -> None:
        client = RealtimeClient(self.realtime_config)
        audio = AudioIO(
            sample_rate_hz=self.audio_config.sample_rate_hz,
            channels=self.audio_config.channels,
            chunk_frames=self.audio_config.chunk_frames,
            mic_queue_max_chunks=self.audio_config.mic_queue_max_chunks,
            speaker_queue_max_chunks=self.audio_config.speaker_queue_max_chunks,
        )
        await client.connect()
        await client.send_json(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": build_instructions(person_name, person_prompt),
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad"},
                },
            }
        )
        await client.send_json(
            {
                "type": "response.create",
                "response": {
                    "instructions": f"Start with a short greeting to {person_name}.",
                },
            }
        )
        audio.start()
        sender_task = asyncio.create_task(self._send_mic_audio(client, audio))
        receiver_task = asyncio.create_task(self._recv_audio(client, audio))
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.05)
        finally:
            sender_task.cancel()
            receiver_task.cancel()
            await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
            audio.stop()
            await client.close()

    async def _send_mic_audio(self, client: RealtimeClient, audio: AudioIO) -> None:
        while not self._stop_event.is_set():
            chunk = await asyncio.to_thread(audio.read_mic_chunk, 0.1)
            if not chunk:
                continue
            await client.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )

    async def _recv_audio(self, client: RealtimeClient, audio: AudioIO) -> None:
        while not self._stop_event.is_set():
            event = await client.recv_json()
            event_type = event.get("type")
            if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                delta = event.get("delta")
                if delta:
                    audio.queue_speaker_audio(base64.b64decode(delta))
