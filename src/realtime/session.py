from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import dataclass
from typing import TypedDict, cast

from loguru import logger

from src.config import AudioConfig, RealtimeConfig

from .audio_io import AudioIO
from .client import RealtimeClient


class _AudioDeltaEvent(TypedDict, total=False):
    type: str
    delta: str


class _TranscriptDeltaEvent(TypedDict, total=False):
    type: str
    item_id: str
    delta: str


class _ContentPart(TypedDict, total=False):
    transcript: str


class _ConversationItem(TypedDict, total=False):
    role: str
    content: list[_ContentPart]


class _ItemAddedEvent(TypedDict, total=False):
    type: str
    item: _ConversationItem


class _Event(TypedDict, total=False):
    type: str


@dataclass
class RealtimeSessionRunner:
    realtime_config: RealtimeConfig
    audio_config: AudioConfig

    def __post_init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._error: Exception | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: RealtimeClient | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Exception | None:
        return self._error

    def start(self, person_name: str, person_prompt: str) -> None:
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

    def stop(self, wait: bool = False) -> None:
        self._stop_event.set()
        loop = self._loop
        client = self._client
        if loop is not None and loop.is_running() and client is not None:
            try:
                asyncio.run_coroutine_threadsafe(client.close(), loop)
            except Exception:
                pass
        if not wait:
            return
        thread = self._thread
        if thread is not None:
            thread.join()
            if self._thread is thread and not thread.is_alive():
                self._thread = None

    def _run_thread(self, person_name: str, person_prompt: str) -> None:
        try:
            asyncio.run(self._run_session(person_name, person_prompt))
        except Exception as exc:
            self._error = exc
            logger.exception("Voice session failed for {}: {}", person_name, exc)
        finally:
            if self._thread is threading.current_thread():
                self._thread = None
            self._done_event.set()

    async def _run_session(self, person_name: str, person_prompt: str) -> None:
        client = RealtimeClient(self.realtime_config)
        self._loop = asyncio.get_running_loop()
        self._client = client
        audio = AudioIO(
            sample_rate_hz=self.audio_config.sample_rate_hz,
            channels=self.audio_config.channels,
            chunk_frames=self.audio_config.chunk_frames,
            speaker_chunk_frames=self.audio_config.speaker_chunk_frames,
            mic_queue_max_chunks=self.audio_config.mic_queue_max_chunks,
            speaker_queue_max_chunks=self.audio_config.speaker_queue_max_chunks,
        )
        await client.connect()
        instructions = person_prompt.strip().replace("{person_name}", person_name)
        logger.info("System prompt sent for {}:\n{}", person_name, instructions)
        await client.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.realtime_config.model,
                    "instructions": instructions,
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.audio_config.sample_rate_hz,
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": self.realtime_config.vad_threshold,
                                "silence_duration_ms": self.realtime_config.vad_silence_duration_ms,
                                "prefix_padding_ms": self.realtime_config.vad_prefix_padding_ms,
                                "interrupt_response": True,
                            },
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.audio_config.sample_rate_hz,
                            },
                            "voice": self.realtime_config.voice,
                            "speed": self.realtime_config.speed,
                        },
                    },
                },
            }
        )
        await self._wait_for_session_updated(client)
        audio.start()
        sender_task = asyncio.create_task(self._send_mic_audio(client, audio))
        receiver_task = asyncio.create_task(self._recv_audio(client, audio))
        await client.send_json(
            {
                "type": "response.create",
                "response": {},
            }
        )
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.05)
        finally:
            sender_task.cancel()
            receiver_task.cancel()
            await client.close()
            await asyncio.gather(sender_task, return_exceptions=True)
            audio.stop()
            self._client = None
            self._loop = None

    async def _wait_for_session_updated(self, client: RealtimeClient, timeout_s: float = 5.0) -> None:
        async def _wait() -> None:
            while True:
                event = cast(_Event, await client.recv_json())
                event_type = event.get("type")
                if event_type == "session.updated":
                    return
                if event_type == "error":
                    raise RuntimeError(f"Realtime error before session.updated: {event}")

        await asyncio.wait_for(_wait(), timeout=timeout_s)

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
        output_transcripts: dict[str, str] = {}
        block_bytes = self.audio_config.speaker_chunk_frames * self.audio_config.channels * 2
        speaker_buf = bytearray()
        stop_wait_task = asyncio.create_task(asyncio.to_thread(self._stop_event.wait))
        try:
            while True:
                recv_task = asyncio.create_task(client.recv_json())
                done, _ = await asyncio.wait(
                    {recv_task, stop_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_wait_task in done:
                    recv_task.cancel()
                    await asyncio.gather(recv_task, return_exceptions=True)
                    break
                event = recv_task.result()
                event_type = event.get("type")
                if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                    e = cast(_AudioDeltaEvent, event)
                    delta = e.get("delta")
                    if delta:
                        speaker_buf.extend(base64.b64decode(delta))
                        while len(speaker_buf) >= block_bytes:
                            audio.queue_speaker_audio(bytes(speaker_buf[:block_bytes]))
                            del speaker_buf[:block_bytes]
                elif event_type == "response.output_audio_transcript.delta":
                    e = cast(_TranscriptDeltaEvent, event)
                    item_id = e.get("item_id") or ""
                    output_transcripts[item_id] = output_transcripts.get(item_id, "") + (e.get("delta") or "")
                elif event_type == "response.done":
                    for text in output_transcripts.values():
                        if text.strip():
                            asyncio.create_task(asyncio.to_thread(logger.info, "Response: {}", text.strip()))
                    output_transcripts.clear()
                elif event_type == "conversation.item.added":
                    e = cast(_ItemAddedEvent, event)
                    item: _ConversationItem = e.get("item") or {}
                    role = item.get("role")
                    for part in item.get("content") or []:
                        transcript = part.get("transcript")
                        if not transcript:
                            continue
                        if role == "user":
                            asyncio.create_task(asyncio.to_thread(logger.info, "User said: {}", transcript.strip()))
                        elif role == "assistant":
                            asyncio.create_task(asyncio.to_thread(logger.info, "Response: {}", transcript.strip()))
        finally:
            stop_wait_task.cancel()
            await asyncio.gather(stop_wait_task, return_exceptions=True)
