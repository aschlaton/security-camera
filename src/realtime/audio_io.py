from __future__ import annotations

import queue

import sounddevice as sd


class AudioIO:
    def __init__(
        self,
        sample_rate_hz: int,
        channels: int,
        chunk_frames: int,
        speaker_chunk_frames: int,
        mic_queue_max_chunks: int,
        speaker_queue_max_chunks: int,
    ) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._chunk_frames = chunk_frames
        self._speaker_chunk_frames = speaker_chunk_frames
        self._mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=mic_queue_max_chunks)
        self._speaker_queue: queue.Queue[bytes] = queue.Queue(maxsize=speaker_queue_max_chunks)
        self._speaker_pending = bytearray()
        self._mic_stream: sd.RawInputStream | None = None
        self._speaker_stream: sd.RawOutputStream | None = None

    def start(self) -> None:
        self._mic_stream = sd.RawInputStream(
            samplerate=self._sample_rate_hz,
            channels=self._channels,
            dtype="int16",
            blocksize=self._chunk_frames,
            callback=self._on_mic_chunk,
        )
        self._speaker_stream = sd.RawOutputStream(
            samplerate=self._sample_rate_hz,
            channels=self._channels,
            dtype="int16",
            blocksize=self._speaker_chunk_frames,
            callback=self._on_speaker_chunk,
        )
        self._mic_stream.start()
        self._speaker_stream.start()

    def stop(self) -> None:
        if self._mic_stream is not None:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None
        if self._speaker_stream is not None:
            self._speaker_stream.stop()
            self._speaker_stream.close()
            self._speaker_stream = None
        self._clear_queues()

    def read_mic_chunk(self, timeout: float) -> bytes | None:
        try:
            return self._mic_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def queue_speaker_audio(self, pcm16_bytes: bytes) -> None:
        try:
            self._speaker_queue.put_nowait(pcm16_bytes)
        except queue.Full:
            pass

    def clear_speaker(self) -> None:
        while not self._speaker_queue.empty():
            try:
                self._speaker_queue.get_nowait()
            except queue.Empty:
                break
        self._speaker_pending.clear()

    def _on_mic_chunk(self, indata: bytes, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        _ = frames, time_info, status
        try:
            self._mic_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _on_speaker_chunk(self, outdata: bytearray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        _ = time_info, status
        expected = frames * self._channels * 2
        while len(self._speaker_pending) < expected:
            try:
                self._speaker_pending.extend(self._speaker_queue.get_nowait())
            except queue.Empty:
                break
        if len(self._speaker_pending) >= expected:
            outdata[:] = self._speaker_pending[:expected]
            del self._speaker_pending[:expected]
            return
        outdata[: len(self._speaker_pending)] = self._speaker_pending
        outdata[len(self._speaker_pending) :] = b"\x00" * (expected - len(self._speaker_pending))
        self._speaker_pending.clear()

    def _clear_queues(self) -> None:
        while not self._mic_queue.empty():
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                break
        while not self._speaker_queue.empty():
            try:
                self._speaker_queue.get_nowait()
            except queue.Empty:
                break
