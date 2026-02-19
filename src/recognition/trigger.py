from __future__ import annotations

import time


class RecognitionTrigger:
    def __init__(
        self,
        initial_similarity_threshold: float,
        initial_required_frames: int,
        session_similarity_threshold: float,
        session_missed_frames: int,
        cooldown_seconds: float,
    ) -> None:
        self._initial_similarity_threshold = initial_similarity_threshold
        self._initial_required_frames = initial_required_frames
        self._session_similarity_threshold = session_similarity_threshold
        self._session_missed_frames = session_missed_frames
        self._cooldown_seconds = cooldown_seconds
        self._idle_name: str | None = None
        self._idle_count = 0
        self._session_miss_count = 0
        self._cooldown_until: dict[str, float] = {}

    def update_idle(self, matched_name: str | None, similarity: float | None) -> str | None:
        now = time.monotonic()
        if (
            matched_name is None
            or similarity is None
            or similarity < self._initial_similarity_threshold
        ):
            self._idle_name = None
            self._idle_count = 0
            return None

        if now < self._cooldown_until.get(matched_name, 0.0):
            self._idle_name = None
            self._idle_count = 0
            return None

        if self._idle_name == matched_name:
            self._idle_count += 1
        else:
            self._idle_name = matched_name
            self._idle_count = 1

        if self._idle_count >= self._initial_required_frames:
            self._idle_name = None
            self._idle_count = 0
            return matched_name
        return None

    def reset_session_tracking(self) -> None:
        self._session_miss_count = 0

    def update_session(
        self,
        active_person: str | None,
        matched_name: str | None,
        similarity: float | None,
    ) -> bool:
        if active_person is None:
            self._session_miss_count = 0
            return False

        is_confident_active = (
            matched_name == active_person
            and similarity is not None
            and similarity >= self._session_similarity_threshold
        )
        if is_confident_active:
            self._session_miss_count = 0
            return False

        self._session_miss_count += 1
        return self._session_miss_count > self._session_missed_frames

    def start_cooldown(self, name: str) -> None:
        self._cooldown_until[name] = time.monotonic() + self._cooldown_seconds
