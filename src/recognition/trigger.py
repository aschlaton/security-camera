from __future__ import annotations

import time


class RecognitionTrigger:
    def __init__(self, required_consecutive_matches: int, cooldown_seconds: float) -> None:
        self._required = required_consecutive_matches
        self._cooldown_seconds = cooldown_seconds
        self._current_name: str | None = None
        self._count = 0
        self._cooldown_until: dict[str, float] = {}

    def update(self, matched_name: str | None) -> str | None:
        now = time.monotonic()
        if matched_name is None:
            self._current_name = None
            self._count = 0
            return None

        if now < self._cooldown_until.get(matched_name, 0.0):
            self._current_name = None
            self._count = 0
            return None

        if self._current_name == matched_name:
            self._count += 1
        else:
            self._current_name = matched_name
            self._count = 1

        if self._count >= self._required:
            self._current_name = None
            self._count = 0
            return matched_name
        return None

    def start_cooldown(self, name: str) -> None:
        self._cooldown_until[name] = time.monotonic() + self._cooldown_seconds
