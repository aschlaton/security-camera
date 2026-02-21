from __future__ import annotations

from contextlib import suppress
from typing import Any

from openai import AsyncOpenAI

from src.config import RealtimeConfig


class RealtimeClient:
    def __init__(self, config: RealtimeConfig) -> None:
        self._config = config
        self._sdk: AsyncOpenAI | None = None
        self._conn_manager: Any | None = None
        self._conn: Any | None = None
        self._events: Any | None = None

    async def connect(self) -> None:
        if not self._config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for realtime session")
        self._sdk = AsyncOpenAI(api_key=self._config.openai_api_key)
        self._conn_manager = self._sdk.realtime.connect(model=self._config.model)
        self._conn = await self._conn_manager.__aenter__()
        self._events = self._conn.__aiter__()

    async def send_json(self, payload: dict) -> None:
        if self._conn is None:
            raise RuntimeError("Realtime connection is not open")
        await self._conn.send(payload)

    async def recv_json(self) -> dict[str, Any]:
        if self._events is None:
            raise RuntimeError("Realtime connection is not open")
        event = await self._events.__anext__()
        if isinstance(event, dict):
            return event
        if hasattr(event, "model_dump"):
            return event.model_dump(mode="json")
        if hasattr(event, "to_dict"):
            return event.to_dict()
        return {
            "type": getattr(event, "type", "unknown"),
            "raw": repr(event),
        }

    async def close(self) -> None:
        conn = self._conn
        conn_manager = self._conn_manager
        sdk = self._sdk
        self._conn = None
        self._events = None
        self._conn_manager = None
        self._sdk = None

        if conn is not None:
            with suppress(Exception):
                await conn.close()
        if conn_manager is not None:
            with suppress(Exception):
                await conn_manager.__aexit__(None, None, None)
        if sdk is not None:
            with suppress(Exception):
                await sdk.close()
