from __future__ import annotations

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
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._events = None
        if self._conn_manager is not None:
            await self._conn_manager.__aexit__(None, None, None)
            self._conn_manager = None
        if self._sdk is not None:
            await self._sdk.close()
            self._sdk = None
