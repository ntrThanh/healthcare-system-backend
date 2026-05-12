from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple


STOP_WORDS = {
    "tat", "tắt", "dung", "dừng", "ngat", "ngắt",
    "huy", "hủy", "stop", "cancel", "abort", "pause",
}


def is_cancel_message(message: str | None) -> bool:
    if not message:
        return False
    text = message.strip().lower()
    text = text.replace(".", "").replace("!", "").replace("?", "")
    return text in STOP_WORDS


@dataclass
class GenerationHandle:
    user_id: str
    session_id: str
    event: asyncio.Event


class CancellationManager:
    """Tracks currently running chat generations so they can be stopped."""

    def __init__(self) -> None:
        self._active: Dict[Tuple[str, str], asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def start(self, user_id: str, session_id: str) -> asyncio.Event:
        key = (user_id, session_id)
        async with self._lock:
            old_event = self._active.get(key)
            if old_event:
                old_event.set()

            event = asyncio.Event()
            self._active[key] = event
            return event

    async def cancel(self, user_id: str, session_id: str | None = None) -> int:
        async with self._lock:
            matched = []
            for key, event in self._active.items():
                key_user_id, key_session_id = key
                if key_user_id == user_id and (session_id is None or key_session_id == session_id):
                    event.set()
                    matched.append(key)
            return len(matched)

    async def finish(self, user_id: str, session_id: str, event: asyncio.Event) -> None:
        key = (user_id, session_id)
        async with self._lock:
            if self._active.get(key) is event:
                self._active.pop(key, None)


cancellation_manager = CancellationManager()
