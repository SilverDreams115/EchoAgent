from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Callable

from echo.types import ActivityEvent


class ActivityBus:
    def __init__(self, max_events: int = 200) -> None:
        self._events: deque[ActivityEvent] = deque(maxlen=max_events)
        self._watchers: list[Callable[[ActivityEvent], None]] = []
        self._lock = Lock()

    def emit(self, stage: str, status: str, message: str, detail: str = "") -> ActivityEvent:
        event = ActivityEvent(stage=stage, status=status, message=message, detail=detail)
        with self._lock:
            self._events.append(event)
            watchers = list(self._watchers)
        for watcher in watchers:
            watcher(event)
        return event

    def recent(self, limit: int = 8) -> list[ActivityEvent]:
        with self._lock:
            return list(self._events)[-limit:]

    def watch(self, callback: Callable[[ActivityEvent], None]) -> None:
        with self._lock:
            self._watchers.append(callback)
