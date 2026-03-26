from __future__ import annotations

from dataclasses import dataclass
import time

from echo.config import Settings


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass(slots=True)
class RuntimeBudget:
    total_ms: int
    deadline_ms: int
    reserve_ms: int

    def remaining_ms(self) -> int:
        return max(0, self.deadline_ms - monotonic_ms())

    def expired(self) -> bool:
        return self.remaining_ms() <= 0

    def request_timeout_seconds(self, request_cap_seconds: int) -> int:
        available_ms = max(1000, self.remaining_ms() - self.reserve_ms)
        return max(1, min(request_cap_seconds, available_ms // 1000))

    def allows_retry(self, *, min_retry_window_ms: int) -> bool:
        return self.remaining_ms() > self.reserve_ms + min_retry_window_ms


def build_runtime_budget(settings: Settings) -> RuntimeBudget:
    total_ms = max(1000, settings.backend_timeout * 1000)
    reserve_ms = max(4000, settings.backend_preflight_timeout * 1000)
    return RuntimeBudget(
        total_ms=total_ms,
        deadline_ms=monotonic_ms() + total_ms,
        reserve_ms=reserve_ms,
    )
