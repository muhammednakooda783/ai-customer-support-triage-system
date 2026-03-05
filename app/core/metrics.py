from __future__ import annotations

from collections import Counter
from threading import Lock


class InMemoryMetrics:
    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()
        self._lock = Lock()

    def increment(self, key: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        with self._lock:
            self._counters[key] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)
