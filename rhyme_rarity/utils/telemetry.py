"""Structured telemetry helpers for instrumenting search workflows."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, Optional


class StructuredTelemetry:
    """Lightweight collector for timing, counters, and metadata."""

    def __init__(
        self,
        time_fn: Optional[Callable[[], float]] = None,
        *,
        max_events: int = 256,
    ) -> None:
        self._time_fn = time_fn or time.perf_counter
        self._max_events = max(1, int(max_events))
        self._lock = threading.RLock()
        self._trace_id = 0
        self._latest_snapshot: Dict[str, Any] = {}
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self._timings: Dict[str, Dict[str, float]] = {}
        self._counters: Dict[str, float] = {}
        self._events: list[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
        self._trace_name: Optional[str] = None

    def now(self) -> float:
        """Return the current monotonic time used for telemetry measurements."""

        return float(self._time_fn())

    def start_trace(self, name: str) -> int:
        """Reset telemetry state and start a new trace."""

        with self._lock:
            self._trace_id += 1
            self._reset_state()
            self._trace_name = name
            self._metadata["trace_name"] = name
            self._metadata["start_time"] = self.now()
            return self._trace_id

    def _record_timing(
        self,
        name: str,
        duration: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        duration = max(0.0, float(duration))
        with self._lock:
            bucket = self._timings.get(name)
            if bucket is None:
                bucket = {
                    "count": 0,
                    "total": 0.0,
                    "min": duration,
                    "max": duration,
                }
                self._timings[name] = bucket

            bucket["count"] += 1
            bucket["total"] += duration
            bucket["min"] = duration if bucket["count"] == 1 else min(bucket["min"], duration)
            bucket["max"] = max(bucket["max"], duration)
            bucket["avg"] = bucket["total"] / bucket["count"] if bucket["count"] else 0.0

            event: Dict[str, Any] = {"name": name, "duration": duration}
            if payload:
                event["metadata"] = dict(payload)
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

    @contextmanager
    def timer(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Context manager that records a timing measurement for ``name``."""

        payload: Dict[str, Any] = dict(metadata) if metadata else {}
        start = self.now()
        try:
            yield payload
        finally:
            self._record_timing(name, self.now() - start, payload)

    def record_timing(
        self,
        name: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an explicit timing measurement with optional metadata."""

        self._record_timing(name, duration, metadata)

    def increment(self, name: str, amount: float = 1.0) -> None:
        """Increment a counter by ``amount`` (defaults to one)."""

        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + float(amount)

    record_counter = increment

    def annotate(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata to the current trace."""

        with self._lock:
            self._metadata[key] = value

    def snapshot(self) -> Dict[str, Any]:
        """Capture the current telemetry data for the active trace."""

        with self._lock:
            snapshot = {
                "trace_id": self._trace_id,
                "name": self._trace_name,
                "timings": {key: dict(value) for key, value in self._timings.items()},
                "counters": dict(self._counters),
                "events": [dict(event) for event in self._events],
                "metadata": dict(self._metadata),
            }
            self._latest_snapshot = deepcopy(snapshot)
            return snapshot

    def latest_snapshot(self) -> Dict[str, Any]:
        """Return the most recently recorded snapshot, if available."""

        with self._lock:
            if not self._latest_snapshot:
                return {}
            return deepcopy(self._latest_snapshot)


__all__ = ["StructuredTelemetry"]
