"""Structured telemetry helpers for instrumenting search workflows."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from .observability import get_logger

TelemetryListener = Callable[[str, Dict[str, Any]], None]


class StructuredTelemetry:
    """Lightweight collector for timing, counters, and metadata."""

    def __init__(
        self,
        time_fn: Optional[Callable[[], float]] = None,
        *,
        max_events: int = 256,
        listeners: Optional[Iterable[TelemetryListener]] = None,
    ) -> None:
        self._time_fn = time_fn or time.perf_counter
        self._max_events = max(1, int(max_events))
        self._lock = threading.RLock()
        self._trace_id = 0
        self._latest_snapshot: Dict[str, Any] = {}
        self._listeners: list[TelemetryListener] = list(listeners or [])
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

    def _build_snapshot_locked(self) -> Dict[str, Any]:
        return {
            "trace_id": self._trace_id,
            "name": self._trace_name,
            "timings": {key: dict(value) for key, value in self._timings.items()},
            "counters": dict(self._counters),
            "events": [dict(event) for event in self._events],
            "metadata": dict(self._metadata),
        }

    def _update_latest_snapshot_locked(self) -> None:
        self._latest_snapshot = deepcopy(self._build_snapshot_locked())

    def _listeners_snapshot(self) -> Tuple[TelemetryListener, ...]:
        with self._lock:
            return tuple(self._listeners)

    def _notify_listeners(self, event_type: str, payload: Dict[str, Any]) -> None:
        for listener in self._listeners_snapshot():
            try:
                listener(event_type, dict(payload))
            except Exception:
                continue

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
            self._update_latest_snapshot_locked()
            trace_id = self._trace_id

        self._notify_listeners(
            "trace_started",
            {"trace_id": trace_id, "name": name},
        )
        return trace_id

    def _record_timing(
        self,
        name: str,
        duration: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        duration = max(0.0, float(duration))
        metadata = dict(payload) if payload else {}
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
            if metadata:
                event["metadata"] = dict(metadata)
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

            self._update_latest_snapshot_locked()

        self._notify_listeners(
            "timing",
            {"name": name, "duration": duration, "metadata": metadata},
        )

    @contextmanager
    def timer(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Context manager that records a timing measurement for ``name``."""

        payload: Dict[str, Any] = dict(metadata) if metadata else {}
        start = self.now()
        self._notify_listeners(
            "timer_started",
            {"name": name, "metadata": dict(payload)},
        )
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

        value = float(amount)
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value
            current_value = self._counters[name]
            self._update_latest_snapshot_locked()

        self._notify_listeners(
            "counter",
            {"name": name, "delta": value, "value": current_value},
        )

    record_counter = increment

    def annotate(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata to the current trace."""

        with self._lock:
            self._metadata[key] = value
            self._update_latest_snapshot_locked()

        self._notify_listeners("metadata", {"key": key, "value": value})

    def snapshot(self) -> Dict[str, Any]:
        """Capture the current telemetry data for the active trace."""

        with self._lock:
            snapshot = self._build_snapshot_locked()
            self._latest_snapshot = deepcopy(snapshot)
            return snapshot

    def latest_snapshot(self) -> Dict[str, Any]:
        """Return the most recently recorded snapshot, if available."""

        with self._lock:
            if not self._latest_snapshot:
                return {}
            return deepcopy(self._latest_snapshot)

    def add_listener(self, listener: TelemetryListener) -> None:
        """Register ``listener`` to receive telemetry events."""

        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: TelemetryListener) -> None:
        """Remove a telemetry listener when it is no longer needed."""

        with self._lock:
            self._listeners = [entry for entry in self._listeners if entry is not listener]


class TelemetryLogger:
    """Listener that emits telemetry activity to the project logger."""

    def __init__(
        self,
        *,
        logger: Optional[logging.LoggerAdapter] = None,
        level: int = logging.INFO,
        level_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self._logger = logger or get_logger(__name__).bind(component="telemetry")
        self._default_level = level
        self._level_map = dict(level_map or {})

    def __call__(self, event_type: str, payload: Dict[str, Any]) -> None:
        level = self._level_map.get(event_type, self._default_level)
        if not self._logger.isEnabledFor(level):  # type: ignore[attr-defined]
            return

        context = {"telemetry.event": event_type}
        context.update({str(key): value for key, value in payload.items()})

        name = (
            payload.get("name")
            or payload.get("key")
            or payload.get("trace_id")
            or "event"
        )
        message = f"Telemetry {event_type}: {name}"

        try:
            self._logger.log(level, message, context=context)
        except Exception:
            # Logging must never interrupt search execution.
            return


__all__ = ["StructuredTelemetry", "TelemetryLogger", "TelemetryListener"]
