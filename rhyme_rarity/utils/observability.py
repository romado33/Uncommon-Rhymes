"""Lightweight observability helpers used across the project.

This module intentionally keeps optional dependencies soft so the broader
application can run without Prometheus or OpenTelemetry being installed.  The
helpers exposed here provide a common interface that gracefully degrades to
no-ops when optional instrumentation libraries are unavailable.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional


try:  # pragma: no cover - optional dependency probing
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Histogram as _PromHistogram
except Exception:  # pragma: no cover - Prometheus not installed
    _PromCounter = None
    _PromHistogram = None

try:  # pragma: no cover - optional dependency probing
    from opentelemetry import trace as _otel_trace
except Exception:  # pragma: no cover - OpenTelemetry not installed
    _otel_trace = None


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Simple adapter that renders structured context inline with messages."""

    def bind(self, **context: Any) -> "StructuredLoggerAdapter":
        merged = dict(self.extra)
        merged.update(context)
        return StructuredLoggerAdapter(self.logger, merged)

    def process(self, msg: str, kwargs: Dict[str, Any]):
        event_context: Dict[str, Any] = dict(self.extra)
        provided = kwargs.pop("context", None)
        if isinstance(provided, dict):
            event_context.update(provided)
        if event_context:
            try:
                payload = json.dumps(event_context, sort_keys=True, default=str)
            except TypeError:
                payload = json.dumps({k: str(v) for k, v in event_context.items()})
            msg = f"{msg} | {payload}"
        return msg, kwargs


def get_logger(name: str, **context: Any) -> StructuredLoggerAdapter:
    """Return a project logger with optional bound context."""

    base_logger = logging.getLogger(name)
    return StructuredLoggerAdapter(base_logger, context)


class _MetricWrapper:
    """Base wrapper providing ``labels`` passthrough for metrics."""

    def __init__(self, impl: Any = None) -> None:
        self._impl = impl

    def labels(self, **labels: Any):  # type: ignore[override]
        impl = getattr(self._impl, "labels", None)
        if impl is None:
            return self.__class__(None)
        try:
            return self.__class__(impl(**labels))
        except Exception:  # pragma: no cover - defensive guard
            return self.__class__(None)


class CounterHandle(_MetricWrapper):
    """Graceful wrapper around Prometheus counters."""

    def inc(self, amount: float = 1.0) -> None:
        if self._impl is None:
            return
        try:
            self._impl.inc(amount)
        except Exception:  # pragma: no cover - defensive guard
            return


class HistogramHandle(_MetricWrapper):
    """Graceful wrapper around Prometheus histograms."""

    def observe(self, value: float) -> None:
        if self._impl is None:
            return
        try:
            self._impl.observe(value)
        except Exception:  # pragma: no cover - defensive guard
            return

    @contextmanager
    def time(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start)


def create_counter(
    name: str,
    documentation: str,
    label_names: Optional[Iterable[str]] = None,
) -> CounterHandle:
    """Create a counter that survives when Prometheus is not installed."""

    if _PromCounter is None:
        return CounterHandle()
    try:
        impl = _PromCounter(name, documentation, labelnames=tuple(label_names or ()))
    except ValueError:
        try:  # pragma: no cover - only executed when metric pre-exists
            from prometheus_client import REGISTRY  # type: ignore

            impl = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive guard
            impl = None
    except Exception:  # pragma: no cover - defensive guard
        impl = None
    return CounterHandle(impl)


def create_histogram(
    name: str,
    documentation: str,
    label_names: Optional[Iterable[str]] = None,
) -> HistogramHandle:
    """Create a histogram that gracefully no-ops when unavailable."""

    if _PromHistogram is None:
        return HistogramHandle()
    try:
        impl = _PromHistogram(name, documentation, labelnames=tuple(label_names or ()))
    except ValueError:
        try:  # pragma: no cover - only executed when metric pre-exists
            from prometheus_client import REGISTRY  # type: ignore

            impl = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive guard
            impl = None
    except Exception:  # pragma: no cover - defensive guard
        impl = None
    return HistogramHandle(impl)


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start an OpenTelemetry span if tracing is available."""

    if _otel_trace is None:
        yield None
        return

    tracer = _otel_trace.get_tracer("rhyme_rarity")
    with tracer.start_as_current_span(name) as span:  # type: ignore[attr-defined]
        if attributes:
            for key, value in attributes.items():
                if not isinstance(key, str):
                    continue
                try:
                    span.set_attribute(key, value)
                except Exception:  # pragma: no cover - defensive guard
                    continue
        yield span


def add_span_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    """Attach ``attributes`` to ``span`` if tracing is active."""

    if span is None:
        return
    for key, value in attributes.items():
        if not isinstance(key, str):
            continue
        try:
            span.set_attribute(key, value)
        except Exception:  # pragma: no cover - defensive guard
            continue


def record_exception(span: Any, error: BaseException) -> None:
    """Log an exception to an active span when tracing is available."""

    if span is None:
        return
    try:  # pragma: no cover - defensive guard
        span.record_exception(error)
        span.set_attribute("error", True)
    except Exception:
        return


__all__ = [
    "StructuredLoggerAdapter",
    "get_logger",
    "CounterHandle",
    "HistogramHandle",
    "create_counter",
    "create_histogram",
    "start_span",
    "add_span_attributes",
    "record_exception",
]

