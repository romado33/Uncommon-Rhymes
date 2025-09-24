"""Utility helpers for logging, metrics, and tracing.

This module centralises optional observability primitives so other packages
can instrument their behaviour without needing to guard every import.  All
helpers degrade gracefully when the optional dependencies are unavailable so
the core application can continue to function in constrained environments.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly in environments with the deps
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover - safety net for partial installations
    Counter = Gauge = Histogram = None  # type: ignore[assignment]


class _NoopMetric:
    """Fallback metric implementation used when prometheus_client is absent."""

    def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
        return self

    def inc(self, amount: float = 1.0) -> None:
        return None

    def observe(self, value: float) -> None:
        return None

    def set(self, value: float) -> None:
        return None


def _build_metric(factory: Optional[Any], *args: Any, **kwargs: Any) -> Any:
    if factory is None:
        return _NoopMetric()
    try:
        return factory(*args, **kwargs)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to construct metric", extra={"metric_args": args})
        return _NoopMetric()


SEARCH_REQUESTS_TOTAL = _build_metric(
    Counter,
    "search_requests_total",
    "Number of rhyme searches routed to each engine.",
    labelnames=("source", "filters"),
)

SEARCH_LATENCY_SECONDS = _build_metric(
    Histogram,
    "search_latency_seconds",
    "Latency distribution for rhyme searches by stage.",
    labelnames=("stage",),
)

SEARCH_ERRORS_TOTAL = _build_metric(
    Counter,
    "search_errors_total",
    "Count of handled errors while serving rhyme searches.",
    labelnames=("stage",),
)

CACHE_HIT_RATIO = _build_metric(
    Gauge,
    "cache_hit_ratio",
    "Rolling ratio of cache hits for rhyme search helpers.",
    labelnames=("cache",),
)

DATABASE_INITIALIZATION_TOTAL = _build_metric(
    Counter,
    "database_initialization_total",
    "Number of times the application prepared the patterns database.",
    labelnames=("mode",),
)

RARITY_REFRESH_FAILURES_TOTAL = _build_metric(
    Counter,
    "rarity_refresh_failures_total",
    "Count of failed rarity map refresh attempts.",
)


try:  # pragma: no cover - only executed when opentelemetry is available
    from opentelemetry import trace
except Exception:  # pragma: no cover - keep tracing optional
    trace = None  # type: ignore[assignment]


def _get_tracer() -> Optional[Any]:
    if trace is None:
        return None
    try:
        return trace.get_tracer("rhyme_rarity")
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to acquire OpenTelemetry tracer")
        return None


@contextmanager
def traced_span(name: str, **attributes: Any) -> Iterator[Optional[Any]]:
    """Context manager that starts an OpenTelemetry span when available."""

    tracer = _get_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as span:  # pragma: no cover - thin
        for key, value in attributes.items():
            try:
                span.set_attribute(key, value)
            except Exception:
                logger.debug("Failed to set span attribute", exc_info=True)
        yield span


@contextmanager
def observe_stage_latency(stage: str, **attributes: Any) -> Iterator[Optional[Any]]:
    """Record execution time for a stage and attach tracing metadata."""

    start = perf_counter()
    span_name = attributes.pop("span_name", stage)
    with traced_span(span_name, stage=stage, **attributes) as span:
        yield span

    duration = perf_counter() - start
    SEARCH_LATENCY_SECONDS.labels(stage=stage).observe(duration)
    if span is not None:
        try:
            span.set_attribute("duration_ms", duration * 1000.0)
        except Exception:
            logger.debug("Failed to annotate span duration", exc_info=True)


def record_search_request(source: str, filters_active: bool) -> None:
    label = "active" if filters_active else "none"
    SEARCH_REQUESTS_TOTAL.labels(source=source, filters=label).inc()


def record_search_error(stage: str, **metadata: Any) -> None:
    SEARCH_ERRORS_TOTAL.labels(stage=stage).inc()
    if metadata:
        logger.debug("search.error", extra={"stage": stage, **metadata})


def update_cache_hit_ratio(cache: str, hits: int, lookups: int) -> None:
    ratio = 0.0
    if lookups > 0:
        ratio = max(0.0, min(1.0, hits / float(lookups)))
    CACHE_HIT_RATIO.labels(cache=cache).set(ratio)


def record_database_initialization(mode: str, **metadata: Any) -> None:
    DATABASE_INITIALIZATION_TOTAL.labels(mode=mode).inc()
    if metadata:
        logger.debug("database.initialization", extra={"mode": mode, **metadata})


def record_rarity_refresh_failure(**metadata: Any) -> None:
    RARITY_REFRESH_FAILURES_TOTAL.inc()
    if metadata:
        logger.debug("rarity.refresh_failure", extra=metadata)


__all__ = [
    "CACHE_HIT_RATIO",
    "DATABASE_INITIALIZATION_TOTAL",
    "RARITY_REFRESH_FAILURES_TOTAL",
    "SEARCH_ERRORS_TOTAL",
    "SEARCH_LATENCY_SECONDS",
    "SEARCH_REQUESTS_TOTAL",
    "observe_stage_latency",
    "record_database_initialization",
    "record_rarity_refresh_failure",
    "record_search_error",
    "record_search_request",
    "traced_span",
    "update_cache_hit_ratio",
]

