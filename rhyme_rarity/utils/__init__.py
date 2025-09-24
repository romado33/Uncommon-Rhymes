"""Utility helpers shared across the :mod:`rhyme_rarity` package."""

from __future__ import annotations

from .profile import normalize_profile_dict
from .syllables import estimate_syllable_count
from .telemetry import StructuredTelemetry
from .observability import (
    StructuredLoggerAdapter,
    add_span_attributes,
    create_counter,
    create_histogram,
    get_logger,
    record_exception,
    start_span,
)

__all__ = [
    "normalize_profile_dict",
    "estimate_syllable_count",
    "StructuredTelemetry",
    "StructuredLoggerAdapter",
    "add_span_attributes",
    "create_counter",
    "create_histogram",
    "get_logger",
    "record_exception",
    "start_span",
]
