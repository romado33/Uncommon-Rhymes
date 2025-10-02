"""Helpers for configuring consistent project logging output."""

from __future__ import annotations

import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_CONFIGURED = False


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    try:
        return int(level)
    except (TypeError, ValueError):
        normalized = str(level).strip().upper()
        return getattr(logging, normalized, logging.INFO)


def configure_logging(level: Optional[str | int] = None, *, force: bool = False) -> None:
    """Initialise root logging handlers for the application.

    Hugging Face Spaces captures stdout/stderr for the runtime logs. By configuring
    a basic handler that emits structured messages at ``INFO`` by default we ensure
    long-running rhyme searches surface their internal status without requiring the
    UI to expose telemetry directly.
    """

    global _CONFIGURED

    if _CONFIGURED and not force:
        return

    env_level = os.environ.get("RHYMES_LOG_LEVEL")
    resolved_level = _resolve_level(level if level is not None else env_level)

    logging.basicConfig(level=resolved_level, format=_DEFAULT_FORMAT)
    logging.getLogger("rhyme_rarity").setLevel(resolved_level)
    _CONFIGURED = True


__all__ = ["configure_logging"]
