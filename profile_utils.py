"""Utility helpers for working with analyzer profile objects."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict


def normalize_profile_dict(profile_obj: Any) -> Dict[str, Any]:
    """Return a dictionary representation of a profile-like object.

    Many analyzer APIs return custom profile objects that may implement an
    ``as_dict`` method, behave like a mapping, or simply expose attributes on
    ``__dict__``.  This helper centralizes the fallbacks used across modules so
    that we consistently produce a ``dict`` without duplicating conversion
    logic.
    """

    if profile_obj is None:
        return {}

    if hasattr(profile_obj, "as_dict"):
        try:
            converted = profile_obj.as_dict()
            if converted is not None:
                return dict(converted)
        except Exception:
            pass

    if isinstance(profile_obj, dict):
        return dict(profile_obj)

    if isinstance(profile_obj, Mapping):
        return dict(profile_obj)

    for attr in ("__dict__", "_asdict"):
        if hasattr(profile_obj, attr):
            try:
                candidate = getattr(profile_obj, attr)
                candidate = candidate() if callable(candidate) else candidate
                if candidate is not None:
                    return dict(candidate)
            except Exception:
                pass

    try:
        return dict(vars(profile_obj))
    except Exception:
        pass

    try:
        return dict(profile_obj)
    except Exception:
        return {}
