"""Compatibility shim for the legacy Module 3 import path."""

from __future__ import annotations

from cultural.engine import CulturalIntelligenceEngine
from cultural.profiles import ArtistProfile, CulturalContext

__all__ = [
    "CulturalIntelligenceEngine",
    "ArtistProfile",
    "CulturalContext",
]
