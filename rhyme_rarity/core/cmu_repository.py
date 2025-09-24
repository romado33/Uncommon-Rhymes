"""Repository abstraction for CMU-derived rhyme lookups."""

from __future__ import annotations

from typing import Any, List, Optional, Protocol

from .analyzer import EnhancedPhoneticAnalyzer, get_cmu_rhymes


class CmuRhymeRepository(Protocol):
    """Protocol describing the minimal CMU lookup interface used by services."""

    def lookup(
        self,
        source_word: str,
        limit: int,
        *,
        analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cmu_loader: Optional[Any] = None,
    ) -> List[Any]:
        """Return CMU candidates for ``source_word`` with the given ``limit``."""


class DefaultCmuRhymeRepository:
    """Repository implementation backed by :func:`get_cmu_rhymes`."""

    def lookup(
        self,
        source_word: str,
        limit: int,
        *,
        analyzer: Optional[EnhancedPhoneticAnalyzer] = None,
        cmu_loader: Optional[Any] = None,
    ) -> List[Any]:
        return get_cmu_rhymes(
            source_word,
            limit=limit,
            analyzer=analyzer,
            cmu_loader=cmu_loader,
        )


__all__ = ["CmuRhymeRepository", "DefaultCmuRhymeRepository"]

