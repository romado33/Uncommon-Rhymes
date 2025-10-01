# -*- coding: utf-8 -*-
import os, pytest
from src.reverse import rhyme_from_phrase


def test_phrase_to_rhymes_him_so(tmp_path):
    # Use default demo DB; if patterns.db not present, skip gracefully.
    db = "patterns.db"
    if not os.path.exists(db):
        pytest.skip("patterns.db not present in repo checkout")

    singles, phrases = rhyme_from_phrase("him so", db_path=db, slant_strength=0.6, allow_propers=True, limit=100)
    assert isinstance(singles, list) and isinstance(phrases, list)
    # At least one single-word hit
    assert len(singles) > 0
    # At least one multi-word hit retrieved from DB
    assert len(phrases) > 0


def test_phrase_gate_loose_allows_assonance(tmp_path):
    db = "patterns.db"
    if not os.path.exists(db):
        pytest.skip("patterns.db not present")
    singles_tight, phrases_tight = rhyme_from_phrase("him so", db_path=db, slant_strength=0.2, allow_propers=True, limit=50)
    singles_loose, phrases_loose = rhyme_from_phrase("him so", db_path=db, slant_strength=0.9, allow_propers=True, limit=50)
    # Loose should return >= tight
    assert len(singles_loose) >= len(singles_tight)
    assert len(phrases_loose) >= len(phrases_tight)
