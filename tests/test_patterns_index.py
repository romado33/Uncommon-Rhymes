# -*- coding: utf-8 -*-
import os, sqlite3, pytest
from src.reverse import ensure_db_indices


def test_patterns_indices():
    db = "patterns.db"
    if not os.path.exists(db):
        pytest.skip("patterns.db not present")
    conn = sqlite3.connect(db)
    ensure_db_indices(conn)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(phrases)")
    cols = [r[1] for r in cur.fetchall()]
    assert "last_word_rime_key" in cols
    assert "last_two_syllables_key" in cols
    # Index presence (sqlite master)
    idxs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()]
    assert any("idx_last_word_rime_key" in x for x in idxs)
    assert any("idx_last_two_syllables_key" in x for x in idxs)
    conn.close()
