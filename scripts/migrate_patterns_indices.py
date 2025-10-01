# -*- coding: utf-8 -*-
"""
Adds and indexes:
  - phrases.last_word_rime_key
  - phrases.last_two_syllables_key
Also creates meta(schema_version, seed_hash) if missing.
Backfills keys using Module 1 phonetics.
"""
import sqlite3
from pathlib import Path
from typing import List, Tuple
import sys

try:
    from module1_enhanced_core_phonetic import get_pronunciations, extract_rime
except Exception:
    def get_pronunciations(w: str): return []
    def extract_rime(phones: List[str]): return ((), (), "none", 1)


def rime_label(v, c):
    j = lambda seg: ".".join(seg) if seg else ""
    vj, cj = j(v), j(c)
    if vj and cj: return f"/{vj}.{cj}/"
    if vj: return f"/{vj}/"
    return "/∅/"


def main(db_path="patterns.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # meta
    cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    # columns
    cur.execute("PRAGMA table_info(phrases)")
    cols = [r[1] for r in cur.fetchall()]
    if "last_word_rime_key" not in cols:
        cur.execute("ALTER TABLE phrases ADD COLUMN last_word_rime_key TEXT")
    if "last_two_syllables_key" not in cols:
        cur.execute("ALTER TABLE phrases ADD COLUMN last_two_syllables_key TEXT")
    # indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_last_word_rime_key ON phrases(last_word_rime_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_last_two_syllables_key ON phrases(last_two_syllables_key)")
    conn.commit()

    # backfill
    rows = cur.execute("SELECT rowid, phrase FROM phrases").fetchall()
    for rid, phrase in rows:
        toks = [t for t in phrase.lower().split() if t.isalpha()]
        if not toks: continue
        last = toks[-1]
        k1 = []
        for p in get_pronunciations(last) or []:
            v, c, s, span = extract_rime(p)
            k1.append(rime_label(v, c))
        k1 = k1[0] if k1 else None

        k2 = None
        if len(toks) >= 2:
            pen = toks[-2]
            pen_pr = get_pronunciations(pen) or []
            fin_pr = get_pronunciations(last) or []
            labels = []
            for a in pen_pr:
                v_pen, c_pen, *_ = extract_rime(a)
                for b in fin_pr:
                    v_fin, c_fin, *_ = extract_rime(b)
                    labels.append(rime_label(tuple(list(v_pen) + list(v_fin)), c_fin))
            if labels:
                k2 = labels[0]

        cur.execute("UPDATE phrases SET last_word_rime_key=?, last_two_syllables_key=? WHERE rowid=?",
                    (k1, k2, rid))
    conn.commit()
    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version','1')")
    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "patterns.db")
