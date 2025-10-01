# Code Review: `src/reverse.py`

## Overview
`src/reverse.py` stitches together pronunciation lookups, rhyme-key derivation, and database retrieval to surface both single-word and multi-word rhymes for a supplied phrase. The file also ships a substantial compatibility layer that recreates several APIs if `module1_enhanced_core_phonetic` is not importable.

## Potential Bugs / Reliability Risks

1. **`ensure_db_indices` assumes the `phrases` table already exists**  
   When the bundled database has not been initialised, calling `ensure_db_indices` raises `sqlite3.OperationalError` from the unconditional index creation statements (e.g. `CREATE INDEX ... ON phrases(last_word_rime_key)`). Because the function is invoked on every `retrieve_phrase_candidates` call, the exception aborts rhyme retrieval instead of being handled gracefully.【F:src/reverse.py†L295-L353】  
   *Recommendation:* guard the index creation block with `try/except sqlite3.OperationalError` or short-circuit when `PRAGMA table_info` returns no columns, so the caller can receive an empty result instead of a runtime failure.

2. **`retrieve_phrase_candidates` performs SQL string interpolation without validating `conds`**  
   If both key lists are empty, `where` becomes an empty string, yielding `WHERE ` followed by nothing and a syntax error at execution time. Although the caller currently returns early when both key lists are empty, the helper would be safer if it asserted `conds` before constructing the SQL fragment. This keeps the function robust if it gets reused elsewhere.【F:src/reverse.py†L322-L347】

3. **`rank_slant_candidates` silently falls back to default vocabulary when an empty list is passed**  
   The truthy check `if candidate_words:` treats an explicitly supplied empty list as "no candidate list provided". Downstream callers therefore cannot request "no candidates" or provide an iterable that happens to be empty without triggering the expensive default lookup. Propagating this ambiguity can surprise API users and make tests harder to reason about.【F:src/reverse.py†L185-L213】  
   *Recommendation:* distinguish `None` from an empty list (e.g., `if candidate_words is not None:`) so the function's behaviour is explicit.

## Refactoring Opportunities

* **Isolate the fallback compatibility layer.**  The module-level `try/except` currently mingles production logic with a sizable block of fallback utilities, making the file lengthy and harder to scan. Extracting the fallback implementation into a dedicated module (e.g. `reverse_fallback.py`) keeps `reverse.py` focused on the pipeline and simplifies unit testing of both code paths.【F:src/reverse.py†L20-L220】

* **Cache rime lookups for repeated queries.**  `_last_stressed_rime_labels_for_word` recomputes pronunciations on every call even though the results are deterministic per word. Decorating it with `@functools.lru_cache` (similar to `_rime_label`) would avoid redundant CMU lookups during phrase scoring, especially when phrases share penultimate or final words.【F:src/reverse.py†L257-L287】

* **Avoid re-running schema migration logic on every query.**  `ensure_db_indices` executes DDL statements on each request. Splitting schema migration into a one-time setup routine (invoked at startup) and using cheap guards inside `retrieve_phrase_candidates` would reduce connection churn and contention when the service scales.【F:src/reverse.py†L295-L353】

## Additional Suggestions

* Unit tests that simulate an empty / freshly created SQLite file would catch the `ensure_db_indices` failure path and document the expected behaviour for missing tables.
* Consider returning structured explanations (e.g. dataclasses) from both code paths so the calling code does not need to introspect tuples vs. objects when building `SingleWordResult` and `PhraseResult` instances.【F:src/reverse.py†L387-L406】

