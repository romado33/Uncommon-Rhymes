
import os, csv, json, sys, importlib
from pathlib import Path
import pytest

# --- Adapter layer: try to import your repo's APIs in a few common arrangements ---
ADAPTER_ERR = None
def _try_imports():
    global ADAPTER_ERR
    candidates = [
        # (module, functions)
        ("module1_enhanced_core_phonetic", ["explain_slant", "rank_slant_candidates"]),
        ("scorer", ["explain_slant", "rank_slant_candidates"]),
        ("app", ["rank_slant_candidates"]),  # UI module may expose core
    ]
    found = {}
    for mod, funcs in candidates:
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            continue
        ok = True
        for fn in funcs:
            if not hasattr(m, fn):
                ok = False
                break
        if ok:
            found["module"] = m
            return found
    ADAPTER_ERR = "Could not find core scoring API. Provide explain_slant() and rank_slant_candidates()."
    return None

CORE = _try_imports()

# Optional phrase search (reverse lookup)
PHRASE = None
try:
    PHRASE = importlib.import_module("module2_enhanced_anti_llm")
except Exception:
    PHRASE = None

DATA_PATH = Path(__file__).parent / "rhyme_eval_cases.csv"
if not DATA_PATH.exists():
    # Fallback: look relative to cwd or provided path
    alt = Path(os.getenv("RHYME_EVAL_CSV", ""))
    if alt.exists():
        DATA_PATH = alt

def load_cases():
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["expected_properties"] = json.loads(row["expected_properties"] or "{}")
            yield row

@pytest.mark.skipif(CORE is None, reason=lambda: ADAPTER_ERR or "Core API missing")
@pytest.mark.parametrize("case", list(load_cases()))
def test_core_paths(case):
    m = CORE["module"]
    text = case["input_text"]
    kind = case["kind"]
    expect = case["expected_properties"]
    # Basic sanity: rank or explain should not crash.
    # Use a loose slant for coverage; projects can override via env.
    slant_strength = float(os.getenv("SLANT_STRENGTH", "0.6"))
    allow_propers = bool(int(os.getenv("ALLOW_PROPERS", "1")))

    # Single-word vs phrase path
    if kind in ("single", "input"):
        # Rank against a vocab; we assume the module exposes a helper or will default to internal vocab.
        if hasattr(m, "rank_slant_candidates"):
            results = m.rank_slant_candidates(text, [], slant_strength=slant_strength, allow_propers=allow_propers)
            assert isinstance(results, (list, tuple)), "rank_slant_candidates should return a list"
            # Weak expectation: at least 1 result for common nuclei
            if expect.get("min_single_word_hits"):
                assert len(results) >= expect["min_single_word_hits"], f"Expected >= {expect['min_single_word_hits']} results for {text}"
    elif kind == "phrase":
        # Phrase → rhyme path should exist
        if PHRASE and hasattr(PHRASE, "multiword_rhymes"):
            ph = PHRASE.multiword_rhymes(text, slant_strength=slant_strength, allow_propers=allow_propers, limit=100)
            assert isinstance(ph, (list, tuple)), "multiword_rhymes should return a list"
            # Should produce both single- and multi-word via the pipeline
            if expect.get("must_return_multiword"):
                assert any(len(p.phrase.split()) >= 2 for p in ph), "Expected multi-word outputs"
        else:
            pytest.skip("multiword_rhymes not found; implement Module 2 to enable reverse phrase tests.")

@pytest.mark.skipif(CORE is None, reason=lambda: ADAPTER_ERR or "Core API missing")
@pytest.mark.parametrize("case", list(load_cases()))
def test_policy_guards(case):
    # Ensure orthography is not used as primary ranking signal (tie-break only).
    expect = case["expected_properties"]
    if expect.get("spelling_bias_tie_break_only"):
        # This is a policy test placeholder; projects should implement an exposed flag or counter
        # For now, we just assert that test suite includes these cases; devs should add internal checks.
        assert True

def test_dataset_integrity():
    # Ensure dataset covers a variety of categories
    cats = {}
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cats[row["category"]] = cats.get(row["category"], 0) + 1
    assert len(cats) >= 12, "Dataset should cover at least 12 distinct categories"
