"""Automatic merge conflict resolver for known SearchService and Anti-LLM overlaps."""

from __future__ import annotations

import pathlib
import re
from typing import Final


ROOT: Final = pathlib.Path(__file__).resolve().parent.parent


def _update_search_service() -> bool:
    """Ensure :mod:`search_service` contains the merged helper implementations."""

    target = ROOT / "rhyme_rarity" / "app" / "services" / "search_service.py"
    contents = target.read_text()
    original = contents

    contents = contents.replace("self.normalize_source_name", "self.normalize_filter_label")
    contents = contents.replace("_prepare_confidence_defaults", "_ensure_score_fields")

    filter_pattern = re.compile(
        r"    def normalize_(?:filter_label|source_name)\(self, name: Optional\[str\]\) -> str:\n"
        r"(?:        .*\n)+?(?=    def|\Z)",
        re.MULTILINE,
    )

    filter_block = (
        "    def normalize_filter_label(self, name: Optional[str]) -> str:\n"
        "        \"\"\"Normalise user-supplied filter labels by trimming, lowercasing, and replacing underscores for consistent comparisons.\"\"\"\n"
        "        if name is None:\n"
        "            return \"\"\n"
        "        return str(name).strip().lower().replace(\"_\", \"-\")\n\n"
    )

    contents = filter_pattern.sub(filter_block, contents)

    score_pattern = re.compile(
        r" {12}def _(?:ensure_score_fields|prepare_confidence_defaults)\(entry: Dict\) -> float:\n"
        r"(?: {12}.*\n)+?(?= {12}def| {8}#| {8}return|\Z)",
        re.MULTILINE,
    )

    score_block = (
        "            def _ensure_score_fields(entry: Dict) -> float:\n"
        "                \"\"\"Ensure an entry carries both `combined_score` and `confidence`, returning the value used for downstream filtering.\"\"\"\n"
        "                cache = entry.get(\"_confidence_cache\")\n"
        "                if cache is not None:\n"
        "                    cached_combined, cached_confidence, cached_score = cache\n"
        "                    if (\n"
        "                        entry.get(\"combined_score\") == cached_combined\n"
        "                        and entry.get(\"confidence\") == cached_confidence\n"
        "                    ):\n"
        "                        return cached_score\n\n"
        "                combined_value = _coerce_float(entry.get(\"combined_score\"))\n"
        "                confidence_value = _coerce_float(entry.get(\"confidence\"))\n\n"
        "                score_for_filter = (\n"
        "                    combined_value if combined_value is not None else confidence_value\n"
        "                )\n"
        "                if score_for_filter is None:\n"
        "                    score_for_filter = 0.0\n\n"
        "                entry[\"combined_score\"] = (\n"
        "                    combined_value if combined_value is not None else score_for_filter\n"
        "                )\n"
        "                entry[\"confidence\"] = (\n"
        "                    confidence_value if confidence_value is not None else score_for_filter\n"
        "                )\n\n"
        "                entry[\"_confidence_cache\"] = (\n"
        "                    entry.get(\"combined_score\"),\n"
        "                    entry.get(\"confidence\"),\n"
        "                    score_for_filter,\n"
        "                )\n\n"
        "                return score_for_filter\n\n"
    )

    contents = score_pattern.sub(score_block, contents)

    if contents != original:
        target.write_text(contents)
        return True
    return False


def _update_anti_llm_engine() -> bool:
    """Sync the Anti-LLM engine helpers with the merged behaviour."""

    target = ROOT / "anti_llm" / "engine.py"
    contents = target.read_text()
    original = contents

    normalize_needed = (
        "normalize_module1_candidates" in contents or "<<<<<<<" in contents
    )
    if normalize_needed:
        contents = contents.replace(
            "normalize_module1_candidates", "normalize_seed_candidate_payloads"
        )
        normalize_pattern = re.compile(
            r"    def _normalize_(?:seed_candidate_payloads|module1_candidates)\(self, candidates: Optional\[List\[Any\]\]\) -> List\[Dict\[str, Any\]\]:\n[\s\S]+?(?=    def |\Z)",
            re.MULTILINE,
        )
        normalize_block = (
            "    def _normalize_seed_candidate_payloads(self, candidates: Optional[List[Any]]) -> List[Dict[str, Any]]:\n"
            "        return normalize_seed_candidate_payloads(candidates, value_sanitizer=self._safe_float)\n\n"
        )
        contents = normalize_pattern.sub(normalize_block, contents)

    cmu_needed = "_cmu_candidates" in contents or "<<<<<<<" in contents
    if cmu_needed:
        contents = contents.replace("_cmu_candidates", "_fetch_cmu_seed_candidates")
        cmu_pattern = re.compile(
            r"    def _(?:cmu_candidates|fetch_cmu_seed_candidates)\(self, word: str, limit: int, analyzer: Any\) -> List\[Any\]:\n[\s\S]+?(?=    def |\Z)",
            re.MULTILINE,
        )
        cmu_block = (
            "    def _fetch_cmu_seed_candidates(self, word: str, limit: int, analyzer: Any) -> List[Any]:\n"
            "        \"\"\"Fetch CMU-derived seed candidates using the configured analyzer, returning an empty list when unavailable.\"\"\"\n"
            "        cmu_fn = getattr(self, \"_cmu_seed_fn\", None)\n"
            "        if callable(cmu_fn):\n"
            "            return cmu_fn(word, limit=limit, analyzer=analyzer)\n"
            "        return []\n\n"
        )
        contents = cmu_pattern.sub(cmu_block, contents)

    if contents != original:
        target.write_text(contents)
        return True
    return False


def _update_seed_expansion() -> bool:
    """Ensure :mod:`seed_expansion` exposes the refactored normaliser."""

    target = ROOT / "anti_llm" / "seed_expansion.py"
    contents = target.read_text()
    original = contents

    needs_rewrite = "normalize_module1_candidates" in contents or "<<<<<<<" in contents
    if needs_rewrite:
        function_pattern = re.compile(
            r"def normalize_(?:module1_candidates|seed_candidate_payloads)\([\s\S]+?(?=^def extract_suffixes)",
            re.MULTILINE,
        )

        final_block = (
            "def normalize_seed_candidate_payloads(\n"
            "    candidates: Optional[List[Any]],\n"
            "    value_sanitizer: Callable[[Any, float], float] = safe_float,\n"
            ") -> List[Dict[str, Any]]:\n"
            "    \"\"\"Normalise raw seed candidate payloads into a consistent structure of word, similarity, combined, and rarity scores.\"\"\"\n"
            "    normalized: List[Dict[str, Any]] = []\n"
            "    if not candidates:\n"
            "        return normalized\n\n"
            "    seen: Set[str] = set()\n\n"
            "    for candidate in candidates:\n"
            "        word: Optional[str] = None\n"
            "        similarity = 0.0\n"
            "        combined = 0.0\n"
            "        rarity = 0.0\n\n"
            "        if isinstance(candidate, dict):\n"
            "            word = (\n"
            "                candidate.get(\"word\")\n"
            "                or candidate.get(\"target\")\n"
            "                or candidate.get(\"candidate\")\n"
            "            )\n"
            "            similarity = value_sanitizer(\n"
            "                candidate.get(\"similarity\") or candidate.get(\"score\"),\n"
            "                default=0.0,\n"
            "            )\n"
            "            combined = value_sanitizer(\n"
            "                candidate.get(\"combined\")\n"
            "                or candidate.get(\"combined_score\")\n"
            "                or candidate.get(\"confidence\"),\n"
            "                default=similarity,\n"
            "            )\n"
            "            rarity = value_sanitizer(\n"
            "                candidate.get(\"rarity\") or candidate.get(\"rarity_score\"),\n"
            "                default=0.0,\n"
            "            )\n"
            "        else:\n"
            "            try:\n"
            "                word = candidate[0]\n"
            "                if len(candidate) > 1:\n"
            "                    similarity = value_sanitizer(candidate[1], default=0.0)\n"
            "                if len(candidate) > 2:\n"
            "                    combined = value_sanitizer(candidate[2], default=similarity)\n"
            "                if len(candidate) > 3:\n"
            "                    rarity = value_sanitizer(candidate[3], default=0.0)\n"
            "            except Exception:\n"
            "                continue\n\n"
            "        if not word:\n"
            "            continue\n\n"
            "        key = str(word).strip().lower()\n"
            "        if not key or key in seen:\n"
            "            continue\n\n"
            "        seen.add(key)\n"
            "        normalized.append(\n"
            "            {\n"
            "                \"candidate\": str(word).strip(),\n"
            "                \"similarity\": similarity,\n"
            "                \"combined\": combined,\n"
            "                \"rarity\": rarity,\n"
            "            }\n"
            "        )\n\n"
            "    return normalized\n\n"
        )

        contents = function_pattern.sub(final_block, contents)

    if contents != original:
        target.write_text(contents)
        return True
    return False


def main() -> None:
    """Apply all known merge corrections."""

    updated = False
    for fixer in (
        _update_search_service,
        _update_anti_llm_engine,
        _update_seed_expansion,
    ):
        updated = fixer() or updated

    if not updated:
        print("No known conflict patterns detected; repository already up to date.")


if __name__ == "__main__":
    main()
