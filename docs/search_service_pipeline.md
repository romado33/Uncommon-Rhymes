# `SearchService.search_rhymes` Pipeline

## Stage Map
```
user input
   ↓
Normalization & filter prep
   ↓
Phonetic scoring & seed harvesting
   ↓
Cultural repository enrichment
   ↓
Anti-LLM pattern generation
   ↓
Result normalization & return
```

> **Note:** The "Anti-LLM" phase does not introduce a distinct dataset. It reuses the CMU-derived phonetic candidates harvested in the previous stage, layering rarity heuristics and synthesis on top of the same pipeline that powers the CMU column in the UI. As a result the product exposes two high-level result families: CMU/Anti-LLM (combined phonetic outputs) and the cultural repository branch.

## 1. Normalization & Input Preparation
* Trim, lowercase, and validate the source word while capturing telemetry of raw inputs and limits.
* Coerce confidence to a float, normalise filter labels (cultural, genre, rhyme type, cadence, Bradley devices), and clamp numeric thresholds for syllables, rarity, stress, and concurrency filters.【F:rhyme_rarity/app/services/search_service.py†L452-L558】
* Capture the caller-provided `slant_strength`, forward it to telemetry, and carry the value through the internal request envelope before clamping to the supported 0–1 range downstream.【F:rhyme_rarity/app/services/search_service.py†L555-L607】【F:rhyme_rarity/app/services/search_service.py†L2430-L2475】
* Resolve which result sources to include, detect whether any filters are active, and fetch shared collaborators: the `EnhancedPhoneticAnalyzer`, optional cultural engine, and CMU loader reference used throughout the remainder of the search.【F:rhyme_rarity/app/services/search_service.py†L526-L568】
* Decompose the phrase, then build a rich phonetic profile of individual words by reusing the analyzer’s descriptive APIs for pronunciations, stress, syllable counts, and anchor metadata.【F:rhyme_rarity/app/services/search_service.py†L567-L652】

## 2. Phonetic Scoring & Seed Harvesting
* Derive rhyme signatures via the cultural engine when available, fall back to CMU rhyme parts, and finally fall back to the spelling-based signature cache, guaranteeing a baseline signature set.【F:rhyme_rarity/app/services/search_service.py†L725-L747】
* Query cached CMU rhyme candidates and compute a dynamic reference similarity threshold; supplement these with repository hints by reusing the analyzer to score candidate similarities.【F:rhyme_rarity/app/services/search_service.py†L749-L799】
* Assemble a detailed phonetic profile for the source phrase, accumulate phonetic matches, and enrich each candidate via analyzer and cultural engine callbacks before caching delivered words for downstream stages.【F:rhyme_rarity/app/services/search_service.py†L808-L1078】
* Apply the slant-strength gate to single-word phonetic matches, using `passes_gate` for the strict setting and proportional floors for relaxed strengths, while recording which end words survive for later stages.【F:rhyme_rarity/app/services/search_service.py†L1882-L1934】【F:rhyme_rarity/app/services/search_service.py†L2520-L2538】
* Harvest rare phonetic seeds only when the Anti-LLM branch is requested, capturing rarity, combined scores, and signature payloads for use as Module 1 seeds. This avoids redundant work when the Anti-LLM engine is disabled.【F:rhyme_rarity/app/services/search_service.py†L1080-L1141】

## 3. Cultural Repository Enrichment
* Query the SQLite repository for source and target matches constrained by the previously normalised filters and phonetic thresholds.【F:rhyme_rarity/app/services/search_service.py†L1204-L1237】
* Build per-entry dictionaries, optionally augment them with cultural context, rarity scores, and refined phonetic/alignment information from the cultural intelligence engine, then sanitise feature and prosody profiles for downstream filtering.【F:rhyme_rarity/app/services/search_service.py†L1238-L1381】
* Sort and time the cultural branch before normalisation merges cultural results with other sources.【F:rhyme_rarity/app/services/search_service.py†L1385-L1401】

## 4. Anti-LLM Pattern Generation
* Invoke the Anti-LLM engine with harvested Module 1 seeds, aggregated rhyme signatures, and already-delivered words to discourage duplicates.【F:rhyme_rarity/app/services/search_service.py†L1403-L1416】
* Respect the slant-strength gate when selecting Module 1 seeds—only allowing words that cleared the single-word filter when gating is active—before passing them into the generator.【F:rhyme_rarity/app/services/search_service.py†L1944-L1957】【F:rhyme_rarity/app/services/search_service.py†L2524-L2529】
* For each generated pattern, carry forward shared phonetic context, prosody metadata, and analyzer-derived phonetics while sanitising feature profiles and enforcing confidence thresholds.【F:rhyme_rarity/app/services/search_service.py†L1417-L1471】

## 5. Result Normalisation & Output
* Downstream helpers ensure minimum confidence, fill in feature profiles, derive rhyme categories/rhythm scores, and apply all user filters across phonetic, cultural, and anti-LLM branches before sorting and truncating the result lists.【F:rhyme_rarity/app/services/search_service.py†L1473-L1937】
* Filter multi-word and anti-LLM candidates against the gated end-word set when applicable, surface diagnostic metadata about the enforced gate, and expose the surviving tiers back to the caller.【F:rhyme_rarity/app/services/search_service.py†L1960-L2060】【F:rhyme_rarity/app/services/search_service.py†L2300-L2356】

## Shared Collaborator Reuse
* `EnhancedPhoneticAnalyzer` powers pronunciation descriptions, phonetic similarity checks, rarity scoring, syllable estimation, and Module 1 seed enrichment across multiple branches.【F:rhyme_rarity/app/services/search_service.py†L590-L651】【F:rhyme_rarity/app/services/search_service.py†L778-L1118】【F:rhyme_rarity/app/services/search_service.py†L1417-L1471】
* `SQLiteRhymeRepository` supplies both related word hints (feeding phonetic scoring) and cultural match rows, letting the service reuse cached repository data in separate stages.【F:rhyme_rarity/app/services/search_service.py†L777-L785】【F:rhyme_rarity/app/services/search_service.py†L1204-L1237】
* The cultural engine contributes rhyme signatures, alignment scoring, contextual rarity, and narrative enrichment, sharing outputs across phonetic and cultural phases.【F:rhyme_rarity/app/services/search_service.py†L731-L937】【F:rhyme_rarity/app/services/search_service.py†L1266-L1343】
* The Anti-LLM engine consumes analyzer-derived Module 1 seeds and prior delivery sets to complement repository results with synthetic patterns.【F:rhyme_rarity/app/services/search_service.py†L1403-L1471】

## Redundant Work & Adjustments
* Module 1 seed harvesting is now gated behind the Anti-LLM branch flag, preventing unnecessary rarity/signature calculations when that branch is disabled.【F:rhyme_rarity/app/services/search_service.py†L1080-L1141】
* Feature profile normalisation now flows through the existing sanitiser helper everywhere, eliminating duplicated conversion logic before filtering and scoring decisions.【F:rhyme_rarity/app/services/search_service.py†L668-L708】【F:rhyme_rarity/app/services/search_service.py†L1484-L1488】
* The service reuses the analyzer instance directly when evaluating rarity scores for CMU fallbacks, avoiding an extra attribute lookup on every candidate.【F:rhyme_rarity/app/services/search_service.py†L833-L887】

These adjustments reduce redundant work while keeping telemetry, caching, and result shaping behaviours intact.
