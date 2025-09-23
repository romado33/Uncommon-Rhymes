---
title: "RhymeRarity"
emoji: "🎤"
colorFrom: "purple"
colorTo: "indigo"
sdk: "gradio"
sdk_version: "4.36.1"
app_file: "app.py"
---

# RhymeRarity

RhymeRarity is an experimental research project that explores uncommon rhyme patterns drawn from authentic hip-hop lyrics. The application bundles three cooperating engines that surface culturally grounded rhymes which are difficult for large language models to produce on demand:

- **Module 1 – Enhanced Core Phonetic Analysis** (`module1_enhanced_core_phonetic.py`): scores phonetic similarity, syllable span, stress alignment, and Bradley-inspired rhyme devices using the CMU Pronouncing Dictionary.
- **Module 2 – Enhanced Anti-LLM Rhyme Engine** (`module2_enhanced_anti_llm.py`): expands on Module 1 results to prioritise patterns that challenge common LLM failure modes while capturing prosody and internal rhyme complexity.
- **Module 3 – Enhanced Cultural Database Engine** (`module3_enhanced_cultural_database.py`): layers cultural context, attribution, rhythm-aware metadata, and genre-aware filtering over the rhyme search.

The project ships with a Gradio interface (`app.py`) that ties the modules together and exposes an interactive search workflow. When the bundled `patterns.db` file is missing the app automatically creates a demo database populated with sample rhyme patterns so you can explore the workflow immediately.

### Research-driven rhyme metrics

The refreshed release integrates insights from contemporary rap-poetics scholarship and popular rhyme dictionaries:

- **Bradley device detection**: classify matches as pure, multisyllabic, assonant, consonant, or slant rhymes following Adam Bradley's *Book of Rhymes* taxonomy.
- **Syllable and stress filters**: constrain search results by syllable span, stress alignment, and cadence tags to emulate RhymeZone, Double-Rhyme, and Bryant's pattern filters while surfacing uncommon, rap-oriented pairings.
- **Prosody analytics**: compute cadence ratios, internal rhyme potential, and sonic blend (assonance/consonance) so writers can balance rarity against rhythmic feel.
- **Rap metadata enrichment**: anti-LLM patterns inherit artist, era, and stylistic signals from the cultural engine while tagging LLM weakness categories for targeted practice.

## Project layout

```
├── app.py                             # Gradio application entry-point
├── module1_enhanced_core_phonetic.py  # Phonetic analysis and CMU dictionary helpers
├── module2_enhanced_anti_llm.py       # Anti-LLM rarity and cultural heuristics
├── module3_enhanced_cultural_database.py  # Cultural attribution and filtering logic
├── cmudict.7b                         # CMU Pronouncing Dictionary excerpt
├── tests/                             # Pytest-based regression suite
│   ├── test_app.py
│   ├── test_cmudict_loader.py
│   └── test_cultural_engine.py
└── requirements.txt
```

## Getting started

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 2. Install dependencies

Install all runtime and development dependencies with:

```bash
pip install -r requirements.txt
```

The requirements include Gradio for the UI, Pandas for potential data export, Pronouncing as an optional fallback rhyme source, and Pytest for the bundled tests.

### 3. Launch the Gradio app

Run the following command from the project root:

```bash
python app.py
```

The first launch will validate (or build) `patterns.db` and start a Gradio server on <http://localhost:7860>. Use the interface to search for rhymes, filter by cultural metadata, and compare results from the phonetic, cultural, and anti-LLM engines. Set the `server_name` and `server_port` arguments inside `app.py` if you need to deploy to a different host or port.

### 4. Run the automated tests

Execute the Pytest suite to verify the phonetic loader, cultural engine, and Gradio workflow helpers:

```bash
pytest
```

Running the tests ensures the fallback database creation and rhyme search behave as expected.

## Additional notes

- The CMU dictionary (`cmudict.7b`) must remain alongside `module1_enhanced_core_phonetic.py` so that the phonetic loader can locate it. You can swap in a larger dictionary file if desired.
- If you maintain your own rhyme database, replace `patterns.db` with your dataset. The application will automatically derive rarity scores and cultural filters from the supplied data.
- Cultural significance filters blend values stored in your database with curated category descriptions (e.g., `classic`, `cultural-icon`, `underground`) so the dropdown always reflects available metadata. The anti-LLM cultural-depth queries automatically detect whichever `cultural_significance` labels exist in the database, allowing you to introduce custom taxonomies without code changes.
- `app.py` contains helper methods such as `format_rhyme_results` and `search_rhymes` that you can import into other projects or wrap with alternative front-ends.

Happy rhyming!
