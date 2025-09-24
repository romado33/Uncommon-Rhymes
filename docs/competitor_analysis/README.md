# Competitor Functionality Audit Playbook

This playbook operationalises the research plan for benchmarking RhymeRarity against
popular rhyme dictionaries including **RhymeZone**, **Double-Rhyme**, **B-Rhymes**
and **Rhymes.com**. It breaks the process into repeatable stages so any contributor
can observe how competitors behave, capture the findings in structured notes, and
map the results onto the existing codebase.

---

## 1. Establish your research workspace

1. Clone this repository or pull the latest changes.
2. Create a working folder under `docs/competitor_analysis/records/` for the audit
   run you are about to perform, e.g. `docs/competitor_analysis/records/2024-09-15/`.
3. Copy the templates from `docs/competitor_analysis/templates/` into that folder.
   They contain the note-taking scaffolding referenced below.
4. Decide which browsers and devices you will test on (desktop and mobile layouts
   can expose different feature sets).
5. Prepare a list of seed prompts (single words, multi-word phrases, rare or slang
   terms) to exercise each site. The `seed_prompts` table inside the
   `site_audit_template.md` template includes a starting point.

---

## 2. Catalogue functionality per competitor site

For each target site repeat the following subsections. The `site_audit_template.md`
file has placeholders that mirror the headings below so you can log the data as
you go.

### 2.1 Manual feature discovery
- Run all prepared seed prompts and add new prompts when the interface suggests
  alternative searches.
- Record how queries are entered (single text box, advanced forms, filters).
- Note the result groupings (perfect vs. near rhymes, multi-syllabic groupings,
  popularity rankings, links to definitions or lyrics).
- Capture any extra utilities surfaced on the page (copy/share buttons, export
  options, audio pronunciation, usage examples, vocabulary helpers).

### 2.2 UI controls and states
- List every toggle, dropdown, slider, or accordion you interact with.
- Document default values versus user-adjustable ranges.
- Take screenshots when a state change reveals extra functionality (e.g., tabs
  for synonyms vs. antonyms, filters that appear only on hover).
- Pay attention to onboarding hints and contextual explanations that clarify how
the site expects users to interpret the results.

### 2.3 Network and data inspection
- Open the browser developer tools (Network tab) before submitting a query.
- Observe XHR/Fetch requests triggered during a search. Log endpoints, query
  parameters, and response payload types (HTML, JSON, XML).
- Save notable responses into the `network_observations` table (mask or omit any
  personally identifiable information if you have an account).
- If the responses are JSON, record field names that imply backend capabilities
  (e.g., `syllables`, `score`, `popularity_rank`, `rhyme_type`).
- Inspect any embedded scripts for inline configuration data that might list
  supported filters or thresholds.

### 2.4 Content provenance and guidance
- Check for FAQ, help, or documentation links that describe how the service
  computes rhymes or defines categories.
- Record references to external datasets, textbooks, or linguistic models.
- Note subscription tiers or gated features so the backlog can reflect whether
  parity requires authentication.

---

## 3. Build a cross-site comparison matrix

1. Consolidate the per-site notes into `comparison_matrix_template.md`.
2. Populate the table columns covering rhyme categorisation, filter controls,
   metadata enrichment, educational aides, and monetisation hooks.
3. Highlight differentiating behaviours using the "Key observations" column so
   stakeholders can quickly identify unique selling points.
4. Add supporting screenshots or request payload samples within collapsible
   sections to keep the matrix compact while retaining evidence.

---

## 4. Map findings onto RhymeRarity

Use the "RhymeRarity linkage" section of the comparison matrix to indicate how
existing modules relate to each capability:

- **Phonetic analysis** → `rhyme_rarity/core/module1_enhanced_core_phonetic.py`
  and the `EnhancedPhoneticAnalyzer` referenced by
  `rhyme_rarity/app/services/search_service.py`.
- **Anti-LLM rarity heuristics** → `anti_llm/module2_enhanced_anti_llm.py` and
  the way `SearchService.search_rhymes` aggregates results from
  `AntiLLMRhymeEngine`.
- **Cultural context** → `cultural/module3_enhanced_cultural_database.py` and the
  `CulturalIntelligenceEngine` filters in the search service.
- **UI workflows** → `rhyme_rarity/app/ui/gradio.py` for existing controls,
  plus any additional widgets that might be required to match competitor flows.

Whenever a competitor feature is already present in RhymeRarity, specify which
functions or endpoints exercise it (e.g., `SearchService.search_rhymes` supports
`allowed_rhyme_types`, `cultural_significance`, and `genres`). If a capability is
missing, note the gap and draft a backlog item in the "Follow-up actions" list in
your record folder.

---

## 5. Verify existing functionality

After mapping features, execute the following to confirm parity:

1. Run exploratory searches through the Gradio UI launched via `python app.py`.
   Mirror the competitor prompts and document whether filters behave as expected.
2. Use the CLI helper `scripts/feature_probe.py` (described below) to exercise
   service-level queries with specific filters and to capture machine-readable
   output for regression purposes.
3. Add or update automated tests under `tests/` when you discover features that
   should be locked down (e.g., verifying syllable filters or cultural matching).
4. Record evidence (screenshots, CLI logs) in your record folder so future audits
   can reuse the groundwork.

---

## 6. Regression and backlog management

- **Regression tracking**: Store the filled templates in git so historical audits
  remain reviewable. Create follow-up issues summarising gaps discovered during
  the comparison.
- **Data refresh**: When adding new datasets to mirror competitor breadth,
  document schema changes or import scripts alongside the audit notes.
- **Release validation**: Re-run the comparison when major features land or when
  competitor sites roll out visible changes.

---

## 7. Tooling reference

### CLI probe
`scripts/feature_probe.py` wraps the `RhymeRarityApp` so you can quickly verify
search filters without launching the UI. Example:

```bash
python scripts/feature_probe.py "love" \
  --limit 25 \
  --min-confidence 0.6 \
  --cultural golden-era underground \
  --genres hip-hop \
  --rhyme-types perfect near \
  --sources phonetic cultural
```

Run `python scripts/feature_probe.py --help` for the full option list.

### Templates
- `site_audit_template.md`: one per competitor site.
- `comparison_matrix_template.md`: roll-up table that compares sites.
- `verification_checklist.md`: keeps track of verification status once features
  are implemented or confirmed.

Keep this playbook updated as the research process evolves so future auditors can
follow consistent steps.
