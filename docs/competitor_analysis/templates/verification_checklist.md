# Verification Checklist

Use this checklist after completing the comparison matrix to confirm which features
exist today and where work remains.

## Parity validation
- [ ] Gradio UI exposes equivalent filters (cultural, genre, rhyme type).
- [ ] CLI probe reproduces competitor query patterns.
- [ ] `SearchService.search_rhymes` returns grouped results with combined scores.
- [ ] Cultural engine surfaces expected metadata (`cultural_significance`, `genres`).
- [ ] Anti-LLM engine supplies rarity metrics for advanced patterns.
- [ ] Database schema supports required metadata fields.
- [ ] Automated tests cover each verified capability.

## Gaps & backlog
| Capability gap | Suggested owner | Priority | Notes |
| --- | --- | --- | --- |
|  |  |  |  |

## Evidence log
Link to CLI outputs, screenshots, or test runs that demonstrate verification.
