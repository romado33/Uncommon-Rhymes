# {SITE_NAME} — Feature Audit

## Metadata
- **Auditor**: {YOUR_NAME}
- **Date**: {YYYY-MM-DD}
- **Device(s) tested**: {e.g., Desktop Chrome 126, iPhone Safari}
- **Account status**: {Guest / Logged in / Subscription tier}

---

## Seed prompts exercised
| Prompt | Purpose | Notes |
| --- | --- | --- |
| love | Baseline common word |  |
| time | Compare with love |  |
| money | Stress multi-syllable results |  |
| inner peace | Phrase handling |  |
| {add more} |  |  |

---

## Feature discovery log
| Timestamp | Action | Observation |
| --- | --- | --- |
| 12:03 | Submitted "love" | Returned groups: perfect rhymes, near rhymes, synonyms |
| 12:06 | Opened advanced filters | Revealed syllable range slider |
| {..} |  |  |

---

## Result structure
- **Primary groupings**: {perfect, near, phrases, ...}
- **Sorting cues**: {alphabetical, popularity, score, ...}
- **Metadata fields**: {syllables, stress pattern, definitions, usage examples, ...}
- **Export/share tools**: {copy, print, permalink, API, ...}

---

## UI controls
| Control | Type | Default | Range / Options | Behaviour |
| --- | --- | --- | --- | --- |
| Advanced filters accordion | Accordion | Closed | N/A | Reveals syllable + rhyme type filters |
| Syllable slider | Slider | 1-4 | 1-8 | Filters results instantly |
| {..} |  |  |  |  |

---

## Network observations
| Request URL | Trigger | Notable parameters | Response format | Notes |
| --- | --- | --- | --- | --- |
| https://example.com/rhyme?word=love | Search submit | word, syllables | JSON | Contains `syllables`, `frequency` |
| {..} |  |  |  |  |

Use this space for prettified payload samples:
```json
{
  "word": "love",
  "matches": [
    {"term": "dove", "score": 0.92, "syllables": 1}
  ]
}
```

---

## Content provenance
- **Help / FAQ references**: {links}
- **Mentioned datasets or corpora**: {description}
- **Educational aides**: {explanations, definitions, videos}
- **Monetisation hooks**: {ads, premium upsell, donations}

---

## Standout behaviours
- Itemise unique or surprising capabilities that RhymeRarity should emulate or deliberately differentiate from.

---

## Follow-up actions
- [ ] {Task 1}
- [ ] {Task 2}
- [ ] Add backlog issue linking to this audit
