# Synthesis Memo: Datasheets for Datasets & Data Cards

**Papers:** 
- Gebru et al., "Datasheets for Datasets" (FAccT 2021)
- Pushkarna et al., "Data Cards: Purposeful and Transparent Dataset Documentation" (FAccT 2022)

**Date:** 2026-04-29  
**Author:** Tenacious Academy

---

## Summary

**Datasheets for Datasets (Gebru et al., 2021):**
- Proposes a standardized documentation format for datasets
- 7 sections: Motivation, Composition, Collection Process, Preprocessing, Uses, Distribution, Maintenance
- Aims to increase transparency and accountability in ML datasets

**Data Cards (Pushkarna et al., 2022):**
- Extends datasheets with modular layered detail
- Three views: Telescopic (high-level), Periscopic (structural), Microscopic (per-task)
- Emphasizes purposefulness and transparency

---

## Design Choice Disagreement

**The papers implicitly assume a single-author voice for dataset documentation.**

I disagree with this assumption for Tenacious-Bench v0.1. The dataset is created by a trainee within a program structure, not by an individual researcher. Therefore:

1. **Author attribution:** "trainee + program staff sign-off" rather than individual name
2. **Provenance tracking:** Pushkarna-style microscopic detail documents per-task provenance (author_model + judge_model + source_probe_id) so any single row can be audited without reading the whole sheet
3. **Version history:** Explicit section for tracking changes across versions

**Why this matters:** In an educational program context, the "author" is not a single researcher but a learning entity operating under staff oversight. The datasheet should reflect this reality, not pretend to be a traditional research artifact.

---

## Evidence from Our Datasheet

Our datasheet.md includes:

```markdown
## Pushkarna Layered Detail (Data Cards)

### Telescopic View (High-Level Summary)
- **Purpose:** Evaluate Tenacious-specific B2B sales agent failure modes
- **Scope:** 254 tasks across 10 dimensions, 3 partitions

### Periscopic View (Structural Summary)
| Partition | Tasks | % | Intended Use |
|-----------|-------|---|--------------|
| train | 125 | 49% | Training data |
| dev | 63 | 25% | Public iteration |
| held_out | 66 | 26% | Sealed evaluation |

### Microscopic View (Per-Task Detail)
Each task includes full input context, rubric, metadata
```

This structure directly applies the Pushkarna layered approach while maintaining the Gebru 7-section framework.

---

## Application to Tenacious-Bench

1. **Motivation section:** Documents why τ²-Bench retail fails for Tenacious-specific behavior
2. **Composition section:** Full breakdown by dimension, source mode, difficulty
3. **Collection process:** Documents multi-LLM pipeline and judge filtering
4. **Preprocessing:** Documents redaction, signal windowing, contamination checks
5. **Uses:** Primary (evaluation), secondary (training), tertiary (template)
6. **Distribution:** HuggingFace Hub, CC-BY-4.0 license
7. **Maintenance:** Version history, update policy, external contribution guidelines

---

## Conclusion

The Gebru/Pushkarna framework provides excellent structure for Tenacious-Bench documentation. The only modification needed is the author-voice adjustment to reflect the educational program context rather than traditional research authorship.

---

**Word count:** ~320