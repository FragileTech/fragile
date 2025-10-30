# Citation Validation Report
## Fragile Gas Framework - Raw Citations

**Date**: 2025-10-28
**Document**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md`
**Citations Directory**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/citations/`

---

## Executive Summary

**Issue Identified**: The single entry in the citations directory (`raw-cite-001.json`) is **not a bibliographic citation** but rather a **cross-reference to another framework document**.

**Status**: INVALID CITATION - Should be removed or reclassified
**Recommendation**: Delete this file and handle the cross-reference in the enrichment stage

---

## Detailed Analysis

### Entry: raw-cite-001.json

**Current Content**:
```json
{
  "temp_id": "raw-cite-001",
  "key_in_text": "03_cloning.md",
  "full_entry_text": "*\"The Keystone Principle and the Contractive Nature of Cloning\"* {prf:ref}`03_cloning.md`",
  "source_section": "§1-introduction"
}
```

**Source Context** (line 29 of framework document):
> This document establishes the *tools* (metrics, axioms, continuity bounds); the companion document establishes the *dynamics* (drift inequalities, convergence rates).

Full sentence:
> **Scope and Limitations**: This document focuses exclusively on defining the framework and proving operator-level stability properties. It does *not* prove convergence to quasi-stationary distributions—that analysis requires the Keystone Principle and hypocoercive Lyapunov analysis, which are developed in the companion document *"The Keystone Principle and the Contractive Nature of Cloning"* {prf:ref}`03_cloning.md`.

### Why This Is Not a Citation

1. **Format**: Uses Jupyter Book cross-reference syntax `{prf:ref}` instead of bibliographic citation format (e.g., `[16]`, `(Author, Year)`)

2. **Target**: References another document in the same framework (`03_cloning.md`) rather than external published work

3. **Purpose**: Internal navigation within the Fragile documentation, not attribution of external research

4. **Schema Mismatch**: RawCitation schema expects bibliographic entries with:
   - `key_in_text`: Citation key like "[16]" or "Han2016"
   - `full_entry_text`: Complete bibliography entry with authors, title, journal, year, etc.

5. **No Bibliography Section**: The framework document contains no bibliography or references section with traditional citations

### What This Actually Is

This is a **document cross-reference** that should be handled as:
- **In Stage 1 (Raw Extraction)**: Should not be extracted as a citation at all
- **In Stage 2 (Enrichment)**: Cross-references between framework documents should be tracked in the relationship graph
- **Entity Type**: This is metadata about document dependencies, not a citation entity

---

## Validation Against RawCitation Schema

### Schema Requirements (from staging_types.py):

```python
class RawCitation(BaseModel):
    """
    Direct transcription of a single bibliographic entry.

    Examples:
        - "[16] R. Han and D. Slepčev. Stochastic dynamics on hypergraphs..."
        - "[5] D.L. Donoho. Compressed sensing. IEEE Trans. Inform. Theory..."
    """

    key_in_text: str  # e.g., "[16]", "Han2016"
    full_entry_text: str  # Complete bibliography entry
```

### Validation Results:

| Field | Expected | Actual | Status |
|-------|----------|--------|--------|
| `temp_id` | `raw-cite-XXX` | `raw-cite-001` | ✓ VALID (format) |
| `key_in_text` | Citation key like "[16]" | `03_cloning.md` | ✗ INVALID (not a citation key) |
| `full_entry_text` | Full bibliography entry | Cross-reference syntax | ✗ INVALID (not a bibliography entry) |
| `source_section` | Section identifier | `§1-introduction` | ✓ VALID |

**Overall Schema Validation**: ✗ FAILED

---

## Recommendations

### Immediate Action

**Delete the file**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/citations/raw-cite-001.json`

**Rationale**: This entry does not represent a bibliographic citation and should not be processed through the citation enrichment pipeline.

### Stage 1 (Raw Extraction) Fix

Update the document-parser agent to:
1. **Exclude Jupyter Book cross-references** from citation extraction
2. **Pattern to exclude**: `{prf:ref}\`...\``
3. **Only extract true bibliographic citations**:
   - Numbered references: `[1]`, `[16]`, `[13, 21, 22]`
   - Author-year format: `(Han, 2016)`, `(Donoho et al., 2009)`
   - From dedicated bibliography/references sections

### Stage 2 (Enrichment) Enhancement

Add **document cross-reference tracking**:
1. Create a separate entity type for framework document dependencies
2. Extract cross-references like `{prf:ref}\`03_cloning.md\`` as relationships
3. Build a document dependency graph for the framework

### Example Proper Citation

If the framework document had actual citations, they would look like:

```json
{
  "temp_id": "raw-cite-001",
  "key_in_text": "[16]",
  "full_entry_text": "[16] R. Han and D. Slepčev. Stochastic dynamics on hypergraphs and the multi-body interaction problem. Journal of Functional Analysis, 2016.",
  "source_section": "§references"
}
```

---

## Conclusion

The Fragile Gas Framework document (`01_fragile_gas_framework.md`) contains **zero bibliographic citations**. The single entry in the citations directory is a misclassified cross-reference to another framework document.

**Actions Required**:
1. Delete `raw-cite-001.json`
2. Update extraction statistics to reflect 0 citations
3. Consider implementing document cross-reference tracking as a separate feature

**Final Validation Status**:
- **Citations extracted**: 1
- **Valid citations**: 0
- **Invalid citations**: 1 (cross-reference)
- **Success rate**: 0%

---

## Notes for Future Extraction

The framework documents use extensive **internal cross-referencing** via Jupyter Book directives:
- Theorem references: `{prf:ref}\`thm-revival-guarantee\``
- Document references: `{prf:ref}\`03_cloning.md\``
- Definition references: `{prf:ref}\`def-swarm-state\``

These should be tracked separately from bibliographic citations and used to build the framework's dependency graph during Stage 2 enrichment.
