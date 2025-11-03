# Label-Based Reference Extraction - Implementation Summary

## Overview

Successfully transformed the DSPy extraction pipeline to extract mathematical object references as **labels** (e.g., `thm-convergence`) instead of **text** (e.g., "Theorem 1.4"). This eliminates downstream reference resolution, improves data quality, and enables direct cross-referencing.

## Implementation Date

November 2, 2025

## Changes Summary

### Files Modified: 2
1. ✅ `src/mathster/parsing/extract_workflow.py` (~150 lines modified)
2. ✅ `src/mathster/core/raw_data.py` (~20 lines modified)

### Files Deleted: 1
3. ❌ `src/mathster/parsing/dspy_parser.py` (duplicate file removed)

### Files Created: 2
4. ✅ `tests/test_label_lookup.py` (unit tests for label lookup)
5. ✅ `tests/test_reference_validation.py` (integration tests for validation)

---

## Phase 1: Label Lookup Helper Function

### File: `src/mathster/parsing/extract_workflow.py`

**Added**: `lookup_label_from_context()` function (lines 54-153)

**Purpose**: Hybrid label resolution strategy
- **Strategy 1**: Search for `:label:` directive in Jupyter Book markup near reference
- **Strategy 2**: Generate standardized label from text if not found

**Function Signature**:
```python
def lookup_label_from_context(
    reference_text: str,
    context: str,
    reference_type: Literal["theorem", "definition", "proof"],
) -> str:
```

**Examples**:
```python
# Example 1: Found directive
lookup_label_from_context("Theorem 1.4", chapter_text, "theorem")
# → "thm-convergence" (found :label: thm-convergence near "Theorem 1.4")

# Example 2: Generated from text
lookup_label_from_context("Lemma 2.3", chapter_text, "theorem")
# → "lem-2-3" (no :label: found, generated from numbering)

# Example 3: Definition
lookup_label_from_context("Lipschitz continuous", chapter_text, "definition")
# → "def-lipschitz-continuous" (generated from term)
```

**Test Results**: ✅ All 7 tests passed (see `tests/test_label_lookup.py`)

---

## Phase 2: DSPy Field Description Updates

### File: `src/mathster/parsing/extract_workflow.py`

Updated 3 field descriptions to enforce label-based extraction:

#### 1. `TheoremExtraction.definition_references` (Lines 191-198)

**Before**:
```python
description="Terms that are clearly defined elsewhere (e.g., ['v-porous on lines', 'Lipschitz continuous'])"
```

**After**:
```python
description=(
    "Labels of definitions referenced in this theorem (e.g., ['def-lipschitz-continuous', 'def-v-porous-on-balls']). "
    "Extract labels from :label: directives if present, or generate using pattern def-{normalized-term}. "
    "ALWAYS use labels (def-*), NEVER use text like 'Lipschitz continuous'."
)
```

#### 2. `ProofExtraction.proves_label` (Lines 204-211)

**Before**:
```python
description="Label or reference to theorem being proved (e.g., 'Theorem 1.1', 'thm-main-result')"
```

**After**:
```python
description=(
    "Label of the theorem this proof proves (e.g., 'thm-main-result', 'lem-gradient-bound'). "
    "Extract from :label: directive near the theorem statement, or generate from theorem numbering (e.g., 'Theorem 1.1' → 'thm-1-1'). "
    "MUST match pattern: thm-*|lem-*|prop-*|cor-*. This is REQUIRED - extraction will fail if not a valid label."
)
```

#### 3. `ProofExtraction.theorem_references` (Lines 228-235)

**Before**:
```python
description="Explicit theorem references (e.g., ['Theorem 1.4', 'Lemma 2.3'])"
```

**After**:
```python
description=(
    "Labels of theorems/lemmas referenced in this proof (e.g., ['thm-convergence', 'lem-gradient-bound']). "
    "Extract from :label: directives or generate from numbering (e.g., 'Theorem 1.4' → 'thm-1-4'). "
    "Use patterns: thm-*, lem-*, prop-*, cor-*. PREFER labels over text when possible."
)
```

---

## Phase 3: DSPy Signature Docstring Update

### File: `src/mathster/parsing/extract_workflow.py`

**Updated**: `ExtractMathematicalConcepts` signature docstring (Lines 560-602)

**Added Section**: REFERENCE EXTRACTION GUIDELINES (CRITICAL)

**Key Guidelines Added**:

1. **Extract Labels, Not Text**:
   - ❌ WRONG: "Theorem 1.4", "Lipschitz continuous"
   - ✅ CORRECT: "thm-convergence", "def-lipschitz-continuous"

2. **Label Lookup Strategy**:
   - First: Search for :label: directive in Jupyter Book markup
   - Fallback: Generate standardized label from text

3. **Label Patterns**:
   - Definitions: `def-{normalized-term}`
   - Theorems: `thm-{number-or-name}`
   - Lemmas: `lem-{number-or-name}`
   - Propositions: `prop-{number-or-name}`
   - Corollaries: `cor-{number-or-name}`

4. **Examples** (3 concrete examples provided in docstring)

5. **Critical Field Warning**: `proves_label` is MANDATORY and MUST be valid label

6. **Fallback Behavior**: When :label: not found, generate from numbering

---

## Phase 4: Validation Logic

### File: `src/mathster/parsing/extract_workflow.py`

**Updated**: `validate_extraction()` function (Lines 436-463)

**Added Reference Format Validation**:

### Strict Validation (ERRORS)

```python
# STRICT: proves_label must be valid label format
for proof in extraction.proofs:
    if not any(proof.proves_label.startswith(p) for p in ["thm-", "lem-", "prop-", "cor-"]):
        errors.append(
            f"Proof '{proof.label}': proves_label MUST be a theorem label "
            f"(thm-*|lem-*|prop-*|cor-*), got '{proof.proves_label}'. "
            f"This is a CRITICAL error - extraction cannot proceed."
        )
```

### Permissive Validation (WARNINGS)

```python
# PERMISSIVE: definition_references (warnings only)
for thm in extraction.theorems:
    for def_ref in thm.definition_references:
        if not def_ref.startswith("def-"):
            warnings.append(
                f"Theorem '{thm.label}': definition_references should be labels "
                f"starting with 'def-', got '{def_ref}'. Consider extracting proper label."
            )

# PERMISSIVE: theorem_references (warnings only)
for proof in extraction.proofs:
    for thm_ref in proof.theorem_references:
        if not any(thm_ref.startswith(p) for p in ["thm-", "lem-", "prop-", "cor-"]):
            warnings.append(
                f"Proof '{proof.label}': theorem_references should be labels "
                f"(thm-*|lem-*|prop-*|cor-*), got '{thm_ref}'. Consider extracting proper label."
            )
```

**Test Results**: ✅ All 5 validation tests passed (see `tests/test_reference_validation.py`)

**Validation Behavior Summary**:
- `proves_label`: **STRICT** (errors on invalid format)
- `definition_references`: **PERMISSIVE** (warnings only)
- `theorem_references`: **PERMISSIVE** (warnings only)

---

## Phase 5: Raw Data Model Documentation

### File: `src/mathster/core/raw_data.py`

Updated 3 field descriptions in data models:

#### 1. `RawTheorem.explicit_definition_references` (Lines 281-288)

**Before**:
```python
description="Terms mentioned that are clearly defined elsewhere, as they appear in text. "
"Examples: ['v-porous on lines', 'Lipschitz continuous', 'uniformly convex']. "
"LLM should identify formal terminology that likely has a Definition."
```

**After**:
```python
description=(
    "Labels of definitions referenced in this theorem (e.g., ['def-lipschitz-continuous', 'def-v-porous-on-balls']). "
    "ALWAYS use label format (def-*), extracted from :label: directives or generated from term names. "
    "LLM should identify formal terminology that likely has a Definition."
)
```

#### 2. `RawProof.proves_label` (Lines 309-318)

**Before**:
```python
description="The label of the theorem this proof is for (e.g., 'Theorem 1.1', 'Lemma 3.4'). "
"Should match the label from the corresponding RawTheorem. "
"If proof says just 'Proof.' without explicit label, infer from context."
```

**After**:
```python
description=(
    "The label of the theorem this proof proves (e.g., 'thm-main-result', 'lem-gradient-bound'). "
    "MUST match the label from the corresponding RawTheorem exactly. "
    "Pattern: thm-*|lem-*|prop-*|cor-*. "
    "If proof header says 'Proof of Theorem 1.1', extract or generate the theorem's label."
)
```

#### 3. `RawProof.explicit_theorem_references` (Lines 343-350)

**Before**:
```python
description="Explicit references to other results in the paper. "
"Examples: ['Theorem 1.4', 'Lemma 2.3', 'Proposition 3.9', 'Corollary 1.2']. "
"Preserve exact text as cited."
```

**After**:
```python
description=(
    "Labels of theorems/lemmas referenced in this proof (e.g., ['thm-convergence', 'lem-gradient-bound']). "
    "ALWAYS use label format (thm-*|lem-*|prop-*|cor-*), extracted from :label: directives or generated from numbering. "
    "Preserve theorem references in label format for downstream cross-referencing."
)
```

---

## Phase 6: Remove Duplicate File

### Deleted: `src/mathster/parsing/dspy_parser.py`

**Reason**: Duplicate of `extract_workflow.py` (42KB file)

**Verification**:
- ✅ Checked for imports: No imports found
- ✅ No references in `dspy_pipeline.py`
- ✅ Safe to delete without breaking changes

---

## Phase 7: Testing

### Test Files Created

#### 1. `tests/test_label_lookup.py`

**Purpose**: Unit tests for `lookup_label_from_context()` function

**Tests** (7 total):
1. ✅ Find :label: directive near reference
2. ✅ Generate label from text when no directive found
3. ✅ Generate definition label
4. ✅ Generate lemma label
5. ✅ Handle line numbers correctly
6. ✅ Generate proposition label
7. ✅ Generate corollary label

**Results**: **All 7 tests passed**

#### 2. `tests/test_reference_validation.py`

**Purpose**: Integration tests for reference validation logic

**Tests** (5 total):
1. ✅ Strict validation catches invalid `proves_label`
2. ✅ Valid `proves_label` passes validation
3. ✅ Permissive validation warns on text `definition_references`
4. ✅ Permissive validation warns on text `theorem_references`
5. ✅ Valid label-based references produce no warnings

**Results**: **All 5 tests passed**

### Verification Commands

```bash
# Verify syntax
python3 -m py_compile src/mathster/parsing/extract_workflow.py
python3 -m py_compile src/mathster/core/raw_data.py
# ✅ All files compiled successfully

# Run unit tests
python3 tests/test_label_lookup.py
# ✅ All 7 tests passed

# Run validation tests
python3 tests/test_reference_validation.py
# ✅ All 5 tests passed
```

---

## Impact Analysis

### Before (Text-Based References)

**Example Extraction Output**:
```json
{
  "label": "proof-main",
  "proves_label": "Theorem 1.1",
  "theorem_references": ["Theorem 1.4", "Lemma 2.3"],
  "definition_references": ["Lipschitz continuous"]
}
```

**Problems**:
- Ambiguous references (which "Theorem 1.1"?)
- Requires downstream LLM-based resolution
- Additional latency and cost
- Brittle pattern matching
- No validation during extraction

### After (Label-Based References)

**Example Extraction Output**:
```json
{
  "label": "proof-main",
  "proves_label": "thm-main-result",
  "theorem_references": ["thm-convergence", "lem-gradient-bound"],
  "definition_references": ["def-lipschitz-continuous"]
}
```

**Benefits**:
- ✅ Unambiguous labels
- ✅ No downstream resolution needed
- ✅ Reduced latency (no post-extraction LLM calls)
- ✅ Validation during extraction
- ✅ Direct cross-referencing enabled
- ✅ Simplified enrichment workflow

---

## Transformation Examples

| Stage | Input (DSPy) | Output (Before) | Output (After) |
|-------|--------------|-----------------|----------------|
| **Theorem Reference** | "By Theorem 1.4..." | `["Theorem 1.4"]` | `["thm-convergence"]` |
| **Definition Reference** | "where f is Lipschitz continuous" | `["Lipschitz continuous"]` | `["def-lipschitz-continuous"]` |
| **Proof Proves** | "Proof of Theorem 1.1" | `"Theorem 1.1"` or `"theorem-1-1"` | `"thm-main-result"` |
| **Lemma Reference** | "By Lemma 2.3..." | `["Lemma 2.3"]` | `["lem-2-3"]` |
| **Multiple References** | "Using Theorem 1.4 and Lemma 2.3" | `["Theorem 1.4", "Lemma 2.3"]` | `["thm-1-4", "lem-2-3"]` |

---

## Label Generation Patterns

### Directive Lookup (Preferred)

When `:label:` directive found in context:

```markdown
:::{prf:theorem} Convergence Result
:label: thm-convergence

The algorithm converges.
:::
```

→ Extract: `"thm-convergence"` ✅

### Generated from Text (Fallback)

When no `:label:` directive found:

| Input Text | Generated Label | Pattern |
|------------|-----------------|---------|
| "Theorem 1.4" | `thm-1-4` | `thm-{numbers}` |
| "Lemma 2.3" | `lem-2-3` | `lem-{numbers}` |
| "Proposition 3.1" | `prop-3-1` | `prop-{numbers}` |
| "Corollary 4.2" | `cor-4-2` | `cor-{numbers}` |
| "Lipschitz continuous" | `def-lipschitz-continuous` | `def-{normalized-term}` |

---

## Validation Behavior

### Hybrid Validation Strategy

| Field | Validation Type | On Invalid Format | Rationale |
|-------|----------------|-------------------|-----------|
| `proves_label` | **STRICT** | ❌ ERROR (fails extraction) | Critical for proof-theorem linking |
| `definition_references` | **PERMISSIVE** | ⚠️ WARNING (continues) | Best-effort extraction, allow improvement |
| `theorem_references` | **PERMISSIVE** | ⚠️ WARNING (continues) | Best-effort extraction, allow improvement |

### Validation Error Examples

**STRICT Error** (fails extraction):
```
Proof 'proof-test': proves_label MUST be a theorem label (thm-*|lem-*|prop-*|cor-*),
got 'Theorem 1.1'. This is a CRITICAL error - extraction cannot proceed.
```

**PERMISSIVE Warning** (continues with warning):
```
Theorem 'thm-main': definition_references should be labels starting with 'def-',
got 'Lipschitz continuous'. Consider extracting proper label.
```

---

## Breaking Changes

### ⚠️ Data Format Change

**Old Format** (text-based):
```json
{
  "proves_label": "Theorem 1.1",
  "theorem_references": ["Theorem 1.4", "Lemma 2.3"],
  "definition_references": ["Lipschitz continuous"]
}
```

**New Format** (label-based):
```json
{
  "proves_label": "thm-main-result",
  "theorem_references": ["thm-convergence", "lem-gradient-bound"],
  "definition_references": ["def-lipschitz-continuous"]
}
```

### Migration Strategy

**Decision**: No migration needed (breaking change accepted by user)

**Rationale**:
- Old files will be regenerated by re-running extraction pipeline
- Fresh extraction preferred over migration
- Clean break ensures consistency

---

## Performance Impact

### Reduced Latency

**Before** (text-based):
```
Extraction → Text References → Post-Processing → LLM Resolution → Labels
                                 (~2-3 LLM calls per reference)
```

**After** (label-based):
```
Extraction → Labels (done during extraction)
             (0 additional LLM calls)
```

**Estimated Savings**:
- No post-extraction LLM calls for reference resolution
- ~2-3 seconds saved per document
- Reduced API costs

### Improved Accuracy

**Before**:
- Pattern matching for "Theorem 1.4" could match wrong theorem
- Ambiguous references in complex documents
- Manual fixes often required

**After**:
- Unambiguous labels from directive or generation
- Validation catches errors during extraction
- Consistent format across all entities

---

## Usage

### Running Extraction

```bash
# Extract with new label-based references
python -m mathster.parsing.dspy_pipeline docs/source/1_euclidean_gas/01_fragile_gas_framework.md

# Output will now contain label-based references
```

### Example Output

**chapter_0.json** (excerpt):
```json
{
  "theorems": [
    {
      "label": "thm-convergence",
      "definition_references": ["def-lipschitz", "def-continuous"],
      ...
    }
  ],
  "proofs": [
    {
      "label": "proof-convergence",
      "proves_label": "thm-convergence",
      "theorem_references": ["lem-bound", "prop-monotone"],
      ...
    }
  ]
}
```

---

## Next Steps (Optional)

### Recommended Enhancements

1. **ReAct Tool Integration** (optional):
   - Add `lookup_label_tool()` for agent to call during extraction
   - Enable agent to actively search context for labels
   - Requires integrating tool with ReAct agent

2. **Label Registry** (optional):
   - Maintain registry of extracted labels per document
   - Validate reference labels exist in registry
   - Catch broken references during extraction

3. **Batch Processing** (optional):
   - Re-extract existing documents with new pipeline
   - Verify all old text references converted to labels
   - Update any downstream tools expecting text format

---

## Summary

### ✅ Implementation Complete

All 7 phases completed successfully:
1. ✅ Label lookup helper function created
2. ✅ DSPy field descriptions updated
3. ✅ Signature docstring enhanced with guidelines
4. ✅ Validation logic added (strict + permissive)
5. ✅ Raw data model documentation updated
6. ✅ Duplicate file removed
7. ✅ Tests created and passing (12/12 tests)

### 📊 Impact

- **Lines Modified**: ~170 lines
- **Files Modified**: 2 files
- **Files Deleted**: 1 file (duplicate)
- **Tests Created**: 2 test files (12 tests total)
- **Test Success Rate**: 100% (12/12 passed)

### 🎯 Result

The DSPy extraction pipeline now extracts **label-based references** for all mathematical objects, eliminating downstream resolution and enabling direct cross-referencing throughout the framework.

**Key Benefits**:
- Unambiguous references
- No post-extraction LLM calls
- Validation during extraction
- Direct cross-referencing enabled
- Simplified enrichment workflow
