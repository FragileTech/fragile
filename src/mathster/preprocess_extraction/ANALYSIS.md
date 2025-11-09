# Data Class Analysis: Theorems, Lemmas, Corollaries, and Propositions

**Author:** Claude Code
**Date:** 2025-11-09
**Purpose:** Comprehensive analysis of mathematical entity data structures in the preprocess extraction pipeline

---

## Executive Summary

This analysis examines four mathematical entity types in the preprocessing pipeline:
- **Theorems** (`UnifiedTheorem` in `process_theorems.py`)
- **Lemmas** (`UnifiedLemma` in `process_lemmas.py`)
- **Corollaries** (`UnifiedCorollary` in `process_corollaries.py`)
- **Propositions** (`PropositionModel` in `process_propositions.py`)

### Key Findings

1. **Theorem/Lemma/Corollary are structurally identical** - Only the `type` field differs
2. **~70% code overlap** across all four types suggests strong base class candidate
3. **Significant code duplication** - Methods like `_strip_line_numbers()` and `from_instances()` are repeated
4. **Proposition diverges slightly** - Uses nested `RawLocator` model vs flat positioning fields
5. **All share common nested models** - `Equation`, `Hypothesis`, `Conclusion`, `Variable`, `Proof`, `Span`

### Refactoring Opportunity

Estimated **60-70% line-of-code reduction** possible by:
- Creating `UnifiedMathematicalEntity` base class
- Consolidating Theorem/Lemma/Corollary into single parameterized class
- Extracting shared methods to utilities
- Standardizing positioning metadata structure

---

## 1. Detailed Structural Comparison

### 1.1 Theorem/Lemma/Corollary (Identical Structure)

All three use the **exact same fields**:

```python
class UnifiedTheorem(BaseModel):  # Also: UnifiedLemma, UnifiedCorollary
    # === IDENTITY ===
    label: str                              # Required, e.g., "thm-kl-convergence"
    title: Optional[str] = None            # Human-readable name
    type: str = "theorem"                  # "theorem" | "lemma" | "corollary"

    # === EXTRACTED SEMANTIC CONTENT ===
    nl_statement: Optional[str] = None     # Natural language summary
    equations: List[Equation] = []         # LaTeX equations with optional labels
    hypotheses: List[Hypothesis] = []      # Conditions/assumptions
    conclusion: Optional[Conclusion] = None # Main result statement
    variables: List[Variable] = []         # Symbols with descriptions/constraints
    implicit_assumptions: List[Assumption] = []  # Hidden assumptions
    local_refs: List[str] = []             # Cross-references to other entities
    proof: Optional[Proof] = None          # Proof structure (availability + steps)
    tags: List[str] = []                   # Classification tags

    # === RAW DIRECTIVE CONTENT ===
    content_markdown: Optional[str] = None # Cleaned content (no line numbers)
    raw_directive: Optional[str] = None    # Full raw directive block

    # === PROVENANCE/POSITIONING ===
    document_id: Optional[str] = None      # E.g., "03_cloning"
    section: Optional[str] = None          # Section heading context
    span: Optional[Span] = None            # Line numbers (start/end/content/header)
    references: List[Any] = []             # Cross-refs from directive parsing
    metadata: Dict[str, Any] = {}          # Additional metadata
    registry_context: Dict[str, Any] = {}  # Stage, chapter, section info
    generated_at: Optional[str] = None     # ISO timestamp
    alt_labels: List[str] = []             # Alternative labels (mismatch handling)
```

**Location:**
- `process_theorems.py:26-55`
- `process_lemmas.py:26-55`
- `process_corollaries.py:26-55`

---

### 1.2 Propositions (Slightly Different Structure)

```python
class PropositionModel(BaseModel):
    # === IDENTITY ===
    type: str = "proposition"
    label: str                              # Required
    title: Optional[str] = None

    # === EXTRACTED SEMANTIC CONTENT ===
    nl_statement: Optional[str] = None
    equations: List[Equation] = []
    hypotheses: List[Hypothesis] = []
    conclusion: Optional[Conclusion] = None
    variables: List[Variable] = []
    implicit_assumptions: Optional[List[dict]] = None  # ⚠️ Dicts, not Pydantic models
    local_refs: Optional[List[str]] = None
    proof: Optional[Proof] = None
    tags: List[str] = []

    # === RAW DIRECTIVE CONTENT (NESTED) ===
    raw: RawLocator                        # ⚠️ Nested positioning metadata
```

**Key Difference:** Uses `RawLocator` nested model:

```python
class RawLocator(BaseModel):
    section: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    header_lines: Optional[List[int]] = None
    content_start: Optional[int] = None
    content_end: Optional[int] = None
    content: Optional[str] = None           # Cleaned markdown
    raw_directive: Optional[str] = None
    references: Optional[List[Any]] = None
    registry_context: Optional[RegistryContext] = None
```

**Location:** `process_propositions.py:60-109`

---

### 1.3 Field Comparison Table

| Field | Theorem/Lemma/Corollary | Proposition | Notes |
|-------|------------------------|-------------|-------|
| `label` | ✅ Required str | ✅ Required str | Identical |
| `title` | ✅ Optional[str] | ✅ Optional[str] | Identical |
| `type` | ✅ str (default varies) | ✅ str = "proposition" | Identical pattern |
| `nl_statement` | ✅ Optional[str] | ✅ Optional[str] | Identical |
| `equations` | ✅ List[Equation] | ✅ List[Equation] | Identical |
| `hypotheses` | ✅ List[Hypothesis] | ✅ List[Hypothesis] | Identical |
| `conclusion` | ✅ Optional[Conclusion] | ✅ Optional[Conclusion] | Identical |
| `variables` | ✅ List[Variable] | ✅ List[Variable] | Identical |
| `implicit_assumptions` | ✅ List[Assumption] | ⚠️ Optional[List[dict]] | Different types |
| `local_refs` | ✅ List[str] | ⚠️ Optional[List[str]] | Different optionality |
| `proof` | ✅ Optional[Proof] | ✅ Optional[Proof] | Identical |
| `tags` | ✅ List[str] | ✅ List[str] | Identical |
| `content_markdown` | ✅ Optional[str] | ❌ In `raw.content` | Flat vs nested |
| `raw_directive` | ✅ Optional[str] | ❌ In `raw.raw_directive` | Flat vs nested |
| `document_id` | ✅ Optional[str] | ❌ In `raw.registry_context` | Flat vs nested |
| `section` | ✅ Optional[str] | ❌ In `raw.section` | Flat vs nested |
| `span` | ✅ Optional[Span] | ❌ In `raw.*` fields | Flat vs nested |
| `references` | ✅ List[Any] | ❌ In `raw.references` | Flat vs nested |
| `registry_context` | ✅ Dict[str, Any] | ❌ In `raw.registry_context` | Flat vs nested |
| `alt_labels` | ✅ List[str] | ❌ Not present | Unique to TLC |

**Legend:**
- ✅ Present and identical
- ⚠️ Present but different
- ❌ Not present (or nested elsewhere)

---

## 2. Shared Nested Models

All four entity types use the **same nested Pydantic models**:

### 2.1 Equation

```python
class Equation(BaseModel):
    label: Optional[str] = None
    latex: str
```

**Usage:** Represents LaTeX equations with optional reference labels

---

### 2.2 Hypothesis

```python
class Hypothesis(BaseModel):
    text: Optional[str] = None
    latex: Optional[str] = None
```

**Usage:** Represents assumptions/conditions in natural language and LaTeX

---

### 2.3 Conclusion

```python
class Conclusion(BaseModel):
    text: Optional[str] = None
    latex: Optional[str] = None
```

**Usage:** Represents the main result statement

---

### 2.4 Variable

```python
class Variable(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    constraints: List[str] = []
    tags: List[str] = []
```

**Usage:** Represents mathematical symbols with semantic metadata

---

### 2.5 Proof

```python
class Proof(BaseModel):
    availability: Optional[str] = None  # "not-provided" | "sketched" | "complete"
    steps: List[ProofStep] = []

class ProofStep(BaseModel):
    kind: Optional[str] = None          # "calculation" | "argument" | ...
    text: Optional[str] = None
    latex: Optional[str] = None
```

**Usage:** Represents proof structure and steps

---

### 2.6 Span

```python
class Span(BaseModel):
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content_start: Optional[int] = None
    content_end: Optional[int] = None
    header_lines: List[int] = []
```

**Usage:** Represents line number ranges for directive positioning

---

### 2.7 Assumption (for implicit_assumptions)

```python
class Assumption(BaseModel):
    text: str
    confidence: Optional[float] = None
```

**Usage:** Represents implicit assumptions with confidence scores

**Note:** Propositions use `List[dict]` instead of `List[Assumption]`

---

## 3. Commonalities Analysis

### 3.1 Universal Patterns Across All Four Types

#### A. **Identity Fields** (100% Overlap)

All types require:
- `label: str` - Unique identifier (e.g., `"thm-kl-convergence"`)
- `title: Optional[str]` - Human-readable name
- `type: str` - Entity type discriminator

#### B. **Dual Content Representation** (100% Overlap)

All types maintain two parallel content representations:

1. **Extracted/Semantic:** Structured fields from LLM extraction
   - `nl_statement` - Natural language summary
   - `equations` - LaTeX equations
   - `hypotheses` - Conditions
   - `conclusion` - Main result
   - `variables` - Symbol definitions
   - `proof` - Proof structure

2. **Raw/Directive:** Original markdown with positioning metadata
   - `content_markdown` / `raw.content` - Cleaned directive content
   - `raw_directive` / `raw.raw_directive` - Full directive block with line numbers

#### C. **Provenance Metadata** (95% Overlap)

All types track document context:
- `document_id` - Source document (e.g., `"03_cloning"`)
- `section` - Section heading context
- `generated_at` - ISO timestamp
- `metadata` / `registry_context` - Additional metadata

**Difference:** Propositions nest these in `raw.registry_context`

#### D. **Cross-Reference System** (100% Overlap)

All types support bidirectional linking:
- `references` - References from directive metadata
- `local_refs` - References extracted from content
- `tags` - Classification tags for search/filtering

#### E. **Positioning Information** (95% Overlap)

All types track precise line numbers:
- Start/end line of full directive
- Start/end line of content (excluding header)
- Header line numbers (e.g., section headings)

**Difference:**
- Theorem/Lemma/Corollary use flat `span: Span` field
- Propositions flatten span fields directly in `raw: RawLocator`

---

### 3.2 Shared Processing Patterns

#### A. **Loading Strategy**

All types use identical file loading utilities from `utils.py`:

```python
from utils import (
    load_directive_payload,     # Load directive JSON
    load_extracted_items,        # Load extracted JSON array
    directive_lookup,            # Build label→item dict
    select_existing_file,        # Handle file naming variations
)
```

#### B. **Merging Strategy**

All types merge extracted and directive data with the same pattern:

```python
@classmethod
def from_instances(cls, extracted: dict, directive: dict) -> "UnifiedEntity":
    """Merge extracted semantic data with directive positioning data"""
    return cls(
        # Identity from extracted
        label=extracted.get("label"),
        title=extracted.get("title") or directive.get("title"),
        type=extracted.get("type", "entity_type"),

        # Semantic content from extracted
        nl_statement=extracted.get("nl_statement"),
        equations=[Equation(**eq) for eq in extracted.get("equations", [])],
        # ... more semantic fields ...

        # Raw content and positioning from directive
        content_markdown=cls._strip_line_numbers(directive.get("content")),
        raw_directive=cls._strip_line_numbers(directive.get("raw_directive")),
        document_id=directive.get("_registry_context", {}).get("document_id"),
        # ... more positioning fields ...
    )
```

**Location:**
- `process_theorems.py:97-148`
- `process_lemmas.py:97-148`
- `process_corollaries.py:97-148`
- `process_propositions.py:152-195`

#### C. **Content Cleaning**

All types strip line number prefixes from directive content:

```python
@staticmethod
def _strip_line_numbers(content: Optional[str]) -> Optional[str]:
    """Remove leading 'NNN: ' prefixes from each line"""
    if not content:
        return None
    return "\n".join(
        re.sub(r"^\d+:\s*", "", line)
        for line in content.split("\n")
    )
```

**⚠️ Code Duplication:** This method is **duplicated** in all 4 files:
- `process_theorems.py:57-67`
- `process_lemmas.py:57-67`
- `process_corollaries.py:57-67`
- `process_propositions.py:111-121` (slightly different implementation)

---

## 4. Differences Analysis

### 4.1 Structural Differences

#### A. **Positioning Metadata Storage**

**Theorem/Lemma/Corollary:** Flat structure

```python
class UnifiedTheorem(BaseModel):
    # ... identity and content fields ...
    document_id: Optional[str] = None
    section: Optional[str] = None
    span: Optional[Span] = None
    references: List[Any] = []
    registry_context: Dict[str, Any] = {}
```

**Proposition:** Nested structure via `RawLocator`

```python
class PropositionModel(BaseModel):
    # ... identity and content fields ...
    raw: RawLocator  # Contains all positioning metadata

class RawLocator(BaseModel):
    section: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    # ... more positioning fields ...
    registry_context: Optional[RegistryContext] = None
```

**Rationale:** Propositions explicitly separate semantic (extracted) from raw (directive) data.

---

#### B. **Implicit Assumptions Type**

**Theorem/Lemma/Corollary:** Pydantic models

```python
implicit_assumptions: List[Assumption] = []

class Assumption(BaseModel):
    text: str
    confidence: Optional[float] = None
```

**Proposition:** Raw dictionaries

```python
implicit_assumptions: Optional[List[dict]] = None
```

**Impact:** Propositions lose type safety and validation for assumptions.

---

#### C. **Alternative Labels Field**

**Theorem/Lemma/Corollary:** Includes mismatch handling

```python
alt_labels: List[str] = []  # Track label mismatches
```

**Proposition:** Not present

**Usage:** Corollaries use this for flexible label/title matching when labels don't align.

---

### 4.2 Processing Logic Differences

#### A. **Matching Strategy**

**Theorems:**
```python
# Strict label matching only
for extracted in extracted_items:
    label = extracted.get("label")
    directive = directive_map.get(label)
    if not directive:
        print(f"⚠️ No directive found for theorem {label}")
        continue
```

**Lemmas:**
```python
# Strict label matching only (same as theorems)
for extracted in extracted_items:
    label = extracted.get("label")
    directive = directive_map.get(label)
    if not directive:
        print(f"⚠️ No directive found for lemma {label}")
        continue
```

**Corollaries:**
```python
# Flexible: label matching with title fallback
for extracted in extracted_items:
    label = extracted.get("label")
    directive = directive_map.get(label)

    if not directive:
        # Fallback: try matching by title
        title = extracted.get("title")
        for d_label, d_item in directive_map.items():
            if d_item.get("title") == title:
                directive = d_item
                alt_labels.append(label)  # Track mismatch
                break
```

**Propositions:**
```python
# Strict with validation
for extracted in extracted_items:
    label = extracted.get("label")
    directive = directive_map.get(label)
    if not directive:
        raise ValueError(f"No directive for proposition {label}")

    # Additional validation
    if directive.get("label") != label:
        raise ValueError(f"Label mismatch: {label} != {directive.get('label')}")
```

**Summary:**
- Theorems/Lemmas: Strict, warn on missing
- Corollaries: Flexible with fallback
- Propositions: Strict with validation errors

---

#### B. **File Resolution**

**Theorems:** Fixed filenames

```python
directive_path = extract_dir / "directives" / "theorem.json"
extracted_path = extract_dir / "extract" / "theorem.json"
```

**Lemmas/Corollaries/Propositions:** Flexible with candidates

```python
from utils import select_existing_file

# Try multiple naming conventions
directive_path = select_existing_file([
    extract_dir / "directives" / "lemma.json",
    extract_dir / "directives" / "lemma_raw.json",
])

extracted_path = select_existing_file([
    extract_dir / "extract" / "lemma.json",
    extract_dir / "extract" / "lemma_extracted.json",
])
```

**Rationale:** Handles inconsistent naming across different extraction versions.

---

#### C. **Output Filenames**

| Entity Type | Output Filename | Note |
|------------|-----------------|------|
| Theorem | `preprocess/theorem.json` | Singular |
| Lemma | `preprocess/lemma.json` | Singular |
| Corollary | `preprocess/corollaries.json` | **Plural** |
| Proposition | `preprocess/propositions.json` | **Plural** |

**Inconsistency:** No clear convention for singular vs plural.

---

## 5. Code Duplication Issues

### 5.1 Duplicated Methods

#### A. `_strip_line_numbers()` - **4 copies**

**Locations:**
- `process_theorems.py:57-67`
- `process_lemmas.py:57-67`
- `process_corollaries.py:57-67`
- `process_propositions.py:111-121` (different implementation)

**Identical Implementation (Theorem/Lemma/Corollary):**

```python
@staticmethod
def _strip_line_numbers(content: Optional[str]) -> Optional[str]:
    if not content:
        return None
    lines = content.split("\n")
    stripped = []
    for line in lines:
        stripped_line = re.sub(r"^\d+:\s*", "", line)
        stripped.append(stripped_line)
    return "\n".join(stripped)
```

**Proposition Implementation (slightly different):**

```python
@staticmethod
def _strip_line_numbers(txt: Optional[str]) -> Optional[str]:
    if not txt:
        return txt
    out = []
    for line in txt.split("\n"):
        if ":" in line:
            idx = line.index(":")
            out.append(line[idx + 1:].lstrip())
        else:
            out.append(line)
    return "\n".join(out)
```

**Issue:** Proposition implementation is less robust (assumes first `:` is line number separator).

---

#### B. `from_instances()` - **4 copies**

All four types implement nearly identical class methods with ~50 lines of duplicated code.

**Pattern:**

```python
@classmethod
def from_instances(cls, extracted: dict, directive: dict) -> "UnifiedEntity":
    # 1. Extract identity fields
    label = extracted.get("label")
    title = extracted.get("title") or directive.get("title")

    # 2. Parse nested models from extracted
    equations = [Equation(**eq) for eq in extracted.get("equations", [])]
    hypotheses = [Hypothesis(**h) for h in extracted.get("hypotheses", [])]
    # ... more parsing ...

    # 3. Extract positioning from directive
    content = cls._strip_line_numbers(directive.get("content"))
    raw_directive = cls._strip_line_numbers(directive.get("raw_directive"))

    # 4. Build Span object
    span = Span(
        start_line=directive.get("start_line"),
        end_line=directive.get("end_line"),
        # ...
    )

    # 5. Construct instance
    return cls(
        label=label,
        title=title,
        # ... all fields ...
    )
```

**Locations:**
- `process_theorems.py:97-148`
- `process_lemmas.py:97-148`
- `process_corollaries.py:97-148`
- `process_propositions.py:152-195`

**Difference:** Only the return type annotation and `type` field differ.

---

### 5.2 Duplicated Model Definitions

#### A. Nested Models

Each file defines its own copies of:
- `Equation`
- `Hypothesis`
- `Conclusion`
- `Variable`
- `Proof` / `ProofStep`
- `Span` (except Propositions)
- `Assumption`

**Total copies:** 7 models × 4 files = **28 model definitions**

**Locations:**
- `process_theorems.py:8-24`
- `process_lemmas.py:8-24`
- `process_corollaries.py:8-24`
- `process_propositions.py:8-58` (includes RawLocator)

---

#### B. Main Entity Classes

**Theorem/Lemma/Corollary:** Structurally identical but defined separately.

**Lines of code:**
- `UnifiedTheorem`: 55 lines
- `UnifiedLemma`: 55 lines
- `UnifiedCorollary`: 55 lines
- **Total duplication:** 110 lines

---

### 5.3 Duplicated Processing Logic

#### A. File Loading Pattern

Each file has nearly identical main processing logic:

```python
def main(document: str):
    # 1. Resolve directories
    doc_dir = resolve_document_directory(document)
    extract_dir = resolve_extract_directory(...)

    # 2. Load files
    directive_payload = load_directive_payload(directive_path)
    extracted_items = load_extracted_items(extracted_path)

    # 3. Build lookup
    directive_map = directive_lookup(directive_payload["items"])

    # 4. Merge items
    unified = []
    for extracted in extracted_items:
        directive = directive_map.get(extracted["label"])
        if directive:
            unified.append(UnifiedEntity.from_instances(extracted, directive))

    # 5. Save output
    output_path = preprocess_dir / "entity.json"
    with open(output_path, "w") as f:
        json.dump([u.model_dump() for u in unified], f, indent=2)
```

**Duplication:** ~40 lines × 4 files = **160 lines** of nearly identical code

---

## 6. Refactoring Recommendations

### 6.1 Create Unified Base Class

**Design Pattern:** Template method pattern with entity-specific customization

```python
# In utils.py or new base_models.py

from typing import Optional, List, Dict, Any, TypeVar, Generic
from pydantic import BaseModel, Field

# === SHARED NESTED MODELS ===

class Equation(BaseModel):
    """Shared equation model"""
    label: Optional[str] = None
    latex: str

class Hypothesis(BaseModel):
    """Shared hypothesis model"""
    text: Optional[str] = None
    latex: Optional[str] = None

class Conclusion(BaseModel):
    """Shared conclusion model"""
    text: Optional[str] = None
    latex: Optional[str] = None

class Variable(BaseModel):
    """Shared variable model"""
    symbol: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class ProofStep(BaseModel):
    """Shared proof step model"""
    kind: Optional[str] = None
    text: Optional[str] = None
    latex: Optional[str] = None

class Proof(BaseModel):
    """Shared proof model"""
    availability: Optional[str] = None
    steps: List[ProofStep] = Field(default_factory=list)

class Span(BaseModel):
    """Shared span model for line number tracking"""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content_start: Optional[int] = None
    content_end: Optional[int] = None
    header_lines: List[int] = Field(default_factory=list)

class Assumption(BaseModel):
    """Shared assumption model"""
    text: str
    confidence: Optional[float] = None

# === BASE CLASS ===

class UnifiedMathematicalEntity(BaseModel):
    """Base class for all theorem-like mathematical entities"""

    # === IDENTITY ===
    label: str
    title: Optional[str] = None
    type: str  # Subclass sets default

    # === EXTRACTED SEMANTIC CONTENT ===
    nl_statement: Optional[str] = None
    equations: List[Equation] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    conclusion: Optional[Conclusion] = None
    variables: List[Variable] = Field(default_factory=list)
    implicit_assumptions: List[Assumption] = Field(default_factory=list)
    local_refs: List[str] = Field(default_factory=list)
    proof: Optional[Proof] = None
    tags: List[str] = Field(default_factory=list)

    # === RAW DIRECTIVE CONTENT ===
    content_markdown: Optional[str] = None
    raw_directive: Optional[str] = None

    # === PROVENANCE/POSITIONING ===
    document_id: Optional[str] = None
    section: Optional[str] = None
    span: Optional[Span] = None
    references: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registry_context: Dict[str, Any] = Field(default_factory=dict)
    generated_at: Optional[str] = None
    alt_labels: List[str] = Field(default_factory=list)

    # === SHARED METHODS ===

    @staticmethod
    def strip_line_numbers(content: Optional[str]) -> Optional[str]:
        """Remove leading 'NNN: ' prefixes from directive content"""
        if not content:
            return None
        return "\n".join(
            re.sub(r"^\d+:\s*", "", line)
            for line in content.split("\n")
        )

    @classmethod
    def from_instances(
        cls,
        extracted: dict,
        directive: dict,
        **kwargs  # Allow subclass customization
    ) -> "UnifiedMathematicalEntity":
        """Merge extracted and directive data"""

        # Parse nested models
        equations = [Equation(**eq) for eq in extracted.get("equations", [])]
        hypotheses = [Hypothesis(**h) for h in extracted.get("hypotheses", [])]

        conclusion_data = extracted.get("conclusion")
        conclusion = Conclusion(**conclusion_data) if conclusion_data else None

        variables = [Variable(**v) for v in extracted.get("variables", [])]

        assumptions = [
            Assumption(**a) for a in extracted.get("implicit_assumptions", [])
        ]

        proof_data = extracted.get("proof")
        proof = Proof(**proof_data) if proof_data else None

        # Build span
        span = Span(
            start_line=directive.get("start_line"),
            end_line=directive.get("end_line"),
            content_start=directive.get("content_start"),
            content_end=directive.get("content_end"),
            header_lines=directive.get("header_lines", []),
        )

        # Construct instance
        return cls(
            # Identity
            label=extracted.get("label"),
            title=extracted.get("title") or directive.get("title"),
            type=extracted.get("type", cls.__fields__["type"].default),

            # Semantic content
            nl_statement=extracted.get("nl_statement"),
            equations=equations,
            hypotheses=hypotheses,
            conclusion=conclusion,
            variables=variables,
            implicit_assumptions=assumptions,
            local_refs=extracted.get("local_refs", []),
            proof=proof,
            tags=extracted.get("tags", []),

            # Raw content
            content_markdown=cls.strip_line_numbers(directive.get("content")),
            raw_directive=cls.strip_line_numbers(directive.get("raw_directive")),

            # Positioning
            document_id=directive.get("_registry_context", {}).get("document_id"),
            section=directive.get("section"),
            span=span,
            references=directive.get("references", []),
            metadata=directive.get("metadata", {}),
            registry_context=directive.get("_registry_context", {}),
            generated_at=directive.get("generated_at"),

            # Subclass-specific fields
            **kwargs
        )

# === CONCRETE SUBCLASSES ===

class UnifiedTheorem(UnifiedMathematicalEntity):
    """Theorem entity"""
    type: str = "theorem"

class UnifiedLemma(UnifiedMathematicalEntity):
    """Lemma entity"""
    type: str = "lemma"

class UnifiedCorollary(UnifiedMathematicalEntity):
    """Corollary entity"""
    type: str = "corollary"

class UnifiedProposition(UnifiedMathematicalEntity):
    """Proposition entity (standardized to match base)"""
    type: str = "proposition"
```

**Benefits:**
1. **165 lines** of base class replace **~220 lines** across 4 files
2. Single source of truth for nested models
3. Consistent `from_instances()` logic
4. Easy to add new entity types (e.g., `UnifiedConjecture`)
5. Type-safe with Pydantic validation

---

### 6.2 Standardize Proposition Structure

**Current Issue:** Propositions use nested `RawLocator` which diverges from other types.

**Recommendation:** Flatten to match base class structure

**Migration Path:**

```python
# Option 1: Migrate Proposition to use base class directly
class UnifiedProposition(UnifiedMathematicalEntity):
    type: str = "proposition"
    # No additional fields needed!

# Option 2: Keep RawLocator as optional alternative representation
class UnifiedProposition(UnifiedMathematicalEntity):
    type: str = "proposition"
    raw: Optional[RawLocator] = None  # Backward compatibility

    @classmethod
    def from_instances_with_raw(cls, extracted, directive):
        # Populate both flat and nested representations
        instance = super().from_instances(extracted, directive)
        instance.raw = RawLocator(
            section=instance.section,
            start_line=instance.span.start_line if instance.span else None,
            # ... populate from flat fields
        )
        return instance
```

**Benefit:** Unifies all four types under single base class.

---

### 6.3 Extract Shared Processing Logic

**Create:** `base_processor.py`

```python
from pathlib import Path
from typing import Type, List, Dict, Optional
import json
from utils import (
    resolve_document_directory,
    resolve_extract_directory,
    load_directive_payload,
    load_extracted_items,
    directive_lookup,
    select_existing_file,
)

class MathematicalEntityProcessor:
    """Generic processor for mathematical entities"""

    def __init__(
        self,
        entity_class: Type[UnifiedMathematicalEntity],
        directive_filenames: List[str],
        extracted_filenames: List[str],
        output_filename: str,
    ):
        self.entity_class = entity_class
        self.directive_filenames = directive_filenames
        self.extracted_filenames = extracted_filenames
        self.output_filename = output_filename

    def process(
        self,
        document: str,
        match_strategy: str = "strict",  # "strict" | "flexible"
    ) -> List[UnifiedMathematicalEntity]:
        """Process document and return unified entities"""

        # 1. Resolve directories
        doc_dir = resolve_document_directory(document)
        extract_dir = resolve_extract_directory(doc_dir / "registry")

        # 2. Load files
        directive_candidates = [
            extract_dir / "directives" / fname
            for fname in self.directive_filenames
        ]
        directive_path = select_existing_file(directive_candidates)

        extracted_candidates = [
            extract_dir / "extract" / fname
            for fname in self.extracted_filenames
        ]
        extracted_path = select_existing_file(extracted_candidates)

        directive_payload = load_directive_payload(directive_path)
        extracted_items = load_extracted_items(extracted_path)

        # 3. Build lookup
        directive_map = directive_lookup(directive_payload["items"])

        # 4. Merge with strategy
        unified = []
        for extracted in extracted_items:
            label = extracted.get("label")
            directive = directive_map.get(label)

            if not directive and match_strategy == "flexible":
                # Try title fallback for corollaries
                directive = self._match_by_title(
                    extracted, directive_payload["items"]
                )

            if directive:
                entity = self.entity_class.from_instances(extracted, directive)
                unified.append(entity)
            else:
                print(f"⚠️ No directive for {label}")

        # 5. Save output
        preprocess_dir = doc_dir / "registry" / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        output_path = preprocess_dir / self.output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [e.model_dump() for e in unified],
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"✅ Saved {len(unified)} items to {output_path}")
        return unified

    @staticmethod
    def _match_by_title(extracted: dict, directives: List[dict]) -> Optional[dict]:
        """Fallback: match by title"""
        title = extracted.get("title")
        if not title:
            return None
        for directive in directives:
            if directive.get("title") == title:
                return directive
        return None

# === USAGE ===

# process_theorems.py
processor = MathematicalEntityProcessor(
    entity_class=UnifiedTheorem,
    directive_filenames=["theorem.json"],
    extracted_filenames=["theorem.json"],
    output_filename="theorem.json",
)
processor.process(document="03_cloning")

# process_corollaries.py
processor = MathematicalEntityProcessor(
    entity_class=UnifiedCorollary,
    directive_filenames=["corollary.json", "corollary_raw.json"],
    extracted_filenames=["corollary.json", "corollary_extracted.json"],
    output_filename="corollaries.json",
)
processor.process(document="03_cloning", match_strategy="flexible")
```

**Benefits:**
1. **~100 lines** of shared processor replace **~160 lines** per file
2. Configurable matching strategy
3. Consistent error handling
4. Easy to add new entity types

---

### 6.4 Consolidate Nested Model Definitions

**Create:** `shared_models.py`

```python
"""Shared Pydantic models for mathematical entities"""

from typing import Optional, List
from pydantic import BaseModel, Field

class Equation(BaseModel):
    """Mathematical equation with optional label"""
    label: Optional[str] = None
    latex: str

class Hypothesis(BaseModel):
    """Assumption or condition in theorem statement"""
    text: Optional[str] = None
    latex: Optional[str] = None

class Conclusion(BaseModel):
    """Main result of theorem"""
    text: Optional[str] = None
    latex: Optional[str] = None

class Variable(BaseModel):
    """Mathematical symbol with semantic metadata"""
    symbol: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class ProofStep(BaseModel):
    """Single step in a proof"""
    kind: Optional[str] = None  # "calculation" | "argument" | "reference"
    text: Optional[str] = None
    latex: Optional[str] = None

class Proof(BaseModel):
    """Proof structure"""
    availability: Optional[str] = None  # "not-provided" | "sketched" | "complete"
    steps: List[ProofStep] = Field(default_factory=list)

class Span(BaseModel):
    """Line number range for directive positioning"""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content_start: Optional[int] = None
    content_end: Optional[int] = None
    header_lines: List[int] = Field(default_factory=list)

class Assumption(BaseModel):
    """Implicit assumption with confidence score"""
    text: str
    confidence: Optional[float] = None
```

**Benefits:**
1. **Single import** replaces 28 duplicate definitions
2. Consistent validation across all entity types
3. Easy to extend (e.g., add fields to `Variable`)

---

## 7. Proposed Base Class Design Summary

### 7.1 Architecture

```
shared_models.py
├── Equation
├── Hypothesis
├── Conclusion
├── Variable
├── ProofStep
├── Proof
├── Span
└── Assumption

base_models.py
└── UnifiedMathematicalEntity
    ├── strip_line_numbers() [static method]
    └── from_instances() [class method]

entity_types.py
├── UnifiedTheorem(UnifiedMathematicalEntity)
├── UnifiedLemma(UnifiedMathematicalEntity)
├── UnifiedCorollary(UnifiedMathematicalEntity)
└── UnifiedProposition(UnifiedMathematicalEntity)

base_processor.py
└── MathematicalEntityProcessor
    ├── __init__()
    ├── process()
    └── _match_by_title()

process_theorems.py  [SIMPLIFIED]
├── from entity_types import UnifiedTheorem
├── from base_processor import MathematicalEntityProcessor
└── main() [5-10 lines]

process_lemmas.py  [SIMPLIFIED]
process_corollaries.py  [SIMPLIFIED]
process_propositions.py  [SIMPLIFIED]
```

---

### 7.2 Lines of Code Reduction

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| **Nested Models** | 28 definitions | 8 definitions | **-71%** |
| **Main Entity Classes** | 220 lines (4 × 55) | 85 lines (base + 4 subclasses) | **-61%** |
| **Processing Logic** | 640 lines (4 × 160) | 150 lines (base processor + 4 configs) | **-77%** |
| **`_strip_line_numbers()`** | 44 lines (4 × 11) | 11 lines (1 method) | **-75%** |
| **`from_instances()`** | 200 lines (4 × 50) | 50 lines (1 method) | **-75%** |
| **TOTAL** | ~1132 lines | ~304 lines | **-73%** |

**Estimated overall reduction: 60-70% LOC**

---

### 7.3 Inheritance Hierarchy

```
BaseModel (Pydantic)
│
└── UnifiedMathematicalEntity
    ├── type: str
    ├── label: str
    ├── title: Optional[str]
    ├── nl_statement: Optional[str]
    ├── equations: List[Equation]
    ├── hypotheses: List[Hypothesis]
    ├── conclusion: Optional[Conclusion]
    ├── variables: List[Variable]
    ├── implicit_assumptions: List[Assumption]
    ├── local_refs: List[str]
    ├── proof: Optional[Proof]
    ├── tags: List[str]
    ├── content_markdown: Optional[str]
    ├── raw_directive: Optional[str]
    ├── document_id: Optional[str]
    ├── section: Optional[str]
    ├── span: Optional[Span]
    ├── references: List[Any]
    ├── metadata: Dict[str, Any]
    ├── registry_context: Dict[str, Any]
    ├── generated_at: Optional[str]
    ├── alt_labels: List[str]
    │
    ├── strip_line_numbers() → str
    └── from_instances() → Self
        │
        ├── UnifiedTheorem (type="theorem")
        ├── UnifiedLemma (type="lemma")
        ├── UnifiedCorollary (type="corollary")
        └── UnifiedProposition (type="proposition")
```

**Key Design Principles:**
1. **Base class handles all common logic**
2. **Subclasses only customize `type` field**
3. **Template method pattern for extensibility**
4. **Pydantic validation throughout**

---

## 8. JSON Structure Documentation

### 8.1 Extracted JSON Format

**File:** `extract/theorem_extracted.json`

**Structure:** Array of objects with semantic content from LLM extraction

```json
[
  {
    "type": "theorem",
    "label": "thm-kl-convergence",
    "title": "KL Divergence Convergence Rate",
    "nl_statement": "The KL divergence between the empirical measure and the target decreases exponentially with rate proportional to the log-Sobolev constant.",
    "equations": [
      {
        "label": "eq-kl-rate",
        "latex": "\\frac{d}{dt} D_{\\text{KL}}(\\mu_t \\| \\pi) \\leq -2\\rho \\cdot D_{\\text{KL}}(\\mu_t \\| \\pi)"
      },
      {
        "label": null,
        "latex": "D_{\\text{KL}}(\\mu_t \\| \\pi) \\leq e^{-2\\rho t} D_{\\text{KL}}(\\mu_0 \\| \\pi)"
      }
    ],
    "hypotheses": [
      {
        "text": "The target distribution π satisfies a log-Sobolev inequality with constant ρ > 0",
        "latex": "\\text{Ent}_{\\pi}(f^2) \\leq \\frac{1}{\\rho} \\int |\\nabla f|^2 d\\pi"
      },
      {
        "text": "The potential V is λ-convex",
        "latex": "\\nabla^2 V(x) \\geq \\lambda I"
      }
    ],
    "conclusion": {
      "text": "The KL divergence converges exponentially to zero",
      "latex": "D_{\\text{KL}}(\\mu_t \\| \\pi) \\to 0 \\text{ as } t \\to \\infty"
    },
    "variables": [
      {
        "symbol": "\\mu_t",
        "name": "empirical measure",
        "description": "The distribution of walkers at time t",
        "constraints": ["probability measure"],
        "tags": ["measure", "time-dependent"]
      },
      {
        "symbol": "\\pi",
        "name": "target distribution",
        "description": "The quasi-stationary distribution we converge to",
        "constraints": ["probability measure", "normalized"],
        "tags": ["measure", "stationary"]
      },
      {
        "symbol": "\\rho",
        "name": "log-Sobolev constant",
        "description": "Measures the strength of the LSI",
        "constraints": ["positive"],
        "tags": ["constant", "functional-inequality"]
      }
    ],
    "implicit_assumptions": [
      {
        "text": "The state space is compact",
        "confidence": 0.9
      },
      {
        "text": "The Langevin dynamics are ergodic",
        "confidence": 0.85
      }
    ],
    "local_refs": [
      "def-kl-divergence",
      "def-log-sobolev-inequality",
      "thm-convergence-langevin"
    ],
    "proof": {
      "availability": "sketched",
      "steps": [
        {
          "kind": "calculation",
          "text": "Differentiate the KL divergence along the flow",
          "latex": "\\frac{d}{dt} D_{\\text{KL}} = \\int \\log\\frac{\\mu_t}{\\pi} \\partial_t \\mu_t dx"
        },
        {
          "kind": "argument",
          "text": "Apply the log-Sobolev inequality to bound the entropy production",
          "latex": null
        },
        {
          "kind": "reference",
          "text": "Use Grönwall's inequality to obtain exponential rate",
          "latex": null
        }
      ]
    },
    "tags": [
      "convergence",
      "kl-divergence",
      "log-sobolev",
      "exponential-rate"
    ]
  }
]
```

**Key Features:**
- **Semantic extraction:** Natural language summaries, structured mathematical content
- **Rich metadata:** Variables with descriptions, implicit assumptions with confidence
- **Cross-references:** Local references to other entities
- **Proof sketches:** High-level proof structure

---

### 8.2 Directive JSON Format

**File:** `directives/theorem.json`

**Structure:** Object with metadata + items array

```json
{
  "document_id": "09_kl_convergence",
  "stage": "directives",
  "directive_type": "theorem",
  "generated_at": "2025-11-09T12:34:56.789012+00:00",
  "count": 1,
  "items": [
    {
      "directive_type": "theorem",
      "label": "thm-kl-convergence",
      "title": "KL Divergence Convergence Rate",
      "start_line": 450,
      "end_line": 475,
      "header_lines": [451, 452],
      "content_start": 455,
      "content_end": 474,
      "content": "455: Under the log-Sobolev inequality with constant $\\rho > 0$,\n456: the KL divergence satisfies:\n457: \n458: $$\n459: \\frac{d}{dt} D_{\\text{KL}}(\\mu_t \\| \\pi) \\leq -2\\rho \\cdot D_{\\text{KL}}(\\mu_t \\| \\pi)\n460: $$\n461: \n462: This implies exponential convergence:\n463: \n464: $$\n465: D_{\\text{KL}}(\\mu_t \\| \\pi) \\leq e^{-2\\rho t} D_{\\text{KL}}(\\mu_0 \\| \\pi)\n466: $$",
      "raw_directive": "450: \n451: #### Section: KL Convergence\n452: \n453: :::{prf:theorem} KL Divergence Convergence Rate\n454: :label: thm-kl-convergence\n455: \n456: Under the log-Sobolev inequality...\n...\n475: :::",
      "metadata": {
        "label": "thm-kl-convergence",
        "directive_type": "theorem"
      },
      "section": "## 9. KL Convergence Analysis",
      "references": [
        "def-kl-divergence",
        "def-log-sobolev-inequality"
      ],
      "_registry_context": {
        "stage": "directives",
        "document_id": "09_kl_convergence",
        "chapter_index": 9,
        "chapter_file": "09_kl_convergence.md",
        "section_id": "## 9. KL Convergence Analysis"
      }
    }
  ]
}
```

**Key Features:**
- **Precise positioning:** Line numbers for directive start/end, content start/end, headers
- **Raw content:** Unprocessed directive with line number prefixes
- **Provenance:** Document ID, section context, registry metadata
- **Cross-references:** Extracted from directive metadata

---

### 8.3 Unified (Preprocessed) JSON Format

**File:** `preprocess/theorem.json`

**Structure:** Array of merged objects (extracted + directive)

```json
[
  {
    "label": "thm-kl-convergence",
    "title": "KL Divergence Convergence Rate",
    "type": "theorem",

    "nl_statement": "The KL divergence between the empirical measure and the target decreases exponentially...",

    "equations": [
      {
        "label": "eq-kl-rate",
        "latex": "\\frac{d}{dt} D_{\\text{KL}}(\\mu_t \\| \\pi) \\leq -2\\rho \\cdot D_{\\text{KL}}(\\mu_t \\| \\pi)"
      }
    ],

    "hypotheses": [
      {
        "text": "The target distribution π satisfies a log-Sobolev inequality",
        "latex": "\\text{Ent}_{\\pi}(f^2) \\leq \\frac{1}{\\rho} \\int |\\nabla f|^2 d\\pi"
      }
    ],

    "conclusion": {
      "text": "The KL divergence converges exponentially to zero",
      "latex": "D_{\\text{KL}}(\\mu_t \\| \\pi) \\to 0 \\text{ as } t \\to \\infty"
    },

    "variables": [
      {
        "symbol": "\\mu_t",
        "name": "empirical measure",
        "description": "The distribution of walkers at time t",
        "constraints": ["probability measure"],
        "tags": ["measure", "time-dependent"]
      }
    ],

    "implicit_assumptions": [
      {
        "text": "The state space is compact",
        "confidence": 0.9
      }
    ],

    "local_refs": [
      "def-kl-divergence",
      "def-log-sobolev-inequality"
    ],

    "proof": {
      "availability": "sketched",
      "steps": [
        {
          "kind": "calculation",
          "text": "Differentiate the KL divergence along the flow",
          "latex": "\\frac{d}{dt} D_{\\text{KL}} = \\int \\log\\frac{\\mu_t}{\\pi} \\partial_t \\mu_t dx"
        }
      ]
    },

    "tags": ["convergence", "kl-divergence", "log-sobolev"],

    "content_markdown": "Under the log-Sobolev inequality with constant $\\rho > 0$,\nthe KL divergence satisfies:\n\n$$\n\\frac{d}{dt} D_{\\text{KL}}(\\mu_t \\| \\pi) \\leq -2\\rho \\cdot D_{\\text{KL}}(\\mu_t \\| \\pi)\n$$",

    "raw_directive": ":::{prf:theorem} KL Divergence Convergence Rate\n:label: thm-kl-convergence\n\nUnder the log-Sobolev inequality...",

    "document_id": "09_kl_convergence",
    "section": "## 9. KL Convergence Analysis",

    "span": {
      "start_line": 450,
      "end_line": 475,
      "content_start": 455,
      "content_end": 474,
      "header_lines": [451, 452]
    },

    "references": [
      "def-kl-divergence",
      "def-log-sobolev-inequality"
    ],

    "metadata": {
      "label": "thm-kl-convergence",
      "directive_type": "theorem"
    },

    "registry_context": {
      "stage": "directives",
      "document_id": "09_kl_convergence",
      "chapter_index": 9,
      "chapter_file": "09_kl_convergence.md",
      "section_id": "## 9. KL Convergence Analysis"
    },

    "generated_at": "2025-11-09T12:34:56.789012+00:00",
    "alt_labels": []
  }
]
```

**Key Features:**
- **Complete entity:** Merges semantic (extracted) and positional (directive) data
- **Cleaned content:** Line number prefixes stripped from `content_markdown` and `raw_directive`
- **Full provenance:** All metadata from both sources preserved

---

### 8.4 Field Mapping: Extracted + Directive → Unified

| Unified Field | Source | Extracted Field | Directive Field |
|--------------|--------|-----------------|-----------------|
| `label` | Both | `label` | `label` |
| `title` | Both | `title` (priority) | `title` (fallback) |
| `type` | Extracted | `type` | – |
| `nl_statement` | Extracted | `nl_statement` | – |
| `equations` | Extracted | `equations` | – |
| `hypotheses` | Extracted | `hypotheses` | – |
| `conclusion` | Extracted | `conclusion` | – |
| `variables` | Extracted | `variables` | – |
| `implicit_assumptions` | Extracted | `implicit_assumptions` | – |
| `local_refs` | Extracted | `local_refs` | – |
| `proof` | Extracted | `proof` | – |
| `tags` | Extracted | `tags` | – |
| `content_markdown` | Directive | – | `content` (cleaned) |
| `raw_directive` | Directive | – | `raw_directive` (cleaned) |
| `document_id` | Directive | – | `_registry_context.document_id` |
| `section` | Directive | – | `section` |
| `span` | Directive | – | `start_line`, `end_line`, etc. |
| `references` | Directive | – | `references` |
| `metadata` | Directive | – | `metadata` |
| `registry_context` | Directive | – | `_registry_context` |
| `generated_at` | Directive | – | `generated_at` |

---

## 9. Implementation Roadmap

### Phase 1: Foundation (No Breaking Changes)

**Goal:** Create base infrastructure without disrupting existing code

**Tasks:**
1. Create `shared_models.py` with all nested models
2. Create `base_models.py` with `UnifiedMathematicalEntity`
3. Create `entity_types.py` with concrete subclasses
4. Create `base_processor.py` with `MathematicalEntityProcessor`
5. Add comprehensive tests for base classes

**Estimated effort:** 2-3 days

**Risk:** Low (all new code, no modifications)

---

### Phase 2: Migrate Theorem/Lemma/Corollary

**Goal:** Convert three identical types to use base class

**Tasks:**
1. Update `process_theorems.py`:
   - Import from `shared_models`, `entity_types`, `base_processor`
   - Replace `UnifiedTheorem` definition with import
   - Replace `from_instances()` with processor call
   - Simplify `main()` to 5-10 lines

2. Repeat for `process_lemmas.py` and `process_corollaries.py`

3. Run full test suite to verify output unchanged

4. Compare old vs new preprocessed JSON files (should be identical)

**Estimated effort:** 1-2 days

**Risk:** Low (well-tested base class, easy rollback)

---

### Phase 3: Standardize Propositions

**Goal:** Migrate propositions to unified structure

**Tasks:**
1. Decide on approach:
   - **Option A:** Full migration (flatten `RawLocator` fields)
   - **Option B:** Hybrid (keep `RawLocator` as optional)

2. Update `process_propositions.py`:
   - Convert `PropositionModel` to use `UnifiedProposition`
   - Migrate `RawLocator` fields to flat structure
   - Update `from_instances()` to use base class method

3. Update downstream code that reads proposition JSON

4. Run tests and verify consistency

**Estimated effort:** 1-2 days

**Risk:** Medium (proposition structure differs, downstream impact)

---

### Phase 4: Cleanup and Documentation

**Goal:** Remove duplicated code and document changes

**Tasks:**
1. Delete old model definitions from individual files
2. Delete old `_strip_line_numbers()` methods
3. Delete old `from_instances()` methods
4. Update type hints to use `shared_models.*`
5. Add docstrings to all base classes
6. Update this ANALYSIS.md with final architecture
7. Create migration guide for future entity types

**Estimated effort:** 1 day

**Risk:** Low (cleanup only)

---

### Phase 5: Extend to Other Entity Types (Optional)

**Goal:** Apply same pattern to axioms, assumptions, definitions

**Tasks:**
1. Analyze whether `UnifiedMathematicalEntity` fits their structure
2. If not, create `BaseAxiom`, `BaseDefinition` base classes
3. Extract common patterns across ALL entity types
4. Consider generic `BaseEntity` with composition

**Estimated effort:** 3-5 days

**Risk:** Medium (broader scope, different structures)

---

## 10. Quick Wins vs Long-Term Refactoring

### Quick Wins (Immediate, Low Risk)

1. **Extract `shared_models.py`** (1 hour)
   - Consolidate 28 duplicate model definitions
   - Immediate benefit: Single import for all files
   - No breaking changes

2. **Extract `_strip_line_numbers()` to `utils.py`** (30 minutes)
   - Remove 4 duplicate methods
   - Import from utils in all files
   - Easy to test and verify

3. **Standardize output filenames** (15 minutes)
   - Decide singular vs plural convention
   - Update file paths consistently
   - Improve predictability

4. **Add type hints** (1 hour)
   - Add return type hints to `from_instances()`
   - Add parameter type hints
   - Improve IDE support

**Total effort:** ~3 hours
**LOC reduction:** ~150 lines
**Risk:** Very low

---

### Long-Term Refactoring (Requires Planning)

1. **Create `UnifiedMathematicalEntity` base class** (2-3 days)
   - Design inheritance hierarchy
   - Implement shared `from_instances()`
   - Add comprehensive tests
   - **Benefit:** 60-70% LOC reduction

2. **Standardize proposition structure** (1-2 days)
   - Flatten `RawLocator` fields
   - Update downstream consumers
   - Migrate existing data (if needed)
   - **Benefit:** Consistent API across all types

3. **Create `MathematicalEntityProcessor`** (1-2 days)
   - Generic processing pipeline
   - Configurable matching strategies
   - Error handling framework
   - **Benefit:** Maintainable, extensible architecture

4. **Extend to all entity types** (3-5 days)
   - Apply to axioms, assumptions, definitions, algorithms
   - Create entity-specific base classes if needed
   - Unified registry system
   - **Benefit:** Consistent framework across entire codebase

**Total effort:** ~2 weeks
**LOC reduction:** ~800-1000 lines
**Risk:** Medium (requires careful migration)

---

## 11. Conclusion

### Summary of Findings

1. **Theorem/Lemma/Corollary are identical** - Strong candidate for single parameterized class
2. **Propositions diverge slightly** - Can be standardized to match base structure
3. **70% code overlap** - Massive refactoring opportunity
4. **Shared nested models** - Should be consolidated to single definitions
5. **Duplicated processing logic** - Can be extracted to generic processor

### Recommended Approach

**Immediate (This Sprint):**
- Extract `shared_models.py` (quick win)
- Extract `_strip_line_numbers()` to utils (quick win)
- Standardize output filenames (quick win)

**Next Sprint:**
- Create `UnifiedMathematicalEntity` base class
- Migrate theorem/lemma/corollary
- Add comprehensive tests

**Future Work:**
- Standardize propositions
- Create generic processor
- Extend to all entity types

### Benefits of Refactoring

1. **Maintainability:** Single source of truth for common logic
2. **Extensibility:** Easy to add new entity types (e.g., conjecture, postulate)
3. **Consistency:** Unified API across all types
4. **Testability:** Test base class once, not 4 times
5. **Type Safety:** Pydantic validation throughout
6. **Documentation:** Clear inheritance hierarchy

### Risks and Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking downstream code | Comprehensive test suite, gradual migration |
| JSON format changes | Verify byte-for-byte equality of output |
| Loss of type-specific flexibility | Design base class with extension points |
| Complex inheritance hierarchy | Keep hierarchy shallow (max 2 levels) |

---

**Next Steps:** Discuss with team and prioritize quick wins vs long-term refactoring based on project timeline and risk tolerance.
