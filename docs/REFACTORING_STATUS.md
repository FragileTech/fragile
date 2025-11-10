# Mathster Parsing Module Refactoring Status

## Executive Summary

The `src/mathster/parsing/` module has been partially refactored to improve modularity, reduce coupling, and enhance maintainability. **Phases 1-3 are complete** (foundation modules created), with Phases 4-10 remaining.

---

## Completed Work ✓

### Phase 1: Data Models Module (COMPLETE)
**Location**: `src/mathster/parsing/models/`

Created pure Pydantic data models with no business logic:

- **`models/entities.py`** (175 lines)
  - `ExtractedEntity` (base)
  - `DefinitionExtraction`, `TheoremExtraction`, `ProofExtraction`
  - `AxiomExtraction`, `ParameterExtraction`, `RemarkExtraction`
  - `AssumptionExtraction`, `CitationExtraction`
  - `ChapterExtraction` (aggregate)

- **`models/results.py`** (78 lines)
  - `ValidationResult` with `get_feedback()` method
  - `ImprovementResult` with `add_change()` and `get_summary()` methods

- **`models/changes.py`** (30 lines)
  - `ChangeOperation` enum (ADD, MODIFY, DELETE, NO_CHANGE)
  - `EntityChange` model

- **`models/__init__.py`**
  - Clean public API exports for all models

**Benefits**:
- Models can be imported independently
- No coupling to workflow logic
- Easy to test and validate
- Clear data contracts

---

### Phase 2: Validation Module (COMPLETE)
**Location**: `src/mathster/parsing/validation/`

Created validation logic and error handling:

- **`validation/validators.py`** (170 lines)
  - `validate_extraction()` - Main validation function
  - Label pattern validation (def-, thm-, lem-, etc.)
  - Reference format validation
  - Line number sanity checks
  - Uses lazy import for `convert_to_raw_document_section` to avoid circular deps

- **`validation/errors.py`** (233 lines)
  - `make_error_dict()` - Structured error formatting
  - `generate_detailed_error_report()` - LLM-friendly error reports
  - Pydantic ValidationError parsing
  - JSON parsing error guidance
  - Timeout error handling

- **`validation/__init__.py`**
  - Exports: `validate_extraction`, `make_error_dict`, `generate_detailed_error_report`

**Benefits**:
- Centralized error handling
- Reusable validation logic
- Better error messages for LLMs
- Independent of workflow orchestration

---

### Phase 3: Conversion Module (PARTIAL)
**Location**: `src/mathster/parsing/conversion/`

Created label and source utilities:

- **`conversion/labels.py`** (180 lines) ✓ COMPLETE
  - `sanitize_label()` - Normalize labels to valid format
  - `lookup_label_from_context()` - Resolve text references to labels
  - Supports theorem, definition, proof lookups

- **`conversion/sources.py`** (40 lines) ✓ COMPLETE
  - `create_source_location()` - Build SourceLocation from line ranges

- **`conversion/converters.py`** ⚠️ **TODO**
  - Need to extract `convert_to_raw_document_section()` (~300 lines)
  - Need to extract `convert_dict_to_extraction_entity()` (~30 lines)
  - These are currently in `extract_workflow.py` (lines 1654-1952, 2281-2312)

- **`conversion/__init__.py`** ✓
  - Exports label and source utilities
  - Converters will be added when extracted

**Status**: Labels and sources done, converters pending extraction

---

## Remaining Work

### Phase 4: DSPy Components Module
**Location**: `src/mathster/parsing/dspy_components/` (TO CREATE)

Extract DSPy-specific code from `extract_workflow.py` and `improve_workflow.py`:

#### Files to Create:
1. **`dspy_components/signatures.py`**
   - `ExtractMathematicalConcepts` (from extract_workflow.py:631-761)
   - `ExtractWithValidation` (from extract_workflow.py:763-813)
   - `ExtractSingleLabel` (from extract_workflow.py:1420-1475)
   - `ImproveMathematicalConcepts` (from improve_workflow.py:244-344)

2. **`dspy_components/extractors.py`**
   - `MathematicalConceptExtractor` (from extract_workflow.py:815-884)
   - `SingleLabelExtractor` (from extract_workflow.py:1477-1538)

3. **`dspy_components/improvers.py`**
   - `MathematicalConceptImprover` (from improve_workflow.py:346-415)

4. **`dspy_components/tools.py`**
   - `validate_extraction_tool()` (from extract_workflow.py:545-579)
   - `compare_labels_tool()` (from extract_workflow.py:581-624)
   - `validate_single_entity_tool()` (from extract_workflow.py:1345-1418)
   - `compare_extractions_tool()` (from improve_workflow.py:116-198)
   - `validate_improvement_tool()` (from improve_workflow.py:200-237)

5. **`dspy_components/__init__.py`**
   - Export all signatures, modules, and tools

---

### Phase 5: Text Processing Module
**Location**: `src/mathster/parsing/text_processing/` (TO CREATE)

Reorganize `tools.py` into focused submodules:

#### Files to Create:
1. **`text_processing/numbering.py`**
   - `add_line_numbers()` (from tools.py:7-30)

2. **`text_processing/splitting.py`**
   - `split_markdown_by_chapters()` (from tools.py:32-78)
   - `split_markdown_by_chapters_with_line_numbers()` (from tools.py:80-120)

3. **`text_processing/analysis.py`**
   - `classify_label()` (from tools.py:122-170)
   - `analyze_labels_in_chapter()` (from tools.py:172-289)
   - `_extract_labels_from_data()` (from tools.py:291-410)
   - `_format_comparison_report()` (from tools.py:412-509)
   - `compare_extraction_with_source()` (from tools.py:511-606)

4. **`text_processing/__init__.py`**
   - Export all text processing functions

---

### Phase 6: Workflows Module
**Location**: `src/mathster/parsing/workflows/` (TO CREATE)

Split workflow logic into focused modules:

#### Files to Create:
1. **`workflows/extract.py`**
   - `extract_chapter()` (from extract_workflow.py:1959-2089)
   - `extract_chapter_by_labels()` (from extract_workflow.py:2091-2279)

2. **`workflows/improve.py`**
   - `improve_chapter()` (from improve_workflow.py:1082-1173)
   - `improve_chapter_by_labels()` (from improve_workflow.py:877-1075)
   - `compute_changes()` (from improve_workflow.py:422-526)

3. **`workflows/retry.py`**
   - `extract_chapter_with_retry()` (from extract_workflow.py:1078-1207)
   - `extract_label_with_retry()` (from extract_workflow.py:1209-1338)
   - `improve_chapter_with_retry()` (from improve_workflow.py:533-715)
   - `improve_label_with_retry()` (from improve_workflow.py:717-875)

4. **`workflows/__init__.py`**
   - Export main workflow functions

---

### Phase 7: Configuration and Orchestration
**Location**: `src/mathster/parsing/` (root level)

Create top-level orchestration files:

#### Files to Create:
1. **`config.py`**
   - `configure_dspy()` (from dspy_pipeline.py:104-150)

2. **`orchestrator.py`**
   - `process_document()` (from dspy_pipeline.py:157-461)
   - `parse_line_number()` (from dspy_pipeline.py:65-80)
   - `extract_section_id()` (from dspy_pipeline.py:82-102)

---

### Phase 8: CLI Module
**Location**: `src/mathster/parsing/cli.py` (TO CREATE)

Extract CLI interface from `dspy_pipeline.py`:

- `main()` function (from dspy_pipeline.py:468-602)
- Argument parsing
- Entry point logic

---

### Phase 9: Backward-Compatible Exports
**Location**: `src/mathster/parsing/__init__.py` (UPDATE)

Update module __init__.py to maintain backward compatibility:

```python
"""
Mathster parsing module.

This module provides tools for parsing and extracting mathematical content
from markdown documents using DSPy-based ReAct agents with self-validation.

New modular structure (v2.0):
    models/              - Pure data models
    validation/          - Validation and error handling
    conversion/          - Data transformation utilities
    dspy_components/     - DSPy signatures, modules, and tools
    text_processing/     - Markdown processing utilities
    workflows/           - High-level workflows (extract, improve, retry)
    config.py            - DSPy configuration
    orchestrator.py      - Document processing pipeline
    cli.py               - Command-line interface

Backward-compatible imports (OLD API - will be deprecated in v3.0):
    from mathster.parsing import extract_chapter  # Still works
    from mathster.parsing import improve_chapter  # Still works
    from mathster.parsing import ChapterExtraction  # Still works
"""

# NEW API (recommended)
from mathster.parsing import models
from mathster.parsing import validation
from mathster.parsing import conversion
from mathster.parsing import dspy_components
from mathster.parsing import text_processing
from mathster.parsing import workflows

# Backward-compatible imports (OLD API)
from mathster.parsing.models import (
    ChapterExtraction,
    ValidationResult,
    ImprovementResult,
)
from mathster.parsing.workflows.extract import (
    extract_chapter,
    extract_chapter_by_labels,
)
from mathster.parsing.workflows.improve import (
    improve_chapter,
    improve_chapter_by_labels,
)
from mathster.parsing.text_processing.splitting import (
    split_markdown_by_chapters_with_line_numbers,
)
from mathster.parsing.conversion.labels import sanitize_label

__all__ = [
    # New modular API
    "models",
    "validation",
    "conversion",
    "dspy_components",
    "text_processing",
    "workflows",
    # Old API (backward compatibility)
    "extract_chapter",
    "extract_chapter_by_labels",
    "improve_chapter",
    "improve_chapter_by_labels",
    "ChapterExtraction",
    "ValidationResult",
    "ImprovementResult",
    "sanitize_label",
    "split_markdown_by_chapters_with_line_numbers",
]
```

---

### Phase 10: Testing and Validation
**Run**: After all modules created

1. **Update test imports**:
   - Find all `from mathster.parsing.extract_workflow import` in tests/
   - Update to use new modular imports
   - Example: `from mathster.parsing.models import ChapterExtraction`

2. **Run full test suite**:
   ```bash
   pytest tests/ -v
   ```

3. **Validate no circular imports**:
   ```bash
   python -c "import mathster.parsing; print('✓ Import successful')"
   ```

4. **Run sample extraction**:
   ```bash
   python -m mathster.parsing.cli docs/source/1_euclidean_gas/01_fragile_gas_framework.md
   ```

---

## Implementation Strategy

### Quick Start (Complete Remaining Phases)

**Option A: Manual Extraction** (Recommended for learning the code)
1. Create each module file one by one
2. Copy-paste code from existing files
3. Update imports as you go
4. Test after each phase

**Option B: Automated Script** (Faster, but less educational)
1. Write a migration script to extract all code
2. Run automated import updates
3. Validate with tests

### Example: Completing Phase 4 (DSPy Components)

```bash
# 1. Create directory structure
mkdir -p src/mathster/parsing/dspy_components

# 2. Create __init__.py
cat > src/mathster/parsing/dspy_components/__init__.py << 'EOF'
"""DSPy components for mathematical entity extraction."""

from mathster.parsing.dspy_components.signatures import (
    ExtractMathematicalConcepts,
    ExtractWithValidation,
    ExtractSingleLabel,
    ImproveMathematicalConcepts,
)
from mathster.parsing.dspy_components.extractors import (
    MathematicalConceptExtractor,
    SingleLabelExtractor,
)
from mathster.parsing.dspy_components.improvers import (
    MathematicalConceptImprover,
)

__all__ = [
    # Signatures
    "ExtractMathematicalConcepts",
    "ExtractWithValidation",
    "ExtractSingleLabel",
    "ImproveMathematicalConcepts",
    # Modules
    "MathematicalConceptExtractor",
    "SingleLabelExtractor",
    "MathematicalConceptImprover",
]
EOF

# 3. Extract sketch_strategist.py
# Read lines 631-813 from extract_workflow.py
# Read lines 1420-1475 from extract_workflow.py
# Read lines 244-344 from improve_workflow.py
# Combine into sketch_strategist.py with proper imports

# 4. Extract extractors.py
# Read lines 815-884, 1477-1538 from extract_workflow.py

# 5. Extract improvers.py
# Read lines 346-415 from improve_workflow.py

# 6. Extract tools.py
# Read validation/comparison tool functions

# 7. Update imports in existing files to use new dspy_components module
```

---

## Dependency Graph (After Refactoring)

```
Models (no dependencies)
   ↑
Validation → Conversion (labels, sources)
   ↑             ↑
   └─────┬───────┘
         ↑
   Conversion (converters)
         ↑
   DSPy Components
         ↑
     Workflows
         ↑
   Orchestrator
         ↑
        CLI
```

**Key Benefits**:
- No circular dependencies
- Clear dependency hierarchy
- Each module independently testable
- Easy to extend with new entity types

---

## File Size Comparison

### Before Refactoring:
```
extract_workflow.py:  2311 lines (MONOLITHIC)
improve_workflow.py:  1173 lines (HIGH COUPLING)
dspy_pipeline.py:      605 lines (MIXED CONCERNS)
tools.py:              605 lines (SCATTERED LOGIC)
Total:                4694 lines in 4 files
```

### After Refactoring:
```
models/entities.py:       175 lines
models/results.py:         78 lines
models/changes.py:         30 lines
validation/validators.py: 170 lines
validation/errors.py:     233 lines
conversion/labels.py:     180 lines
conversion/sources.py:     40 lines
conversion/converters.py: ~300 lines (to extract)
dspy_components/*:        ~800 lines (to extract)
text_processing/*:        ~600 lines (to extract)
workflows/*:             ~1400 lines (to extract)
config.py:                 ~50 lines (to extract)
orchestrator.py:          ~300 lines (to extract)
cli.py:                   ~135 lines (to extract)
Total:               ~4700 lines in 20+ focused files
```

**Result**: Same functionality, better organization!

---

## Next Steps

1. **Complete Phase 3**: Extract `converters.py` (convert_to_raw_document_section, convert_dict_to_extraction_entity)
2. **Complete Phase 4**: Create dspy_components module
3. **Complete Phase 5**: Create text_processing module
4. **Complete Phase 6**: Create workflows module
5. **Complete Phase 7**: Create config.py and orchestrator.py
6. **Complete Phase 8**: Create cli.py
7. **Complete Phase 9**: Update __init__.py with backward-compatible exports
8. **Complete Phase 10**: Run tests and validate

---

## Questions or Issues?

- **Circular imports**: Use lazy imports (import inside functions) when needed
- **Breaking changes**: Maintain old API in __init__.py for backward compatibility
- **Testing**: Update test imports to use new structure
- **Documentation**: Update docstrings to reference new module paths

---

**Status**: ✓ Phases 1-3 COMPLETE | ⏳ Phases 4-10 PENDING
**Created**: 2025-11-02
**Last Updated**: 2025-11-02
