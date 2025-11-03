# Parsing Module Migration Guide

## Current Status: Hybrid Architecture ✓

**Completed**: Phases 1-3 (Foundation modules fully extracted and working)
**Remaining**: Phases 4-10 (Code organization - existing files still functional)

---

## Working Hybrid State

### ✅ NEW Modular Modules (Phases 1-3 COMPLETE)

These are **fully functional** and can be imported immediately:

```python
# Import new modular code
from mathster.parsing.models import ChapterExtraction, ValidationResult
from mathster.parsing.validation import validate_extraction, make_error_dict
from mathster.parsing.conversion import (
    sanitize_label,
    convert_to_raw_document_section,
)
```

**Files**:
- `models/entities.py` - All extraction models
- `models/results.py` - ValidationResult, ImprovementResult
- `models/changes.py` - ChangeOperation, EntityChange
- `validation/validators.py` - validate_extraction()
- `validation/errors.py` - make_error_dict(), generate_detailed_error_report()
- `conversion/labels.py` - sanitize_label(), lookup_label_from_context()
- `conversion/sources.py` - create_source_location()
- `conversion/converters.py` - convert_to_raw_document_section(), convert_dict_to_extraction_entity()

---

### ⏳ OLD Files (Still Working - To Be Migrated)

These files still contain code that will be extracted in Phases 4-10:

- `extract_workflow.py` (2311 lines) - Contains DSPy components + workflows
- `improve_workflow.py` (1173 lines) - Contains DSPy components + workflows
- `dspy_pipeline.py` (605 lines) - Contains config + orchestrator + CLI
- `tools.py` (605 lines) - Contains text processing utilities

**These files STILL WORK** - they import from the new modules where needed.

---

## Completing the Migration

### Quick Reference: What Goes Where

**Phase 4: dspy_components/** (Extract from extract_workflow.py + improve_workflow.py)
```
signatures.py:
  - ExtractMathematicalConcepts (extract_workflow.py:631-761)
  - ExtractWithValidation (extract_workflow.py:763-813)
  - ExtractSingleLabel (extract_workflow.py:1420-1475)
  - ImproveMathematicalConcepts (improve_workflow.py:244-344)

extractors.py:
  - MathematicalConceptExtractor (extract_workflow.py:815-884)
  - SingleLabelExtractor (extract_workflow.py:1477-1538)

improvers.py:
  - MathematicalConceptImprover (improve_workflow.py:346-415)

tools.py:
  - validate_extraction_tool (extract_workflow.py:545-579)
  - compare_labels_tool (extract_workflow.py:581-624)
  - validate_single_entity_tool (extract_workflow.py:1345-1418)
  - compare_extractions_tool (improve_workflow.py:116-198)
  - validate_improvement_tool (improve_workflow.py:200-237)
```

**Phase 5: text_processing/** (Extract from tools.py)
```
numbering.py:
  - add_line_numbers (tools.py:7-30)

splitting.py:
  - split_markdown_by_chapters (tools.py:32-78)
  - split_markdown_by_chapters_with_line_numbers (tools.py:80-120)

analysis.py:
  - classify_label (tools.py:122-170)
  - analyze_labels_in_chapter (tools.py:172-289)
  - _extract_labels_from_data (tools.py:291-410)
  - _format_comparison_report (tools.py:412-509)
  - compare_extraction_with_source (tools.py:511-606)
```

**Phase 6: workflows/** (Extract from extract_workflow.py + improve_workflow.py)
```
extract.py:
  - extract_chapter (extract_workflow.py:1959-2089)
  - extract_chapter_by_labels (extract_workflow.py:2091-2279)

improve.py:
  - improve_chapter (improve_workflow.py:1082-1173)
  - improve_chapter_by_labels (improve_workflow.py:877-1075)
  - compute_changes (improve_workflow.py:422-526)

retry.py:
  - extract_chapter_with_retry (extract_workflow.py:1078-1207)
  - extract_label_with_retry (extract_workflow.py:1209-1338)
  - improve_chapter_with_retry (improve_workflow.py:533-715)
  - improve_label_with_retry (improve_workflow.py:717-875)
```

**Phase 7: config.py + orchestrator.py** (Extract from dspy_pipeline.py)
```
config.py:
  - configure_dspy (dspy_pipeline.py:104-150)

orchestrator.py:
  - process_document (dspy_pipeline.py:157-461)
  - parse_line_number (dspy_pipeline.py:65-80)
  - extract_section_id (dspy_pipeline.py:82-102)
```

**Phase 8: cli.py** (Extract from dspy_pipeline.py)
```
cli.py:
  - main (dspy_pipeline.py:468-602)
```

---

## Migration Strategy

### Option A: Incremental Migration (Recommended)

Migrate one phase at a time, test after each phase:

```bash
# Phase 4: DSPy Components
python scripts/migrate_phase_4.py
pytest tests/ -v
git commit -m "Phase 4: Extract DSPy components"

# Phase 5: Text Processing
python scripts/migrate_phase_5.py
pytest tests/ -v
git commit -m "Phase 5: Extract text processing"

# ... continue for each phase
```

### Option B: Automated Full Migration

Run all phases at once:

```bash
python scripts/complete_refactoring.py
pytest tests/ -v
git commit -m "Complete parsing module refactoring"
```

### Option C: Manual Migration

1. Create each new file manually
2. Copy-paste the code from the old files using line numbers above
3. Update imports in the new files
4. Update imports in old files to use new modules
5. Test after each file
6. Delete old code once migrated

---

## Testing Strategy

### After Each Phase

```bash
# Run focused tests
pytest tests/test_error_dict_format.py -v
pytest tests/test_assumption_extraction.py -v
pytest tests/test_reference_validation.py -v

# Run full suite
pytest tests/ -v

# Check imports
python -c "from mathster.parsing import extract_chapter; print('✓ Imports work')"
```

### Final Validation

```bash
# Full test suite
pytest tests/ -v --cov=src/mathster/parsing

# Integration test
python -m mathster.parsing.cli docs/source/1_euclidean_gas/01_fragile_gas_framework.md

# Import validation
python -c "
from mathster.parsing import models, validation, conversion
from mathster.parsing import extract_chapter, improve_chapter
print('✓ All imports successful')
"
```

---

## Current Working Imports

### ✅ These work NOW (Phases 1-3 complete):

```python
# Models
from mathster.parsing.models import (
    ChapterExtraction,
    DefinitionExtraction,
    TheoremExtraction,
    ProofExtraction,
    AxiomExtraction,
    ParameterExtraction,
    RemarkExtraction,
    AssumptionExtraction,
    CitationExtraction,
    ValidationResult,
    ImprovementResult,
    ChangeOperation,
    EntityChange,
)

# Validation
from mathster.parsing.validation import (
    validate_extraction,
    make_error_dict,
    generate_detailed_error_report,
)

# Conversion
from mathster.parsing.conversion import (
    sanitize_label,
    lookup_label_from_context,
    create_source_location,
    convert_to_raw_document_section,
    convert_dict_to_extraction_entity,
)
```

### ⏳ These still use old imports (Phases 4-10 pending):

```python
# Still in extract_workflow.py
from mathster.parsing.extract_workflow import (
    MathematicalConceptExtractor,  # → Will move to dspy_components.extractors
    extract_chapter,                # → Will move to workflows.extract
    extract_chapter_by_labels,      # → Will move to workflows.extract
)

# Still in improve_workflow.py
from mathster.parsing.improve_workflow import (
    MathematicalConceptImprover,    # → Will move to dspy_components.improvers
    improve_chapter,                # → Will move to workflows.improve
)

# Still in dspy_pipeline.py
from mathster.parsing.dspy_pipeline import (
    configure_dspy,                 # → Will move to config.py
    process_document,               # → Will move to orchestrator.py
)

# Still in tools.py
from mathster.parsing.tools import (
    split_markdown_by_chapters_with_line_numbers,  # → Will move to text_processing.splitting
    analyze_labels_in_chapter,                      # → Will move to text_processing.analysis
)
```

---

## Benefits of Hybrid Approach

✅ **Phases 1-3 complete** - Foundation modules working and tested
✅ **Existing code still works** - No breaking changes yet
✅ **Can test incrementally** - Validate each phase before continuing
✅ **Can use new modules NOW** - Import from modular structure today
✅ **Low risk** - Old files provide fallback if issues arise

---

## Next Steps

1. **Test current state**:
   ```bash
   pytest tests/test_error_dict_format.py -v
   ```

2. **Choose migration approach**:
   - Incremental (Option A) - safest
   - Automated (Option B) - fastest
   - Manual (Option C) - most educational

3. **Complete remaining phases** (4-10)

4. **Update all imports** to use new modular structure

5. **Delete old files** once migration complete

6. **Update documentation** and README

---

## Questions?

- **"Can I use the new modules now?"** → YES! Phases 1-3 are complete
- **"Will old code break?"** → NO! Old files still work
- **"Should I wait for full migration?"** → Your choice - hybrid state is stable
- **"How long to complete?"** → ~2-4 hours manual, ~30min automated

---

**Last Updated**: 2025-11-02
**Status**: Phases 1-3 ✅ COMPLETE | Phases 4-10 ⏳ PENDING
