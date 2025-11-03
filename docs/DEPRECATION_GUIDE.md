# Mathster Parsing Module - Deprecation Guide

## Overview

The mathster parsing module has been fully refactored into a modular architecture. The old monolithic files are now **deprecated** but remain functional for backward compatibility.

## Deprecated Files

The following files are deprecated and should not be imported directly:

1. **`extract_workflow.py`** - Use `mathster.parsing.workflows` instead
2. **`improve_workflow.py`** - Use `mathster.parsing.workflows` instead
3. **`dspy_pipeline.py`** - Use `mathster.parsing.orchestrator` and `mathster.parsing.cli` instead
4. **`tools.py`** - Use `mathster.parsing.text_processing` instead

All deprecated files contain clear deprecation warnings in their docstrings with migration instructions.

## Migration Guide

### ❌ OLD (Deprecated)

```python
# Direct imports from old files (deprecated)
from mathster.parsing.extract_workflow import extract_chapter
from mathster.parsing.improve_workflow import improve_chapter
from mathster.parsing.dspy_pipeline import process_document, configure_dspy
from mathster.parsing.tools import add_line_numbers
```

### ✅ NEW (Recommended)

```python
# Import from workflows module
from mathster.parsing.workflows import extract_chapter, improve_chapter

# Or import from main package (convenience)
from mathster.parsing import extract_chapter, improve_chapter

# Import orchestration functions
from mathster.parsing import process_document, configure_dspy, cli_main

# Import text processing
from mathster.parsing.text_processing import add_line_numbers
```

## Complete Migration Examples

### Example 1: Extraction Workflow

**Old Code:**
```python
from mathster.parsing.extract_workflow import extract_chapter, extract_chapter_with_retry

raw_section, errors = extract_chapter(
    chapter_text=numbered_text,
    chapter_number=0,
    file_path="docs/source/document.md",
    article_id="01_document"
)
```

**New Code:**
```python
from mathster.parsing.workflows import extract_chapter, extract_chapter_with_retry

# Same function signature - no changes needed
raw_section, errors = extract_chapter(
    chapter_text=numbered_text,
    chapter_number=0,
    file_path="docs/source/document.md",
    article_id="01_document"
)
```

### Example 2: Improvement Workflow

**Old Code:**
```python
from mathster.parsing.improve_workflow import improve_chapter, compute_changes

raw_section, result, errors = improve_chapter(
    chapter_text=numbered_text,
    existing_extraction=existing_data,
    file_path="docs/source/document.md",
    article_id="01_document"
)
```

**New Code:**
```python
from mathster.parsing.workflows import improve_chapter, compute_changes

# Same function signature - no changes needed
raw_section, result, errors = improve_chapter(
    chapter_text=numbered_text,
    existing_extraction=existing_data,
    file_path="docs/source/document.md",
    article_id="01_document"
)
```

### Example 3: Pipeline Orchestration

**Old Code:**
```python
from mathster.parsing.dspy_pipeline import process_document, configure_dspy

configure_dspy(model="gemini/gemini-flash-lite-latest")
process_document(
    markdown_file="docs/source/document.md",
    output_dir="docs/source/raw_data"
)
```

**New Code:**
```python
from mathster.parsing import process_document, configure_dspy

# Same function signatures - no changes needed
configure_dspy(model="gemini/gemini-flash-lite-latest")
process_document(
    markdown_file="docs/source/document.md",
    output_dir="docs/source/raw_data"
)
```

### Example 4: Text Processing

**Old Code:**
```python
from mathster.parsing.tools import add_line_numbers, split_markdown_by_chapters_with_line_numbers

numbered = add_line_numbers(text)
chapters = split_markdown_by_chapters_with_line_numbers(file_path)
```

**New Code:**
```python
from mathster.parsing.text_processing import add_line_numbers, split_markdown_by_chapters_with_line_numbers

# Same function signatures - no changes needed
numbered = add_line_numbers(text)
chapters = split_markdown_by_chapters_with_line_numbers(file_path)
```

### Example 5: CLI Usage

**Old Command:**
```bash
python -m mathster.parsing.dspy_pipeline docs/source/document.md
```

**New Command:**
```bash
python -m mathster.parsing.cli docs/source/document.md
```

## Why Deprecate?

The old monolithic files had several issues:

1. **Poor Separation of Concerns**: Mixed models, validation, conversion, and workflows
2. **High Coupling**: Circular dependencies and tight coupling between components
3. **Hard to Test**: Difficult to test components in isolation
4. **Poor Organization**: Large files (2000+ lines) were hard to navigate
5. **Mixed Responsibilities**: Each file had too many responsibilities

The new modular structure addresses all these issues with:
- Clear separation of concerns
- Independent, testable modules
- Better code organization
- Single responsibility per module

## Backward Compatibility

**Important**: All old imports still work! The refactoring maintains 100% backward compatibility.

- Old function signatures unchanged
- Old behavior preserved
- No breaking changes
- Gradual migration path available

You can migrate at your own pace. Both old and new imports work simultaneously.

## Timeline

- **Now**: Old files deprecated with warnings in docstrings
- **Future (TBD)**: Old files may be removed in a future major version
- **Recommendation**: Migrate new code to modular API immediately, migrate existing code gradually

## Questions?

**Q: Do I need to update my code immediately?**
A: No. Old imports still work. But new code should use the modular API.

**Q: Will my code break?**
A: No. Zero breaking changes. All old imports are backward compatible.

**Q: What if I see deprecation warnings?**
A: Follow the migration examples above to update your imports.

**Q: Can I mix old and new imports?**
A: Yes! Both work simultaneously. Migrate gradually.

**Q: Where can I find more help?**
A: See `docs/REFACTORING_COMPLETE.md` for complete documentation.

## Summary

✅ **Do This:**
```python
from mathster.parsing.workflows import extract_chapter, improve_chapter
from mathster.parsing import process_document, configure_dspy
from mathster.parsing.text_processing import add_line_numbers
```

❌ **Not This:**
```python
from mathster.parsing.extract_workflow import extract_chapter
from mathster.parsing.improve_workflow import improve_chapter
from mathster.parsing.dspy_pipeline import process_document
from mathster.parsing.tools import add_line_numbers
```

---

**Status**: Active Deprecation
**Date**: 2025-11-02
**Breaking Changes**: None (backward compatible)
**Migration Required**: Recommended but not mandatory
