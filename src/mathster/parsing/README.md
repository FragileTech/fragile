# Mathster Parsing Module

Mathematical entity extraction from markdown documents using DSPy-based ReAct agents.

## Quick Start

### New Modular API (Recommended)

```python
# Import modules
from mathster.parsing import models, validation, conversion

# Create extraction
extraction = models.ChapterExtraction(
    section_id="## 1. Introduction",
    definitions=[
        models.DefinitionExtraction(
            label="def-lipschitz",
            line_start=10,
            line_end=15,
            term="Lipschitz continuous",
        )
    ],
)

# Validate
result = validation.validate_extraction(
    extraction.model_dump(),
    file_path="docs/source/1_euclidean_gas/01_framework.md",
    article_id="01_fragile_gas_framework",
    chapter_text=chapter_text,
)

# Convert
raw_section, warnings = conversion.convert_to_raw_document_section(
    extraction,
    file_path="docs/source/1_euclidean_gas/01_framework.md",
    article_id="01_fragile_gas_framework",
    chapter_text=chapter_text,
)
```

### Legacy API (Backward Compatible)

```python
# Old imports still work
from mathster.parsing import extract_chapter, improve_chapter
from mathster.parsing import ChapterExtraction, ValidationResult

# Extract from markdown
raw_section, errors = extract_chapter(
    chapter_text=chapter_with_line_numbers,
    chapter_number=0,
    file_path="docs/source/1_euclidean_gas/01_framework.md",
    article_id="01_fragile_gas_framework",
)

# Improve existing extraction
raw_section, changes, errors = improve_chapter(
    chapter_text=chapter_with_line_numbers,
    existing_extraction=previous_extraction,
    file_path="docs/source/1_euclidean_gas/01_framework.md",
    article_id="01_fragile_gas_framework",
)
```

## Module Structure

### ✅ Foundation Modules (COMPLETE)

#### `models/` - Pure Data Models
- **entities.py**: `ChapterExtraction`, `DefinitionExtraction`, `TheoremExtraction`, etc.
- **results.py**: `ValidationResult`, `ImprovementResult`
- **changes.py**: `ChangeOperation`, `EntityChange`

```python
from mathster.parsing.models import ChapterExtraction, ValidationResult
from mathster.parsing.models.entities import DefinitionExtraction
from mathster.parsing.models.changes import ChangeOperation
```

#### `validation/` - Validation & Error Handling
- **validators.py**: `validate_extraction()`
- **errors.py**: `make_error_dict()`, `generate_detailed_error_report()`

```python
from mathster.parsing.validation import validate_extraction, make_error_dict
```

#### `conversion/` - Data Transformation
- **converters.py**: `convert_to_raw_document_section()`, `convert_dict_to_extraction_entity()`
- **labels.py**: `sanitize_label()`, `lookup_label_from_context()`
- **sources.py**: `create_source_location()`

```python
from mathster.parsing.conversion import sanitize_label, convert_to_raw_document_section
```

### ⏳ Legacy Files (Still Functional)

- **extract_workflow.py**: Extract entities from markdown
- **improve_workflow.py**: Enhance existing extractions
- **dspy_pipeline.py**: Orchestration and CLI
- **tools.py**: Text processing utilities

## Common Tasks

### Create an Extraction

```python
from mathster.parsing.models import ChapterExtraction, TheoremExtraction

extraction = ChapterExtraction(
    section_id="## 2. Main Results",
    theorems=[
        TheoremExtraction(
            label="thm-convergence",
            line_start=50,
            line_end=75,
            statement_type="theorem",
            conclusion_formula_latex="\\lim_{n\\to\\infty} x_n = x^*",
            definition_references=["def-lipschitz", "def-bounded"],
        )
    ],
)
```

### Validate an Extraction

```python
from mathster.parsing.validation import validate_extraction

result = validate_extraction(
    extraction.model_dump(),
    file_path="docs/source/1_euclidean_gas/01_framework.md",
    article_id="01_fragile_gas_framework",
    chapter_text=numbered_chapter_text,
)

if result.is_valid:
    print(f"✓ Valid! Entities: {result.entities_validated}")
else:
    print(f"✗ Errors: {result.errors}")
```

### Create Error Dicts

```python
from mathster.parsing.validation import make_error_dict

# Simple error
error = make_error_dict("Parsing failed")

# Error with debugging context
error = make_error_dict(
    "Failed to convert definition def-lipschitz",
    value={"label": "def-lipschitz", "term": "..."}
)
```

### Sanitize Labels

```python
from mathster.parsing.conversion import sanitize_label

# Normalize raw labels
label = sanitize_label("## 1. Introduction")  # → "section-1-introduction"
label = sanitize_label("param_Theta")  # → "param-theta"
label = sanitize_label("def_Energy")  # → "def-energy"
```

## CLI Usage

```bash
# Extract from markdown
python -m mathster.parsing.dspy_pipeline docs/source/1_euclidean_gas/01_framework.md

# With options
python -m mathster.parsing.dspy_pipeline \
    docs/source/1_euclidean_gas/01_framework.md \
    --model gemini/gemini-flash-lite-latest \
    --fallback-model anthropic/claude-haiku-4-5 \
    --max-retries 5 \
    --improvement-mode single_label
```

## Testing

```bash
# Run error dict tests
python tests/test_error_dict_format.py

# Test imports
python -c "from mathster.parsing import models, validation, conversion; print('✓')"

# Test backward compatibility
python -c "from mathster.parsing import extract_chapter; print('✓')"
```

## Documentation

- **REFACTORING_COMPLETE.md** - Full refactoring summary
- **MIGRATION_GUIDE.md** - Complete migration instructions
- **REFACTORING_STATUS.md** - Initial analysis and planning

## Architecture

```
Models (no dependencies)
   ↑
Validation → Conversion
   ↑             ↑
   └─────┬───────┘
         ↑
    Workflows (legacy)
         ↑
   Orchestrator (legacy)
         ↑
        CLI (legacy)
```

## Status

- ✅ **Phases 1-3**: Foundation modules (models, validation, conversion)
- ✅ **Phase 9**: Backward-compatible exports
- ✅ **Phase 10**: All tests passing
- ⏳ **Phases 4-8**: Remaining code organization (optional)

**Production Ready**: Both new modular API and legacy API fully functional.

## License

See project root LICENSE file.
