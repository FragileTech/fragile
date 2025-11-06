"""
Mathster parsing module.

This module provides tools for parsing and extracting mathematical content
from markdown documents using DSPy-based ReAct agents with self-validation.

## New Modular Architecture (v2.0)

The parsing module has been refactored into focused submodules:

### Foundation Modules (✅ COMPLETE - Phases 1-3):
- **models/** - Pure Pydantic data models
  - entities: ChapterExtraction, DefinitionExtraction, TheoremExtraction, etc.
  - results: ValidationResult, ImprovementResult
  - changes: ChangeOperation, EntityChange

- **validation/** - Validation logic and error handling
  - validators: validate_extraction()
  - errors: make_error_dict(), generate_detailed_error_report()

- **conversion/** - Data transformation utilities
  - converters: convert_to_raw_document_section()
  - labels: sanitize_label(), lookup_label_from_context()
  - sources: create_source_location()

### Extended Modules (✅ COMPLETE - Phases 4-8):
- **dspy_components/** - DSPy Signatures, Modules, and tools
  - signatures: ExtractMathematicalConcepts, ImproveMathematicalConcepts
  - extractors: MathematicalConceptExtractor, SingleLabelExtractor
  - improvers: MathematicalConceptImprover
  - tools: validate_extraction_tool, compare_extractions_tool

- **text_processing/** - Markdown processing utilities
  - numbering: add_line_numbers()
  - splitting: split_markdown_by_chapters(), split_markdown_by_chapters_with_line_numbers()
  - analysis: classify_label(), analyze_labels_in_chapter(), compare_extraction_with_source()

- **workflows/** - High-level workflows (extract, improve, retry)
  - Re-exports from extract_workflow and improve_workflow (backward compatible)

- **config.py** - DSPy configuration
  - configure_dspy()

- **orchestrator.py** - Document processing pipeline
  - process_document(), parse_line_number(), extract_section_id()

- **cli.py** - Command-line interface
  - main()

## Usage

### New Modular API (Recommended):
```python
# Import from specific modules
from mathster.parsing.models import ChapterExtraction, ValidationResult
from mathster.parsing.validation import validate_extraction
from mathster.parsing.conversion import sanitize_label

# Or import entire modules
from mathster.parsing import models, validation, conversion
```

### Legacy API (Backward Compatible):
```python
# Old imports still work (for now)
from mathster.parsing import extract_chapter, improve_chapter
from mathster.parsing import ChapterExtraction
```

For detailed migration guide, see: src/mathster/parsing/MIGRATION_GUIDE.md
"""

# =============================================================================
# MODULAR API (All modules complete - Phases 1-10)
# =============================================================================

# Foundation modules (Phases 1-3)
# Extended modules (Phases 4-8)
from mathster.parsing import (
    conversion,
    dspy_components,
    models,
    text_processing,
    validation,
    workflows,
)
from mathster.parsing.cli import main as cli_main
from mathster.parsing.config import configure_dspy
from mathster.parsing.conversion.converters import (
    convert_dict_to_extraction_entity,
    convert_to_raw_document_section,
)
from mathster.parsing.conversion.labels import (
    lookup_label_from_context,
    sanitize_label,
)
from mathster.parsing.conversion.sources import (
    create_source_location,
)
from mathster.parsing.models.changes import (
    ChangeOperation,
    EntityChange,
)

# Export from new modular structure
from mathster.parsing.models.entities import (
    AssumptionExtraction,
    AxiomExtraction,
    ChapterExtraction,
    CitationExtraction,
    DefinitionExtraction,
    ExtractedEntity,
    ParameterExtraction,
    ProofExtraction,
    RemarkExtraction,
    TheoremExtraction,
)
from mathster.parsing.models.results import (
    ImprovementResult,
    ValidationResult,
)
from mathster.parsing.orchestrator import extract_section_id, parse_line_number, process_document
from mathster.parsing.validation.errors import (
    generate_detailed_error_report,
    make_error_dict,
)
from mathster.parsing.validation.validators import (
    validate_extraction,
)

# Import workflow functions from workflows module (which internally uses old files)
from mathster.parsing.workflows import (
    compute_changes,
    extract_chapter,
    extract_chapter_by_labels,
    extract_chapter_with_retry,
    extract_label_with_retry,
    improve_chapter,
    improve_chapter_by_labels,
    improve_chapter_with_retry,
    improve_label_with_retry,
)

# Import parameter extraction workflows
from mathster.parsing.workflows.extract_parameters import (
    extract_parameters_from_chapter,
    improve_parameters_from_chapter,
)


__all__ = [
    "AssumptionExtraction",
    "AxiomExtraction",
    "ChangeOperation",
    # Data models (from models/)
    "ChapterExtraction",
    "CitationExtraction",
    "DefinitionExtraction",
    "EntityChange",
    "ExtractedEntity",
    "ImprovementResult",
    "ParameterExtraction",
    "ProofExtraction",
    "RemarkExtraction",
    "TheoremExtraction",
    "ValidationResult",
    "cli_main",
    "compute_changes",
    # Configuration and orchestration (Phases 7-8)
    "configure_dspy",
    "conversion",
    "convert_dict_to_extraction_entity",
    "convert_to_raw_document_section",
    "create_source_location",
    "dspy_components",
    # Workflow functions (from workflows/)
    "extract_chapter",
    "extract_chapter_by_labels",
    "extract_chapter_with_retry",
    "extract_label_with_retry",
    "extract_parameters_from_chapter",
    "extract_section_id",
    "generate_detailed_error_report",
    "improve_chapter",
    "improve_chapter_by_labels",
    "improve_chapter_with_retry",
    "improve_label_with_retry",
    "improve_parameters_from_chapter",
    "lookup_label_from_context",
    "make_error_dict",
    # Modular submodules (Phases 1-8)
    "models",
    "parse_line_number",
    "process_document",
    # Conversion (from conversion/)
    "sanitize_label",
    "text_processing",
    # Validation (from validation/)
    "validate_extraction",
    "validation",
    "workflows",
]
