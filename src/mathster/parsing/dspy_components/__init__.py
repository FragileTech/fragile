"""
DSPy components for mathematical entity extraction and improvement.

This module contains DSPy-specific code: Signatures, Modules, and tool wrappers
for the ReAct agent-based extraction and improvement workflows.

Submodules:
    - signatures: DSPy Signature definitions for extraction and improvement
    - extractors: DSPy Module implementations for entity extraction
    - improvers: DSPy Module implementations for entity improvement
    - tools: DSPy tool wrappers for validation and comparison
"""

from mathster.parsing.dspy_components.extractors import (
    MathematicalConceptExtractor,
    MathematicalConceptExtractorWithValidation,
    SingleLabelExtractor,
)
from mathster.parsing.dspy_components.improvers import (
    MathematicalConceptImprover,
)
from mathster.parsing.dspy_components.signatures import (
    ExtractMathematicalConcepts,
    ExtractSingleLabel,
    ExtractWithValidation,
    ImproveMathematicalConcepts,
)
from mathster.parsing.dspy_components.tools import (
    compare_extractions_tool,
    compare_labels_tool,
    validate_extraction_tool,
    validate_improvement_tool,
    validate_single_entity_tool,
)


__all__ = [
    # Signatures
    "ExtractMathematicalConcepts",
    "ExtractSingleLabel",
    "ExtractWithValidation",
    "ImproveMathematicalConcepts",
    # Modules
    "MathematicalConceptExtractor",
    "MathematicalConceptExtractorWithValidation",
    "MathematicalConceptImprover",
    "SingleLabelExtractor",
    "compare_extractions_tool",
    "compare_labels_tool",
    # Tools
    "validate_extraction_tool",
    "validate_improvement_tool",
    "validate_single_entity_tool",
]
