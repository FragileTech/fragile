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

from mathster.parsing.dspy_components.signatures import (
    ExtractMathematicalConcepts,
    ExtractWithValidation,
    ExtractSingleLabel,
    ImproveMathematicalConcepts,
)
from mathster.parsing.dspy_components.extractors import (
    MathematicalConceptExtractor,
    MathematicalConceptExtractorWithValidation,
    SingleLabelExtractor,
)
from mathster.parsing.dspy_components.improvers import (
    MathematicalConceptImprover,
)
from mathster.parsing.dspy_components.tools import (
    validate_extraction_tool,
    compare_labels_tool,
    validate_single_entity_tool,
    compare_extractions_tool,
    validate_improvement_tool,
)

__all__ = [
    # Signatures
    "ExtractMathematicalConcepts",
    "ExtractWithValidation",
    "ExtractSingleLabel",
    "ImproveMathematicalConcepts",
    # Modules
    "MathematicalConceptExtractor",
    "MathematicalConceptExtractorWithValidation",
    "SingleLabelExtractor",
    "MathematicalConceptImprover",
    # Tools
    "validate_extraction_tool",
    "compare_labels_tool",
    "validate_single_entity_tool",
    "compare_extractions_tool",
    "validate_improvement_tool",
]
