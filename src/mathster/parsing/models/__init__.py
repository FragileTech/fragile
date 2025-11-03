"""
Data models for mathematical entity extraction and improvement.

This module contains pure Pydantic data models with no business logic.

Submodules:
    - entities: Mathematical entity extraction models (definitions, theorems, etc.)
    - results: Validation and improvement result models
    - changes: Change tracking models for improvements
"""

from mathster.parsing.models.changes import ChangeOperation, EntityChange
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
from mathster.parsing.models.results import ImprovementResult, ValidationResult

__all__ = [
    # Base
    "ExtractedEntity",
    # Entity types
    "DefinitionExtraction",
    "TheoremExtraction",
    "ProofExtraction",
    "AxiomExtraction",
    "ParameterExtraction",
    "RemarkExtraction",
    "AssumptionExtraction",
    "CitationExtraction",
    # Aggregate
    "ChapterExtraction",
    # Results
    "ValidationResult",
    "ImprovementResult",
    # Changes
    "ChangeOperation",
    "EntityChange",
]
