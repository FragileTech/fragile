"""
Core Proof and Theorem Types.

This module contains the fundamental types for the mathematical framework:
- Mathematical entities (math_types): Objects, Properties, Relationships, Theorems, Axioms, Parameters
- Pipeline execution (pipeline_types): PipelineState, Result types, Graph structures
- Proof system (ProofBox, ProofStep)
- Review system (Review, ValidationResult, ReviewRegistry)
- Article system (Article, SourceLocation)
"""

# Import mathematical document types from proof_pipeline
from mathster.proof_pipeline.math_types import (
    Attribute,
    AttributeEvent,
    AttributeEventType,
    Axiom,
    DefinitionBox,
    MathematicalObject,
    Parameter,
    Relationship,
    RelationshipAttribute,
    RelationType,
    TheoremBox,
)
from mathster.proof_pipeline.proof_system import (
    ProofBox,
)

# Import article system types
from mathster.core.article_system import (
    Article,
    SourceLocation,
)

# Import bibliography types
from mathster.core.bibliography import (
    Bibliography,
    Citation,
)

# Import enriched types (for LLM pipeline)
from mathster.core.enriched_data import (
    EnrichedAxiom,
    EnrichedDefinition,
    EnrichedObject,
    EnrichedTheorem,
    EquationBox,
    ParameterBox,
    ParameterScope,
    RemarkBox,
    RemarkType,
)

# Import error tracking types
from mathster.error_tracking import (
    create_logger_for_document,
    ErrorLogEntry,
    ErrorLogger,
    ErrorSummary,
    merge_reports,
    ValidationReport as ExtractionValidationReport,  # Renamed to avoid conflict
)

# Import orchestration types (for LLM pipeline)
from mathster.orchestration import (
    AmbiguousReference,
    EnrichmentError,
    EnrichmentErrorInfo,
    EnrichmentStatus,
    EntityEnrichmentStatus,
    ErrorType,
    ResolutionContext,
    ValidationResult as OrchestrationValidationResult,  # Renamed to avoid conflict
)


# NOTE: model_rebuild() commented out to avoid DualStatement import errors
# These rebuilds are not strictly necessary and cause issues when DualStatement
# is in TYPE_CHECKING guard. If needed, can be manually called after imports.
# Article.model_rebuild()
# EquationBox.model_rebuild()
# ParameterBox.model_rebuild()
# RemarkBox.model_rebuild()
# EnrichedAxiom.model_rebuild()
# EnrichedTheorem.model_rebuild()
# EnrichedDefinition.model_rebuild()
# EnrichedObject.model_rebuild()
