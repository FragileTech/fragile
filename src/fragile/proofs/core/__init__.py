"""
Core Proof and Theorem Types.

This module contains the fundamental types for the mathematical framework:
- Mathematical entities (math_types): Objects, Properties, Relationships, Theorems, Axioms, Parameters
- Pipeline execution (pipeline_types): PipelineState, Result types, Graph structures
- Proof system (ProofBox, ProofStep)
- Review system (Review, ValidationResult, ReviewRegistry)
- Article system (Article, SourceLocation)
"""

# Import mathematical document types from math_types
from fragile.proofs.core.math_types import (
    Axiom,
    AxiomaticParameter,
    DefinitionBox,
    MathematicalObject,
    ObjectType,
    Parameter,
    ParameterType,
    Attribute,
    AttributeEvent,
    AttributeEventType,
    AttributeRefinement,
    RefinementType,
    Relationship,
    RelationshipAttribute,
    RelationType,
    TheoremBox,
    TheoremOutputType,
    create_simple_object,
    create_simple_theorem,
)

# Import pipeline execution types from pipeline_types
from fragile.proofs.core.pipeline_types import (
    DataFlowEdge,
    DataFlowNode,
    DependencyEdge,
    DependencyNode,
    Err,
    Ok,
    PipelineState,
)

# Import proof system types
from fragile.proofs.core.proof_system import (
    AssumptionReference,
    DirectDerivation,
    LemmaApplication,
    ProofBox,
    ProofEngine,
    ProofExpansionRequest,
    ProofInput,
    ProofOutput,
    ProofStep,
    ProofStepStatus,
    ProofStepType,
    AttributeReference,
    SubProofReference,
)
from fragile.proofs.core.proof_integration import (
    ProofTheoremMismatch,
    ProofValidationResult,
    attach_proof_to_theorem,
    create_proof_inputs_from_theorem,
    create_proof_outputs_from_theorem,
    create_proof_sketch_from_theorem,
    extract_relationships_from_proof,
    get_proof_statistics,
    print_validation_result,
    validate_proof_for_theorem,
)

# Import review system types
from fragile.proofs.core.review_system import (
    IssueResolution,
    LLMResponse,
    Review,
    ReviewComparison,
    ReviewIssue,
    ReviewSeverity,
    ReviewSource,
    ReviewStatus,
    ValidationResult,
)
from fragile.proofs.core.review_helpers import (
    ReviewBuilder,
    ReviewAnalyzer,
    DualReviewProtocol,
)

# Import article system types
from fragile.proofs.core.article_system import (
    Article,
    SourceLocation,
)

# Import bibliography types
from fragile.proofs.core.bibliography import (
    Bibliography,
    Citation,
)

# Import enriched types (for LLM pipeline)
from fragile.proofs.core.enriched_types import (
    EquationBox,
    ParameterBox,
    RemarkBox,
    RemarkType,
    ParameterScope,
)

# Import orchestration types (for LLM pipeline)
from fragile.proofs.orchestration import (
    AmbiguousReference,
    EnrichmentError,
    EnrichmentErrorInfo,
    EnrichmentStatus,
    EntityEnrichmentStatus,
    ErrorType,
    ResolutionContext,
    ValidationResult as OrchestrationValidationResult,  # Renamed to avoid conflict
)

# Import error tracking types
from fragile.proofs.error_tracking import (
    ErrorLogger,
    ErrorLogEntry,
    ErrorSummary,
    ValidationReport as ExtractionValidationReport,  # Renamed to avoid conflict
    create_logger_for_document,
    merge_reports,
)

__all__ = [
    # From pipeline_types
    "TheoremOutputType",
    "ObjectType",
    "ParameterType",
    "RefinementType",
    "AttributeEventType",
    "RelationType",
    "Ok",
    "Err",
    "AttributeRefinement",
    "Attribute",
    "AttributeEvent",
    "RelationshipAttribute",
    "Relationship",
    "MathematicalObject",
    "DefinitionBox",
    "Axiom",
    "AxiomaticParameter",
    "Parameter",
    "TheoremBox",
    "DataFlowNode",
    "DataFlowEdge",
    "DependencyNode",
    "DependencyEdge",
    "PipelineState",
    "create_simple_object",
    "create_simple_theorem",
    # Proof system
    "ProofBox",
    "ProofStep",
    "ProofInput",
    "ProofOutput",
    "AttributeReference",
    "AssumptionReference",
    "DirectDerivation",
    "SubProofReference",
    "LemmaApplication",
    "ProofStepType",
    "ProofStepStatus",
    "ProofEngine",
    "ProofExpansionRequest",
    # Proof integration
    "validate_proof_for_theorem",
    "create_proof_inputs_from_theorem",
    "create_proof_outputs_from_theorem",
    "create_proof_sketch_from_theorem",
    "attach_proof_to_theorem",
    "extract_relationships_from_proof",
    "get_proof_statistics",
    "print_validation_result",
    "ProofValidationResult",
    "ProofTheoremMismatch",
    # Review system
    "Review",
    "ReviewIssue",
    "ReviewComparison",
    "ReviewSeverity",
    "ReviewSource",
    "ReviewStatus",
    "ValidationResult",
    "LLMResponse",
    "IssueResolution",
    "ReviewBuilder",
    "ReviewAnalyzer",
    "DualReviewProtocol",
    # Article system
    "Article",
    "SourceLocation",
    # Bibliography system
    "Bibliography",
    "Citation",
    # Enriched types (LLM pipeline)
    "EquationBox",
    "ParameterBox",
    "RemarkBox",
    "RemarkType",
    "ParameterScope",
    # Orchestration (LLM pipeline)
    "ResolutionContext",
    "EnrichmentError",
    "EnrichmentErrorInfo",
    "ErrorType",
    "AmbiguousReference",
    "EnrichmentStatus",
    "EntityEnrichmentStatus",
    "OrchestrationValidationResult",
    # Error tracking (LLM pipeline)
    "ErrorLogger",
    "ErrorLogEntry",
    "ErrorSummary",
    "ExtractionValidationReport",
    "create_logger_for_document",
    "merge_reports",
]

# Rebuild models that reference SourceLocation and DualStatement after they're imported
# This is necessary because we used TYPE_CHECKING forward references

# Import DualStatement and PaperContext for model rebuilding
from fragile.proofs.sympy.dual_representation import DualStatement, PaperContext

Attribute.model_rebuild()
MathematicalObject.model_rebuild()
DefinitionBox.model_rebuild()
Relationship.model_rebuild()
TheoremBox.model_rebuild()
ProofBox.model_rebuild()
Article.model_rebuild()
EquationBox.model_rebuild()
ParameterBox.model_rebuild()
RemarkBox.model_rebuild()
