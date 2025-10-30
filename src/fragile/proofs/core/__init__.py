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
from fragile.proofs.core.math_types import (
    Attribute,
    AttributeEvent,
    AttributeEventType,
    AttributeRefinement,
    Axiom,
    AxiomaticParameter,
    create_simple_object,
    create_simple_theorem,
    DefinitionBox,
    MathematicalObject,
    ObjectType,
    Parameter,
    ParameterType,
    RefinementType,
    Relationship,
    RelationshipAttribute,
    RelationType,
    TheoremBox,
    TheoremOutputType,
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
from fragile.proofs.core.proof_integration import (
    attach_proof_to_theorem,
    create_proof_inputs_from_theorem,
    create_proof_outputs_from_theorem,
    create_proof_sketch_from_theorem,
    extract_relationships_from_proof,
    get_proof_statistics,
    print_validation_result,
    ProofTheoremMismatch,
    ProofValidationResult,
    validate_proof_for_theorem,
)

# Import proof system types
from fragile.proofs.core.proof_system import (
    AssumptionReference,
    AttributeReference,
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
    SubProofReference,
)
from fragile.proofs.core.review_helpers import (
    DualReviewProtocol,
    ReviewAnalyzer,
    ReviewBuilder,
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

# Import type conversion utilities
from fragile.proofs.core.type_conversions import (
    ConversionReport,
    convert_parameter_boxes,
    extract_chapter_document,
    link_all_enriched_types,
    link_equation_to_theorems,
    link_remark_to_entities,
    merge_constraints,
    parameter_box_to_parameter,
)

# Import error tracking types
from fragile.proofs.error_tracking import (
    create_logger_for_document,
    ErrorLogEntry,
    ErrorLogger,
    ErrorSummary,
    merge_reports,
    ValidationReport as ExtractionValidationReport,  # Renamed to avoid conflict
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


__all__ = [
    "AmbiguousReference",
    # Article system
    "Article",
    "AssumptionReference",
    "Attribute",
    "AttributeEvent",
    "AttributeEventType",
    "AttributeReference",
    "AttributeRefinement",
    "Axiom",
    "AxiomaticParameter",
    # Bibliography system
    "Bibliography",
    "Citation",
    "ConversionReport",
    "DataFlowEdge",
    "DataFlowNode",
    "DefinitionBox",
    "DependencyEdge",
    "DependencyNode",
    "DirectDerivation",
    "DualReviewProtocol",
    "EnrichedAxiom",
    "EnrichedDefinition",
    "EnrichedObject",
    "EnrichedTheorem",
    "EnrichmentError",
    "EnrichmentErrorInfo",
    "EnrichmentStatus",
    "EntityEnrichmentStatus",
    # Enriched types (LLM pipeline)
    "EquationBox",
    "Err",
    "ErrorLogEntry",
    # Error tracking (LLM pipeline)
    "ErrorLogger",
    "ErrorSummary",
    "ErrorType",
    "ExtractionValidationReport",
    "IssueResolution",
    "LLMResponse",
    "LemmaApplication",
    "MathematicalObject",
    "ObjectType",
    "Ok",
    "OrchestrationValidationResult",
    "Parameter",
    "ParameterBox",
    "ParameterScope",
    "ParameterType",
    "PipelineState",
    # Proof system
    "ProofBox",
    "ProofEngine",
    "ProofExpansionRequest",
    "ProofInput",
    "ProofOutput",
    "ProofStep",
    "ProofStepStatus",
    "ProofStepType",
    "ProofTheoremMismatch",
    "ProofValidationResult",
    "RefinementType",
    "RelationType",
    "Relationship",
    "RelationshipAttribute",
    "RemarkBox",
    "RemarkType",
    # Orchestration (LLM pipeline)
    "ResolutionContext",
    # Review system
    "Review",
    "ReviewAnalyzer",
    "ReviewBuilder",
    "ReviewComparison",
    "ReviewIssue",
    "ReviewSeverity",
    "ReviewSource",
    "ReviewStatus",
    "SourceLocation",
    "SubProofReference",
    "TheoremBox",
    # From pipeline_types
    "TheoremOutputType",
    "ValidationResult",
    "attach_proof_to_theorem",
    "convert_parameter_boxes",
    "create_logger_for_document",
    "create_proof_inputs_from_theorem",
    "create_proof_outputs_from_theorem",
    "create_proof_sketch_from_theorem",
    "create_simple_object",
    "create_simple_theorem",
    "extract_chapter_document",
    "extract_relationships_from_proof",
    "get_proof_statistics",
    "link_all_enriched_types",
    "link_equation_to_theorems",
    "link_remark_to_entities",
    "merge_constraints",
    "merge_reports",
    # Type conversions (enriched → framework)
    "parameter_box_to_parameter",
    "print_validation_result",
    # Proof integration
    "validate_proof_for_theorem",
]

# Rebuild models that reference SourceLocation and DualStatement after they're imported
# This is necessary because we used TYPE_CHECKING forward references

# Import DualStatement and PaperContext for model rebuilding
from fragile.proofs.sympy_integration.dual_representation import DualStatement, PaperContext


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
EnrichedAxiom.model_rebuild()
EnrichedTheorem.model_rebuild()
EnrichedDefinition.model_rebuild()
EnrichedObject.model_rebuild()
