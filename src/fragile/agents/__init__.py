"""
Autonomous agents for mathematical document processing.

Extract-then-Enrich Pipeline:
- Stage 1 (Raw Extraction): RawDocumentParser extracts verbatim content into raw JSON
- Stage 1.5 (Source Location Enrichment): TextLocationEnricher adds precise line ranges
- Stage 2 (Enrichment): To be implemented - enriches raw data into final structured models

Current Agents:
- RawDocumentParser: Stage 1 - Extracts raw entities from MyST markdown into StagingDocument
- TextLocationEnricher: Stage 1.5 - Enriches raw entities with TextLocation metadata
- extract_document: Convenience function for Stage 1 extraction
- extract_multiple_documents: Batch extraction for multiple documents

Stage 2 enrichment utilities (for future document-refiner agent):
- CrossReferenceAnalyzer: Discovers relationships and dependencies
- DependencyDetector: LLM-based implicit dependency detection
- RelationshipBuilder: Constructs and validates Relationship objects
"""

from fragile.agents.cross_reference_analyzer import CrossReferenceAnalyzer
from fragile.agents.dependency_detector import (
    AxiomDependency,
    DependencyDetector,
    DependencyReport,
    ImplicitRelationship,
    ObjectDependency,
    ParameterDependency,
    PropertyRequirement,
)
from fragile.agents.raw_document_parser import (
    extract_document,
    extract_multiple_documents,
    RawDocumentParser,
)
from fragile.agents.relationship_builder import RelationshipBuilder, RelationshipValidator
from fragile.agents.text_location_enricher import (
    EnrichmentConfig,
    EnrichmentResult,
    TextLocationEnricher,
)


__all__ = [
    "AxiomDependency",
    # Stage 2: Enrichment utilities (for document-refiner)
    "CrossReferenceAnalyzer",
    "DependencyDetector",
    "DependencyReport",
    # Stage 1.5: Source location enrichment
    "EnrichmentConfig",
    "EnrichmentResult",
    "ImplicitRelationship",
    "ObjectDependency",
    "ParameterDependency",
    "PropertyRequirement",
    # Stage 1: Raw extraction
    "RawDocumentParser",
    "RelationshipBuilder",
    "RelationshipValidator",
    # Stage 1.5: Source location enrichment
    "TextLocationEnricher",
    "extract_document",
    "extract_multiple_documents",
]
