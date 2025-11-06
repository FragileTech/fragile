"""
Mathematical Entity Semantic Enrichment Module (Stage 3).

This module transforms raw extracted entities into fully enriched entities with
semantic metadata, domain inference, constraint extraction, and cross-references.

Pipeline Position:
  Stage 1: Parsing → Raw entities (RawParameter, RawTheorem, etc.)
  Stage 1.5: Text Enrichment → Populate full_text fields
  Stage 3: Semantic Enrichment ← THIS MODULE
    → Enriched entities (ParameterBox, EnrichedTheorem, etc.)

Enrichment Process:
  1. Automated enrichment (40%):
     - Enum conversions (scope, remark_type)
     - Field mapping (term → name, etc.)
     - LaTeX generation from symbols
     - Chapter/document extraction from source

  2. Cross-reference analysis (20%):
     - Build parameter usage indices
     - Link entities to dependencies
     - Track relationships

  3. LLM enrichment (40%):
     - Domain inference (ParameterType)
     - Constraint extraction
     - Theorem decomposition
     - Property inference

Usage:
    from mathster.semantic_enrichment import enrich_parameters

    # Enrich parameters in a chapter
    enriched_params = enrich_parameters(
        raw_parameters=chapter_data["parameters"],
        usage_index=build_parameter_usage_index(chapter_data)
    )

Current Status:
  ✅ Phase 1: Foundation utilities complete
  ✅ Phase 2: Parameter enrichment complete
  ⏸️ Phase 3-4: Theorem/Definition enrichment (future work)
"""

from mathster.semantic_enrichment import enrichers, utilities
from mathster.semantic_enrichment.enrichers.parameter_enricher import enrich_parameter
from mathster.semantic_enrichment.utilities import (
    build_parameter_usage_index,
    extract_chapter_from_source,
    extract_constraints_from_text,
    map_parameter_scope,
)

__all__ = [
    "build_parameter_usage_index",
    "enrichers",
    "enrich_parameter",
    "extract_chapter_from_source",
    "extract_constraints_from_text",
    "map_parameter_scope",
    "utilities",
]
