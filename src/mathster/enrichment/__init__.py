"""
Mathematical Entity Enrichment Module.

This module (Stage 1.5 of the pipeline) enriches parsed entities with full text content
by extracting text from source locations and validates semantic correctness.

Stages:
  Stage 1: Parsing (mathster.parsing)
    → chapter_N.json with line ranges

  Stage 1.5: Enrichment (mathster.enrichment) ← THIS MODULE
    → Extract full text from line ranges
    → Validate semantic correctness with DSPy

  Stage 2: Parameter Extraction (mathster.parameter_extraction)
    → Extract parameters

Usage:
    from mathster.enrichment import extract_full_text, enrich_chapter_file, validate_chapter

    # Enrich a chapter file
    enriched_path = enrich_chapter_file("chapter_0.json")

    # Validate semantics (optional)
    report = validate_chapter(enriched_path, entity_types=["parameters"])
"""

from mathster.enrichment import dspy_components, workflows
from mathster.enrichment.text_extractor import (
    enrich_chapter_file,
    extract_full_text,
    extract_full_text_from_dict,
)
from mathster.enrichment.workflows.validate import validate_enriched_chapter

# Convenience alias
validate_chapter = validate_enriched_chapter

__all__ = [
    "dspy_components",
    "enrich_chapter_file",
    "extract_full_text",
    "extract_full_text_from_dict",
    "validate_chapter",
    "validate_enriched_chapter",
    "workflows",
]
