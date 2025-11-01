"""
Prompt Templates for LLM-Based Mathematical Paper Extraction.

This module provides prompt templates for the two-stage Extract-then-Enrich pipeline:
- Stage 1 (extraction.py): Raw entity extraction from markdown sections
- Stage 2 (enrichment.py): Focused semantic parsing and relationship extraction

Usage:
    # Stage 1: Extract raw entities
    from fragile.mathster.prompts import MAIN_EXTRACTION_PROMPT, get_extraction_prompt

    prompt = get_extraction_prompt(
        section_text=markdown_content,
        section_id="§2.1"
    )

    # Stage 2: Enrich and link entities
    from fragile.mathster.prompts import (
        DECOMPOSE_THEOREM_PROMPT,
        PARSE_LATEX_TO_DUAL_PROMPT,
        get_decompose_theorem_prompt
    )

    prompt = get_decompose_theorem_prompt(
        theorem_statement="Let v > 0 and assume..."
    )

Maps to Lean:
    namespace Prompts
      namespace Extraction
        def main_template : String
      end Extraction
      namespace Enrichment
        def decompose_theorem_template : String
        def parse_latex_template : String
      end Enrichment
    end Prompts
"""

# Stage 1: Extraction prompts
# Stage 2: Enrichment prompts
from mathster.prompts.enrichment import (
    ANALYZE_PROOF_STRUCTURE_PROMPT,
    DECOMPOSE_THEOREM_PROMPT,
    EXTRACT_SYMPY_CONTEXT_PROMPT,
    get_analyze_proof_prompt,
    get_decompose_theorem_prompt,
    get_extract_sympy_context_prompt,
    get_link_definition_prompt,
    get_link_equation_prompt,
    get_parse_latex_prompt,
    get_resolve_reference_prompt,
    LINK_DEFINITION_TO_OBJECT_PROMPT,
    LINK_EQUATION_TO_ENTITIES_PROMPT,
    PARSE_LATEX_TO_DUAL_PROMPT,
    RESOLVE_REFERENCE_PROMPT,
)
from mathster.prompts.extraction import (
    DEFINITION_ONLY_PROMPT,
    EQUATION_ONLY_PROMPT,
    get_extraction_prompt,
    get_focused_extraction_prompt,
    MAIN_EXTRACTION_PROMPT,
    THEOREM_ONLY_PROMPT,
)


__all__ = [
    "ANALYZE_PROOF_STRUCTURE_PROMPT",
    # Enrichment prompts (Stage 2)
    "DECOMPOSE_THEOREM_PROMPT",
    "DEFINITION_ONLY_PROMPT",
    "EQUATION_ONLY_PROMPT",
    "EXTRACT_SYMPY_CONTEXT_PROMPT",
    "LINK_DEFINITION_TO_OBJECT_PROMPT",
    "LINK_EQUATION_TO_ENTITIES_PROMPT",
    # Extraction prompts (Stage 1)
    "MAIN_EXTRACTION_PROMPT",
    "PARSE_LATEX_TO_DUAL_PROMPT",
    "RESOLVE_REFERENCE_PROMPT",
    "THEOREM_ONLY_PROMPT",
    "get_analyze_proof_prompt",
    "get_decompose_theorem_prompt",
    "get_extract_sympy_context_prompt",
    "get_extraction_prompt",
    "get_focused_extraction_prompt",
    "get_link_definition_prompt",
    "get_link_equation_prompt",
    "get_parse_latex_prompt",
    "get_resolve_reference_prompt",
]
