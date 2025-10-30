"""
LLM Interface for Mathematical Paper Extraction.

This module provides placeholder function signatures for LLM API calls
used in the Extract-then-Enrich pipeline. The actual implementations
should be provided by Claude Code or the user.

Usage:
    from fragile.proofs.llm import call_main_extraction_llm

    # Claude Code will implement this function
    result = call_main_extraction_llm(
        section_text=markdown_content,
        section_id="§2.1",
        prompt_template=prompt
    )

Maps to Lean:
    namespace LLMInterface
      def call_main_extraction_llm : String → String → String → IO (HashMap String Any)
      def call_semantic_parser_llm : String → Option String → IO (HashMap String Any)
      def call_batch_extraction_llm : List String → String → IO (List (HashMap String Any))
    end LLMInterface
"""

from fragile.proofs.llm.document_container import (
    EnrichedEntities,
    MathematicalDocument,
)
from fragile.proofs.llm.llm_interface import (
    call_batch_extraction_llm,
    call_main_extraction_llm,
    call_semantic_parser_llm,
    estimate_tokens,
    extract_json_from_markdown,
    mock_main_extraction_llm,
    mock_semantic_parser_llm,
    validate_llm_response,
)
from fragile.proofs.llm.pipeline_orchestration import (
    enrich_and_assemble,
    merge_sections,
    process_document,
    process_document_from_file,
    process_multiple_documents,
    process_section,
    process_sections_parallel,
)


__all__ = [
    "EnrichedEntities",
    # Document container
    "MathematicalDocument",
    "call_batch_extraction_llm",
    # Main LLM interface functions (to be implemented)
    "call_main_extraction_llm",
    "call_semantic_parser_llm",
    "enrich_and_assemble",
    # Utility functions (to be implemented)
    "estimate_tokens",
    "extract_json_from_markdown",
    "merge_sections",
    # Mock implementations (for testing only)
    "mock_main_extraction_llm",
    "mock_semantic_parser_llm",
    # Pipeline orchestration (end-to-end)
    "process_document",
    "process_document_from_file",
    "process_multiple_documents",
    "process_section",
    "process_sections_parallel",
    "validate_llm_response",
]
