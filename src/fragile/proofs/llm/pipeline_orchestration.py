"""
Pipeline Orchestration for Extract-then-Enrich.

This module implements the complete Extract-then-Enrich pipeline:

Stage 1: Raw Extraction
  - Split document into sections
  - Extract directive hints (hybrid parsing)
  - Call LLM for each section
  - Get StagingDocument per section

Stage 2: Semantic Enrichment
  - Convert raw entities to enriched models
  - Resolve cross-references
  - Build MathematicalDocument

Maps to Lean:
    namespace PipelineOrchestration
      def process_section : DocumentSection → IO StagingDocument
      def enrich_and_assemble : List StagingDocument → IO MathematicalDocument
      def merge_sections : List StagingDocument → StagingDocument
      def process_document : String → String → IO MathematicalDocument
    end PipelineOrchestration
"""

import logging
from pathlib import Path

from fragile.proofs.core import (
    Axiom,
    DefinitionBox,
    ProofBox,
    TheoremBox,
)
from fragile.proofs.error_tracking import create_logger_for_document, ErrorLogger
from fragile.proofs.llm.document_container import MathematicalDocument
from fragile.proofs.llm.llm_interface import (
    call_main_extraction_llm,
)
from fragile.proofs.prompts import MAIN_EXTRACTION_PROMPT
from fragile.proofs.staging_types import StagingDocument
from fragile.proofs.tools import (
    DocumentSection,
    format_directive_hints_for_llm,
    split_into_sections,
)


logger = logging.getLogger(__name__)


# =============================================================================
# STAGE 1: RAW EXTRACTION
# =============================================================================


def process_section(
    section: DocumentSection,
    prompt_template: str = MAIN_EXTRACTION_PROMPT,
    model: str = "claude-sonnet-4",
    **llm_kwargs,
) -> StagingDocument:
    """
    Process a single section through Stage 1 extraction.

    This function:
    1. Formats directive hints for the LLM
    2. Calls the extraction LLM
    3. Validates the response
    4. Returns StagingDocument

    Args:
        section: The document section to process
        prompt_template: The prompt template to use
        model: LLM model to use
        **llm_kwargs: Additional arguments for LLM call

    Returns:
        StagingDocument with extracted entities

    Examples:
        >>> from fragile.proofs.tools import split_into_sections
        >>> sections = split_into_sections(markdown_text)
        >>> staging_doc = process_section(sections[0])
        >>> staging_doc.section_id
        '§1-introduction'
    """
    logger.info(f"Processing section: {section.section_id}")
    logger.info(f"  Title: {section.title}")
    logger.info(f"  Directives found: {len(section.directives)}")

    # Format directive hints for LLM
    directive_hints = format_directive_hints_for_llm(section.directives)

    # Prepare section text with hints
    section_text_with_hints = f"""
{directive_hints}

---

{section.content}
"""

    # Call extraction LLM
    try:
        result = call_main_extraction_llm(
            section_text=section_text_with_hints,
            section_id=section.section_id,
            prompt_template=prompt_template,
            model=model,
            **llm_kwargs,
        )

        # Validate and return
        staging_doc = StagingDocument.model_validate(result)
        logger.info(f"  Extracted {staging_doc.total_entities} entities")
        return staging_doc

    except Exception as e:
        logger.error(f"Error processing section {section.section_id}: {e}")
        # Return empty staging document on error
        return StagingDocument(
            section_id=section.section_id,
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[],
        )


def process_sections_parallel(
    sections: list[DocumentSection],
    prompt_template: str = MAIN_EXTRACTION_PROMPT,
    model: str = "claude-sonnet-4",
    **llm_kwargs,
) -> list[StagingDocument]:
    """
    Process multiple sections in parallel.

    This is the optimized version that uses batch processing.
    For now, it processes sequentially (parallel implementation requires asyncio).

    Args:
        sections: List of sections to process
        prompt_template: The prompt template to use
        model: LLM model to use
        **llm_kwargs: Additional arguments for LLM calls

    Returns:
        List of StagingDocument objects

    Examples:
        >>> sections = split_into_sections(markdown_text)
        >>> staging_docs = process_sections_parallel(sections)
        >>> len(staging_docs)
        5
    """
    logger.info(f"Processing {len(sections)} sections")

    # TODO: Implement parallel processing with asyncio
    # For now, process sequentially
    staging_docs = []
    for section in sections:
        staging_doc = process_section(
            section, prompt_template=prompt_template, model=model, **llm_kwargs
        )
        staging_docs.append(staging_doc)

    logger.info(f"Completed processing {len(staging_docs)} sections")
    return staging_docs


def merge_sections(staging_docs: list[StagingDocument]) -> StagingDocument:
    """
    Merge multiple section StagingDocuments into a single document.

    This combines all entities from all sections, preserving order.

    Args:
        staging_docs: List of StagingDocument objects to merge

    Returns:
        Single merged StagingDocument

    Examples:
        >>> staging_docs = [stage1, stage2, stage3]
        >>> merged = merge_sections(staging_docs)
        >>> merged.section_id
        'merged-document'
    """
    logger.info(f"Merging {len(staging_docs)} staging documents")

    # Collect all entities
    all_definitions = []
    all_theorems = []
    all_proofs = []
    all_axioms = []
    all_citations = []
    all_equations = []
    all_parameters = []
    all_remarks = []

    for doc in staging_docs:
        all_definitions.extend(doc.definitions)
        all_theorems.extend(doc.theorems)
        all_proofs.extend(doc.proofs)
        all_axioms.extend(doc.axioms)
        all_citations.extend(doc.citations)
        all_equations.extend(doc.equations)
        all_parameters.extend(doc.parameters)
        all_remarks.extend(doc.remarks)

    merged = StagingDocument(
        section_id="merged-document",
        definitions=all_definitions,
        theorems=all_theorems,
        proofs=all_proofs,
        axioms=all_axioms,
        citations=all_citations,
        equations=all_equations,
        parameters=all_parameters,
        remarks=all_remarks,
    )

    logger.info(f"Merged document contains {merged.total_entities} total entities")
    return merged


# =============================================================================
# STAGE 2: SEMANTIC ENRICHMENT
# =============================================================================


def enrich_and_assemble(
    staging_doc: StagingDocument,
    chapter: str | None = None,
    document: str | None = None,
    error_logger: ErrorLogger | None = None,
) -> MathematicalDocument:
    """
    Enrich raw entities and assemble into MathematicalDocument.

    This is Stage 2 of the pipeline. It:
    1. Converts raw staging entities to enriched models
    2. Handles errors gracefully
    3. Builds the final MathematicalDocument

    This is the SIMPLE version that uses from_raw() methods.
    For full semantic enrichment, use the LLM-based enrichment pipeline.

    Args:
        staging_doc: The staging document from Stage 1
        chapter: Chapter identifier
        document: Document identifier
        error_logger: Optional error logger for tracking failures

    Returns:
        MathematicalDocument with enriched entities

    Examples:
        >>> staging_doc = process_section(section)
        >>> math_doc = enrich_and_assemble(staging_doc, chapter="1_euclidean_gas")
        >>> len(math_doc.enriched.theorems)
        12
    """
    logger.info("Starting enrichment and assembly")
    logger.info(f"  Raw entities: {staging_doc.total_entities}")

    # Create document container
    math_doc = MathematicalDocument(
        document_id=document or staging_doc.section_id, chapter=chapter, file_path=None
    )

    # Add staging document
    math_doc = math_doc.add_staging_document(staging_doc)

    # Enrich definitions
    logger.info(f"Enriching {len(staging_doc.definitions)} definitions")
    for raw_def in staging_doc.definitions:
        try:
            if raw_def.source is None:
                raise ValueError(
                    f"Definition {raw_def.temp_id} missing source location. "
                    "Run source_location_enricher before transformation."
                )
            enriched = DefinitionBox.from_raw(
                raw_def, source=raw_def.source, chapter=chapter, document=document
            )
            math_doc = math_doc.add_enriched_definition(enriched)
        except Exception as e:
            logger.error(f"Failed to enrich definition {raw_def.temp_id}: {e}")
            if error_logger:
                error_logger.log_error(
                    error_type="ENRICHMENT_ERROR",
                    message=f"Failed to enrich definition: {e}",
                    entity_id=raw_def.label_text,
                    entity_type="definition",
                )

    # Enrich theorems
    logger.info(f"Enriching {len(staging_doc.theorems)} theorems")
    for raw_thm in staging_doc.theorems:
        try:
            if raw_thm.source is None:
                raise ValueError(
                    f"Theorem {raw_thm.temp_id} missing source location. "
                    "Run source_location_enricher before transformation."
                )
            enriched = TheoremBox.from_raw(
                raw_thm, source=raw_thm.source, chapter=chapter, document=document
            )
            math_doc = math_doc.add_enriched_theorem(enriched)
        except Exception as e:
            logger.error(f"Failed to enrich theorem {raw_thm.temp_id}: {e}")
            if error_logger:
                error_logger.log_error(
                    error_type="ENRICHMENT_ERROR",
                    message=f"Failed to enrich theorem: {e}",
                    entity_id=raw_thm.label_text,
                    entity_type="theorem",
                )

    # Enrich axioms
    logger.info(f"Enriching {len(staging_doc.axioms)} axioms")
    for raw_axiom in staging_doc.axioms:
        try:
            if raw_axiom.source is None:
                raise ValueError(
                    f"Axiom {raw_axiom.temp_id} missing source location. "
                    "Run source_location_enricher before transformation."
                )
            enriched = Axiom.from_raw(
                raw_axiom, source=raw_axiom.source, chapter=chapter, document=document
            )
            math_doc = math_doc.add_enriched_axiom(enriched)
        except Exception as e:
            logger.error(f"Failed to enrich axiom {raw_axiom.temp_id}: {e}")
            if error_logger:
                error_logger.log_error(
                    error_type="ENRICHMENT_ERROR",
                    message=f"Failed to enrich axiom: {e}",
                    entity_id=raw_axiom.label_text,
                    entity_type="axiom",
                )

    # Enrich proofs
    logger.info(f"Enriching {len(staging_doc.proofs)} proofs")
    for raw_proof in staging_doc.proofs:
        try:
            if raw_proof.source is None:
                raise ValueError(
                    f"Proof {raw_proof.temp_id} missing source location. "
                    "Run source_location_enricher before transformation."
                )
            # Need to determine what theorem this proves
            proves = raw_proof.proves_label_text
            enriched = ProofBox.from_raw(raw_proof, source=raw_proof.source, proves=proves)
            math_doc = math_doc.add_enriched_proof(enriched)
        except Exception as e:
            logger.error(f"Failed to enrich proof {raw_proof.temp_id}: {e}")
            if error_logger:
                error_logger.log_error(
                    error_type="ENRICHMENT_ERROR",
                    message=f"Failed to enrich proof: {e}",
                    entity_id=raw_proof.temp_id,
                    entity_type="proof",
                )

    logger.info("Enrichment complete")
    logger.info(f"  Enriched entities: {math_doc.total_enriched_entities}")
    logger.info(f"  Enrichment rate: {math_doc.enrichment_rate:.1f}%")

    return math_doc


# =============================================================================
# END-TO-END PIPELINE
# =============================================================================


def process_document(
    markdown_text: str,
    document_id: str,
    chapter: str | None = None,
    file_path: str | None = None,
    model: str = "claude-sonnet-4",
    enable_error_logging: bool = True,
    log_dir: str | None = None,
    **llm_kwargs,
) -> MathematicalDocument:
    """
    Complete end-to-end Extract-then-Enrich pipeline.

    This is the main entry point for processing a document. It:
    1. Splits document into sections
    2. Extracts raw entities (Stage 1)
    3. Enriches entities (Stage 2)
    4. Returns MathematicalDocument

    Args:
        markdown_text: Full markdown document content
        document_id: Unique document identifier
        chapter: Chapter identifier (e.g., "1_euclidean_gas")
        file_path: Path to source file (for metadata)
        model: LLM model to use
        enable_error_logging: Whether to log errors to file
        log_dir: Directory for log files
        **llm_kwargs: Additional arguments for LLM calls

    Returns:
        Complete MathematicalDocument with raw and enriched entities

    Examples:
        >>> with open("01_fragile_gas_framework.md") as f:
        ...     markdown_text = f.read()
        >>> math_doc = process_document(
        ...     markdown_text,
        ...     document_id="01_fragile_gas_framework",
        ...     chapter="1_euclidean_gas",
        ... )
        >>> print(math_doc.get_summary())
    """
    logger.info("=" * 60)
    logger.info(f"STARTING DOCUMENT PROCESSING: {document_id}")
    logger.info("=" * 60)

    # Set up error logging
    error_logger = None
    if enable_error_logging:
        error_logger = create_logger_for_document(document_id, log_dir=log_dir)
        logger.info(f"Error logging enabled: {error_logger.log_file}")

    try:
        # Stage 0: Split into sections
        logger.info("Stage 0: Splitting document into sections")
        sections = split_into_sections(markdown_text, file_path=file_path or document_id)
        logger.info(f"  Found {len(sections)} sections")

        # Stage 1: Raw extraction
        logger.info("Stage 1: Raw extraction")
        staging_docs = process_sections_parallel(sections, model=model, **llm_kwargs)

        # Merge sections
        merged_staging = merge_sections(staging_docs)
        logger.info(f"  Total raw entities: {merged_staging.total_entities}")

        # Stage 2: Enrichment
        logger.info("Stage 2: Semantic enrichment")
        math_doc = enrich_and_assemble(
            merged_staging, chapter=chapter, document=document_id, error_logger=error_logger
        )

        # Update file_path in document
        if file_path:
            math_doc = math_doc.model_copy(update={"file_path": file_path})

        # Print summary
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\n{math_doc.get_summary()}")

        # Save error log if enabled
        if error_logger:
            error_logger.print_summary()
            report_path = error_logger.save_report()
            logger.info(f"Error report saved: {report_path}")

        return math_doc

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        if error_logger:
            error_logger.log_error(
                error_type="PIPELINE_ERROR",
                message=f"Document processing failed: {e}",
                entity_id=document_id,
                entity_type="document",
            )
            error_logger.save_report()
        raise


def process_document_from_file(
    file_path: str, chapter: str | None = None, model: str = "claude-sonnet-4", **llm_kwargs
) -> MathematicalDocument:
    """
    Process a markdown file through the complete pipeline.

    Convenience wrapper around process_document() that reads from file.

    Args:
        file_path: Path to markdown file
        chapter: Chapter identifier (auto-detected from path if not provided)
        model: LLM model to use
        **llm_kwargs: Additional arguments for LLM calls

    Returns:
        Complete MathematicalDocument

    Examples:
        >>> math_doc = process_document_from_file(
        ...     "docs/source/1_euclidean_gas/01_fragile_gas_framework.md",
        ...     model="claude-sonnet-4",
        ... )
        >>> math_doc.document_id
        '01_fragile_gas_framework'
    """
    path = Path(file_path)

    # Read file
    with open(path, encoding="utf-8") as f:
        markdown_text = f.read()

    # Auto-detect chapter from path if not provided
    if chapter is None:
        # Look for pattern like "1_euclidean_gas" or "2_geometric_gas" in path
        parts = path.parts
        for part in parts:
            if part.startswith(("1_", "2_", "3_")):
                chapter = part
                break

    # Generate document_id from filename
    document_id = path.stem

    return process_document(
        markdown_text=markdown_text,
        document_id=document_id,
        chapter=chapter,
        file_path=str(path),
        model=model,
        **llm_kwargs,
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_multiple_documents(
    file_paths: list[str], chapter: str | None = None, model: str = "claude-sonnet-4", **llm_kwargs
) -> dict[str, MathematicalDocument]:
    """
    Process multiple documents sequentially.

    Args:
        file_paths: List of paths to markdown files
        chapter: Chapter identifier (used if not auto-detected)
        model: LLM model to use
        **llm_kwargs: Additional arguments for LLM calls

    Returns:
        Dictionary mapping document_id to MathematicalDocument

    Examples:
        >>> files = [
        ...     "docs/source/1_euclidean_gas/01_framework.md",
        ...     "docs/source/1_euclidean_gas/02_euclidean_gas.md",
        ... ]
        >>> results = process_multiple_documents(files)
        >>> len(results)
        2
    """
    logger.info(f"Processing {len(file_paths)} documents")

    results = {}
    for file_path in file_paths:
        try:
            math_doc = process_document_from_file(
                file_path, chapter=chapter, model=model, **llm_kwargs
            )
            results[math_doc.document_id] = math_doc
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    logger.info(f"Successfully processed {len(results)}/{len(file_paths)} documents")
    return results
