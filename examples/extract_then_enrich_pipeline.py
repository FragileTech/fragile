"""
Complete Extract-then-Enrich Pipeline Example.

This script demonstrates the end-to-end workflow for processing mathematical
documents through the Extract-then-Enrich pipeline:

1. Stage 0: Document Splitting
2. Stage 1: Raw Extraction (with LLM)
3. Stage 2: Semantic Enrichment
4. Final: MathematicalDocument Assembly

Usage:
    # Process a single document
    python examples/extract_then_enrich_pipeline.py \
        --file docs/source/1_euclidean_gas/01_fragile_gas_framework.md

    # Process multiple documents
    python examples/extract_then_enrich_pipeline.py \
        --directory docs/source/1_euclidean_gas \
        --pattern "*.md"

Requirements:
    - Set ANTHROPIC_API_KEY environment variable for actual LLM calls
    - Or run with --mock flag to use stub implementations
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the pipeline components
from fragile.proofs import (
    # Pipeline orchestration
    process_document_from_file,
    process_multiple_documents,
    # Document splitting and parsing
    split_into_sections,
    extract_jupyter_directives,
    # Container types
    MathematicalDocument,
    # Staging types (optional - for inspection)
    StagingDocument,
)


def example_1_basic_processing():
    """
    Example 1: Process a single markdown document.

    This is the simplest use case - just provide a file path and let
    the pipeline handle everything automatically.
    """
    logger.info("="*60)
    logger.info("EXAMPLE 1: Basic Document Processing")
    logger.info("="*60)

    # Path to a sample document
    file_path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"

    # Check if file exists
    if not Path(file_path).exists():
        logger.warning(f"File not found: {file_path}")
        logger.info("Skipping Example 1")
        return None

    # Process the document
    logger.info(f"Processing: {file_path}")
    math_doc = process_document_from_file(
        file_path=file_path,
        model="claude-sonnet-4"  # Uses stub implementation by default
    )

    # Print summary
    logger.info("\n" + math_doc.get_summary())

    return math_doc


def example_2_inspect_stages():
    """
    Example 2: Inspect intermediate pipeline stages.

    This shows how to manually control each stage of the pipeline
    for debugging or custom processing.
    """
    logger.info("="*60)
    logger.info("EXAMPLE 2: Inspect Pipeline Stages")
    logger.info("="*60)

    file_path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"

    if not Path(file_path).exists():
        logger.warning(f"File not found: {file_path}")
        logger.info("Skipping Example 2")
        return

    # Read the file
    with open(file_path, 'r') as f:
        markdown_text = f.read()

    # Stage 0: Split into sections
    logger.info("Stage 0: Splitting document into sections")
    sections = split_into_sections(markdown_text, file_path=file_path)
    logger.info(f"  Found {len(sections)} sections")

    # Inspect first section
    if sections:
        first_section = sections[0]
        logger.info(f"\nFirst Section:")
        logger.info(f"  ID: {first_section.section_id}")
        logger.info(f"  Title: {first_section.title}")
        logger.info(f"  Level: {first_section.level}")
        logger.info(f"  Lines: {first_section.start_line}-{first_section.end_line}")
        logger.info(f"  Directives found: {len(first_section.directives)}")

        # Show directive types
        if first_section.directives:
            directive_types = {}
            for directive in first_section.directives:
                directive_types[directive.directive_type] = \
                    directive_types.get(directive.directive_type, 0) + 1
            logger.info(f"  Directive breakdown: {directive_types}")


def example_3_batch_processing():
    """
    Example 3: Process multiple documents in batch.

    This shows how to process an entire directory of documents.
    """
    logger.info("="*60)
    logger.info("EXAMPLE 3: Batch Document Processing")
    logger.info("="*60)

    # Find all markdown files in the Euclidean Gas chapter
    doc_dir = Path("docs/source/1_euclidean_gas")

    if not doc_dir.exists():
        logger.warning(f"Directory not found: {doc_dir}")
        logger.info("Skipping Example 3")
        return

    # Collect markdown files
    markdown_files = list(doc_dir.glob("*.md"))
    if not markdown_files:
        logger.warning(f"No markdown files found in {doc_dir}")
        logger.info("Skipping Example 3")
        return

    logger.info(f"Found {len(markdown_files)} markdown files")

    # Process first 3 files (for speed)
    files_to_process = [str(f) for f in markdown_files[:3]]

    logger.info(f"Processing {len(files_to_process)} documents...")
    results = process_multiple_documents(
        file_paths=files_to_process,
        chapter="1_euclidean_gas",
        model="claude-sonnet-4"
    )

    # Print summary for each
    logger.info(f"\nProcessed {len(results)} documents successfully:")
    for doc_id, math_doc in results.items():
        logger.info(f"\n{doc_id}:")
        logger.info(f"  Raw entities: {math_doc.total_raw_entities}")
        logger.info(f"  Enriched entities: {math_doc.total_enriched_entities}")
        logger.info(f"  Enrichment rate: {math_doc.enrichment_rate:.1f}%")


def example_4_custom_enrichment():
    """
    Example 4: Custom enrichment with manual control.

    This shows how to use the from_raw() methods directly for
    custom processing workflows.
    """
    logger.info("="*60)
    logger.info("EXAMPLE 4: Custom Enrichment")
    logger.info("="*60)

    from fragile.proofs import (
        RawTheorem,
        RawAxiom,
        TheoremBox,
        Axiom,
    )

    # Create a raw theorem (simulating extraction)
    raw_theorem = RawTheorem(
        temp_id="raw-thm-001",
        label_text="thm-example",
        name="Example Convergence Theorem",
        statement_text="Under the stated axioms, the system converges exponentially.",
        informal_explanation="This theorem establishes exponential convergence.",
        source_section="§3"
    )

    # Enrich using from_raw()
    logger.info("Enriching theorem using from_raw()...")
    enriched_theorem = TheoremBox.from_raw(
        raw_theorem,
        chapter="1_euclidean_gas",
        document="example_document"
    )

    logger.info(f"\nEnriched Theorem:")
    logger.info(f"  Label: {enriched_theorem.label}")
    logger.info(f"  Name: {enriched_theorem.name}")
    logger.info(f"  Output Type: {enriched_theorem.output_type}")
    logger.info(f"  Statement Type: {enriched_theorem.statement_type}")

    # Create and enrich an axiom
    raw_axiom = RawAxiom(
        temp_id="raw-axiom-001",
        label_text="axiom-bounded-displacement",
        name="Axiom of Bounded Displacement",
        core_assumption_text="All walkers satisfy |x(t+Δt) - x(t)| ≤ ε√Δt",
        parameters_text=["ε > 0", "Δt"],
        condition_text="When Δt < ε²",
        failure_mode_analysis_text="Unphysical teleportation behavior",
        source_section="§1"
    )

    logger.info("\nEnriching axiom using from_raw()...")
    enriched_axiom = Axiom.from_raw(
        raw_axiom,
        chapter="1_euclidean_gas",
        document="example_document"
    )

    logger.info(f"\nEnriched Axiom:")
    logger.info(f"  Label: {enriched_axiom.label}")
    logger.info(f"  Name: {enriched_axiom.name}")
    logger.info(f"  Parameters: {len(enriched_axiom.parameters or [])}")
    logger.info(f"  Has failure mode: {enriched_axiom.failure_mode_analysis is not None}")


def example_5_save_and_load():
    """
    Example 5: Save and load MathematicalDocument.

    This shows how to persist the processed document for later use.
    """
    logger.info("="*60)
    logger.info("EXAMPLE 5: Save and Load Document")
    logger.info("="*60)

    file_path = "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"

    if not Path(file_path).exists():
        logger.warning(f"File not found: {file_path}")
        logger.info("Skipping Example 5")
        return

    # Process document
    logger.info("Processing document...")
    math_doc = process_document_from_file(file_path)

    # Save to JSON
    output_file = Path("examples/output/mathematical_document.json")
    output_file.parent.mkdir(exist_ok=True)

    logger.info(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(math_doc.model_dump(), f, indent=2, default=str)

    logger.info(f"Saved {output_file.stat().st_size} bytes")

    # Load it back
    logger.info("Loading from JSON...")
    with open(output_file, 'r') as f:
        loaded_data = json.load(f)

    # Reconstruct
    from fragile.proofs import MathematicalDocument
    loaded_doc = MathematicalDocument.model_validate(loaded_data)

    logger.info(f"\nLoaded document:")
    logger.info(f"  ID: {loaded_doc.document_id}")
    logger.info(f"  Chapter: {loaded_doc.chapter}")
    logger.info(f"  Enriched entities: {loaded_doc.total_enriched_entities}")


def main():
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(
        description="Extract-then-Enrich Pipeline Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a specific example (1-5)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process a specific file (for Example 1)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )

    args = parser.parse_args()

    # Determine which examples to run
    if args.all:
        examples = [1, 2, 3, 4, 5]
    elif args.example:
        examples = [args.example]
    else:
        # Default: run all examples
        examples = [1, 2, 3, 4, 5]

    # Run selected examples
    for example_num in examples:
        try:
            if example_num == 1:
                example_1_basic_processing()
            elif example_num == 2:
                example_2_inspect_stages()
            elif example_num == 3:
                example_3_batch_processing()
            elif example_num == 4:
                example_4_custom_enrichment()
            elif example_num == 5:
                example_5_save_and_load()
            print()  # Blank line between examples
        except Exception as e:
            logger.error(f"Example {example_num} failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("="*60)
    logger.info("EXAMPLES COMPLETE")
    logger.info("="*60)
    logger.info("\nNote: These examples use stub LLM implementations by default.")
    logger.info("For actual LLM calls, set ANTHROPIC_API_KEY and update")
    logger.info("llm_interface.py with production API code.")


if __name__ == "__main__":
    main()
