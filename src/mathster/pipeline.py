"""
Extract-then-Enrich Pipeline CLI.

Command-line interface for the two-stage mathematical document processing pipeline:
- Stage 1: Raw extraction (LLM-based verbatim transcription)
- Stage 2: Semantic enrichment (cross-referencing, validation)

Usage:
    # Extract raw data from document
    python -m fragile.mathster.pipeline extract <document_path>

    # Extract with custom model
    python -m fragile.mathster.pipeline extract <document_path> --model claude-sonnet-4

    # Extract with custom output directory
    python -m fragile.mathster.pipeline extract <document_path> --output-dir /path/to/output

Examples:
    # Extract single document
    python -m fragile.mathster.pipeline extract docs/source/1_euclidean_gas/03_cloning.md

    # Extract with GPT-4
    python -m fragile.mathster.pipeline extract docs/source/1_euclidean_gas/03_cloning.md \\
        --model gpt-4

    # Extract to custom directory
    python -m fragile.mathster.pipeline extract docs/source/1_euclidean_gas/03_cloning.md \\
        --output-dir /tmp/extraction_output
"""

import argparse
import logging
from pathlib import Path
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for CLI
)
logger = logging.getLogger(__name__)


def extract_command(args):
    """
    Execute Stage 1: Raw extraction.

    Args:
        args: Parsed command-line arguments
    """
    from fragile.agents.raw_document_parser import extract_document

    source_path = Path(args.source)

    # Validate source exists
    if not source_path.exists():
        logger.error(f"❌ Source file not found: {source_path}")
        sys.exit(1)

    if not source_path.is_file():
        logger.error(f"❌ Source must be a file: {source_path}")
        sys.exit(1)

    # Run extraction
    try:
        result = extract_document(
            source_path=source_path, output_dir=args.output_dir, model=args.model
        )

        # Print summary
        stats = result["statistics"]
        logger.info("")
        logger.info("📊 Extraction Summary:")
        logger.info(f"   Total entities: {stats['total_entities']}")
        logger.info(f"   Definitions: {stats['entities_extracted']['definitions']}")
        logger.info(f"   Theorems: {stats['entities_extracted']['theorems']}")
        logger.info(f"   Axioms: {stats['entities_extracted']['axioms']}")
        logger.info(f"   Proofs: {stats['entities_extracted']['mathster']}")
        logger.info(f"   Equations: {stats['entities_extracted']['equations']}")
        logger.info(f"   Parameters: {stats['entities_extracted']['parameters']}")
        logger.info(f"   Remarks: {stats['entities_extracted']['remarks']}")
        logger.info(f"   Citations: {stats['entities_extracted']['citations']}")
        logger.info("")
        logger.info(f"📁 Output directory: {result['output_dir']}/raw_data/")
        logger.info(f"⏱️  Elapsed time: {result['elapsed_seconds']:.1f}s")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract-then-Enrich Pipeline for Mathematical Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract raw data from document
  python -m fragile.mathster.pipeline extract docs/source/1_euclidean_gas/03_cloning.md

  # Extract with custom model
  python -m fragile.mathster.pipeline extract docs/source/1_euclidean_gas/03_cloning.md \\
      --model claude-sonnet-4

  # Extract to custom output directory
  python -m fragile.mathster.pipeline extract docs/source/1_euclidean_gas/03_cloning.md \\
      --output-dir /tmp/output

For more information, see:
  - .claude/agents/document-parser.md (Stage 1: Raw Extraction)
  - .claude/agents/document-refiner.md (Stage 2: Semantic Enrichment)
        """,
    )

    subparsers = parser.add_subparsers(
        title="commands", description="Available pipeline commands", dest="command", required=True
    )

    # =========================================================================
    # Extract Command (Stage 1)
    # =========================================================================
    extract_parser = subparsers.add_parser(
        "extract",
        help="Stage 1: Extract raw data from MyST markdown document",
        description="Perform LLM-based raw extraction of mathematical entities",
    )

    extract_parser.add_argument("source", type=str, help="Path to MyST markdown document")

    extract_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: auto-detect from source path)",
    )

    extract_parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4",
        help="LLM model to use for extraction (default: claude-sonnet-4)",
    )

    extract_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    extract_parser.set_defaults(func=extract_command)

    # =========================================================================
    # Parse and Execute
    # =========================================================================
    args = parser.parse_args()

    # Enable verbose logging if requested
    if hasattr(args, "verbose") and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
