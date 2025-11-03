"""
Command-line interface for the mathematical entity extraction pipeline.

Provides a user-friendly CLI for extracting and improving mathematical
entities from markdown documents using DSPy.
"""

import argparse
from pathlib import Path

from mathster.parsing.orchestrator import process_document


def main():
    """Command-line interface for the DSPy pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract/improve mathematical concepts from markdown documents using DSPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh extraction (batch mode - fast, all labels at once)
  python -m mathster.parsing.cli \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md

  # Single-label extraction (slower but more accurate)
  python -m mathster.parsing.cli \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --extraction-mode single_label

  # Re-run to improve existing extractions
  python -m mathster.parsing.cli \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md

  # Specify custom output directory
  python -m mathster.parsing.cli \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --output-dir custom/output/path

  # Skip certain chapters
  python -m mathster.parsing.cli \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --skip-chapters 0 1

  # Use Sonnet instead of Haiku
  python -m mathster.parsing.cli \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --model anthropic/claude-sonnet-4-20250514
        """
    )

    parser.add_argument(
        "markdown_file",
        type=str,
        help="Path to markdown file to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for chapter JSON files (default: <file_dir>/parser/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="DSPy model identifier (default: gemini-flash-lite-latest)"
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum ReAct iterations per chapter (default: 3)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts on extraction failure (default: 3)"
    )
    parser.add_argument(
        "--fallback-model",
        type=str,
        default="anthropic/claude-haiku-4-5",
        help="Model to use after first extraction failure (default: anthropic/claude-haiku-4-5)"
    )
    parser.add_argument(
        "--skip-chapters",
        type=int,
        nargs="+",
        default=None,
        help="Chapter indices to skip (e.g., --skip-chapters 0 1)"
    )
    parser.add_argument(
        "--extraction-mode",
        type=str,
        choices=["batch", "single_label"],
        default="batch",
        help="Extraction strategy: 'batch' (all labels at once, fast) or "
             "'single_label' (one label at a time, slower but more accurate, default: batch)"
    )
    parser.add_argument(
        "--improvement-mode",
        type=str,
        choices=["batch", "single_label"],
        default="batch",
        help="Improvement strategy: 'batch' (all missed labels at once) or "
             "'single_label' (one missed label at a time with per-label retry, default: batch)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Determine output directory
    markdown_path = Path(args.markdown_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: create 'parser' subdirectory next to source file
        output_dir = markdown_path.parent / "parser"

    # Process document
    try:
        process_document(
            markdown_file=markdown_path,
            output_dir=output_dir,
            model=args.model,
            max_iters=args.max_iters,
            skip_chapters=args.skip_chapters,
            extraction_mode=args.extraction_mode,
            improvement_mode=args.improvement_mode,
            max_retries=args.max_retries,
            fallback_model=args.fallback_model,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
