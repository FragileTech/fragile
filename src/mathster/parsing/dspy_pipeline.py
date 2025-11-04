"""
DSPy pipeline orchestration for mathematical concept extraction and improvement.

**⚠️ DEPRECATED: This file is legacy code. Use the modular API instead:**

```python
# OLD (deprecated):
from mathster.parsing.dspy_pipeline import process_document, configure_dspy

# NEW (recommended):
from mathster.parsing import process_document, configure_dspy
# Or use CLI:
from mathster.parsing import cli_main
```

**Command-line usage:**
```bash
# OLD (deprecated):
python -m mathster.parsing.dspy_pipeline <file>

# NEW (recommended):
python -m mathster.parsing.cli <file>
```

This module provides the main orchestration logic that routes between two workflows:
1. EXTRACT workflow: Fresh extraction of mathematical concepts from documents
   - Batch mode (default): Extract all labels at once (fast, 1 LLM call)
   - Single-label mode: Extract one label at a time (slower, N LLM calls, more accurate)
2. IMPROVE workflow: Enhancement of existing extractions

The pipeline automatically detects whether a chapter has been previously processed
and routes to the appropriate workflow.

Architecture:
    process_document()
      ├─> For each chapter:
      │     ├─> Check if chapter_{N}.json exists
      │     ├─> If NEW: extract_workflow.extract_chapter() OR extract_chapter_by_labels()
      │     └─> If EXISTS: improve_workflow.improve_chapter()
      └─> Save results with error metadata and change tracking

⚠️ For new code, use: `from mathster.parsing import process_document, cli_main`
        docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
        --extraction-mode single_label

    # Re-run on existing data (improves extractions)
    python -m mathster.parsing.dspy_pipeline \\
        docs/source/1_euclidean_gas/01_fragile_gas_framework.md

Output:
    - chapter_0.json, chapter_1.json, ... (one per chapter)
    - Each JSON contains RawDocumentSection with entity metadata
    - Metadata includes extraction errors and improvement changes
"""

import json
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv
import dspy

from mathster.parsing.extract_workflow import extract_chapter, extract_chapter_by_labels
from mathster.parsing.improve_workflow import improve_chapter
from mathster.parsing.tools import split_markdown_by_chapters_with_line_numbers


# Load environment variables from .env file at import time
load_dotenv()


# =============================================================================
# SHARED UTILITIES
# =============================================================================


def parse_line_number(line: str) -> int | None:
    """
    Extract line number from a numbered line.

    Args:
        line: Line in format "NNN: content" or "  NNN: content"

    Returns:
        The line number or None if format doesn't match
    """
    # Match patterns like "  123: " or "123: "
    match = re.match(r"\s*(\d+):\s", line)
    if match:
        return int(match.group(1))
    return None


def extract_section_id(chapter_text: str, chapter_number: int) -> str:
    """
    Extract section identifier from chapter text.

    Args:
        chapter_text: Chapter text with line numbers
        chapter_number: Chapter index

    Returns:
        Section identifier (e.g., "## 1. Introduction" or "Chapter 0 - Preamble")
    """
    # Find first line with "##" header
    for line in chapter_text.split("\n")[:20]:  # Check first 20 lines
        # Remove line number prefix
        content = re.sub(r"^\s*\d+:\s*", "", line)
        if content.startswith("## "):
            return content.strip()

    # Default fallback
    return f"Chapter {chapter_number}" if chapter_number > 0 else "Preamble"


def configure_dspy(
    model: str = "gemini/gemini-flash-lite-latest",
    temperature: float = 0.5,
    max_tokens: int = 50000,
) -> None:
    """
    Configure DSPy with Claude or Gemini model.

    Args:
        model: Model identifier (default: gemini-flash-lite-latest)
               Supports: anthropic/claude-*, gemini/*
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens per response
    """
    # Ensure environment variables are loaded
    load_dotenv()

    # Determine which API key to use based on model
    if "anthropic" in model.lower():
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            msg = (
                "ANTHROPIC_API_KEY environment variable not set. "
                "Please set it to your Anthropic API key."
            )
            raise ValueError(msg)
    elif "gemini" in model.lower():
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            msg = (
                "GEMINI_API_KEY environment variable not set. "
                "Please set it to your Google API key."
            )
            raise ValueError(msg)
    else:
        raise ValueError(
            f"Unsupported model: {model}. " "Supported models: anthropic/claude-*, gemini/*"
        )

    lm = dspy.LM(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    print(f"✓ Configured DSPy with model: {model}")


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================


def process_document(
    markdown_file: str | Path,
    output_dir: str | Path,
    model: str = "gemini/gemini-flash-lite-latest",  # "anthropic/claude-haiku-4-5",
    max_iters: int = 3,
    skip_chapters: list[int] | None = None,
    extraction_mode: str = "batch",
    improvement_mode: str = "batch",
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True,
) -> None:
    """
    Process entire markdown document chapter by chapter.

    This function orchestrates both EXTRACT and IMPROVE workflows:
    - If chapter_{N}.json exists: run IMPROVE workflow
    - If chapter_{N}.json does NOT exist: run EXTRACT workflow

    Args:
        markdown_file: Path to markdown file to process
        output_dir: Directory to save chapter JSON files
        model: DSPy model identifier (default: gemini-flash-lite-latest)
        max_iters: Maximum ReAct iterations per chapter (default: 3)
        skip_chapters: List of chapter indices to skip (default: None)
        extraction_mode: Extraction strategy - "batch" (all labels at once, fast)
                        or "single_label" (one label at a time, slower but more accurate)
        improvement_mode: Improvement strategy - "batch" (all missed labels at once)
                         or "single_label" (one missed label at a time, with per-label retry)
        max_retries: Maximum number of retry attempts on extraction/improvement failure (default: 3)
        fallback_model: Model to use after first extraction/improvement failure (default: claude-haiku-4-5)
        verbose: Print progress information

    Output:
        Creates/updates chapter_{N}.json files in output_dir
    """
    markdown_file = Path(markdown_file)
    output_dir = Path(output_dir)

    if not markdown_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract article_id from filename (e.g., "01_fragile_gas_framework")
    article_id = markdown_file.stem

    # Configure DSPy
    if verbose:
        print("=" * 80)
        print("DSPy Mathematical Concept Extractor/Improver")
        print("=" * 80)
        print(f"Input file: {markdown_file}")
        print(f"Output dir: {output_dir}")
        print(f"Article ID: {article_id}")
        print()

    configure_dspy(model=model)

    # Split into chapters with line numbers
    if verbose:
        print("Splitting document into chapters...")
    chapters = split_markdown_by_chapters_with_line_numbers(markdown_file)
    if verbose:
        print(f"✓ Found {len(chapters)} chapters")
        print()

    # Process each chapter
    for i, chapter_text in enumerate(chapters):
        # Skip chapters if requested
        if skip_chapters and i in skip_chapters:
            if verbose:
                print(f"Skipping chapter {i} (in skip list)")
                print()
            continue

        if verbose:
            print(f"Processing chapter {i}...")

        output_file = output_dir / f"chapter_{i}.json"
        existing_data = None  # Initialize for later checks

        # Determine workflow based on file existence
        if output_file.exists():
            # IMPROVE WORKFLOW
            if verbose:
                print("  → IMPROVE mode (file exists)")

            # Load existing extraction (we'll preserve this as fallback)
            try:
                with open(output_file, encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"  ✗ Failed to load existing data: {e}")
                print("  → Switching to EXTRACT mode")
                # Treat as new file if we can't load existing data
                existing_data = None

            if existing_data is not None:
                try:
                    # Run improvement workflow
                    raw_section, improvement_result, errors = improve_chapter(
                        chapter_text=chapter_text,
                        existing_extraction=existing_data,
                        file_path=str(markdown_file),
                        article_id=article_id,
                        max_iters=max_iters,
                        improvement_mode=improvement_mode,
                        max_retries=max_retries,
                        fallback_model=fallback_model,
                        verbose=verbose,
                    )

                    # ALWAYS preserve original data as base
                    # Strategy: Start with existing data, then overlay improvements
                    if raw_section:
                        # Improvement succeeded - use improved data
                        save_data = raw_section.model_dump()

                        # Add improvement metadata
                        save_data["_improvement_metadata"] = {
                            "changes": [c.model_dump() for c in improvement_result.changes],
                            "summary": {
                                "entities_added": improvement_result.entities_added,
                                "entities_modified": improvement_result.entities_modified,
                                "entities_deleted": improvement_result.entities_deleted,
                                "entities_unchanged": improvement_result.entities_unchanged,
                            },
                        }

                        if errors:
                            save_data["_improvement_errors"] = errors

                        if verbose:
                            print("  ✓ Improvement successful")
                    else:
                        # Improvement conversion failed - PRESERVE ORIGINAL DATA
                        save_data = existing_data.copy()

                        # Add metadata about failed improvement attempt
                        if "_improvement_attempts" not in save_data:
                            save_data["_improvement_attempts"] = []

                        save_data["_improvement_attempts"].append({
                            "status": "failed",
                            "errors": errors,
                            "summary": improvement_result.get_summary()
                            if improvement_result
                            else "No result",
                        })

                        if verbose:
                            print("  ⚠ Improvement failed - preserving original data")

                    # Save data (either improved or original with error metadata)
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)

                    if verbose:
                        if errors:
                            print(f"  ⚠ Saved to {output_file.name} (with {len(errors)} error(s))")
                            for error in errors[:3]:  # Show first 3 errors
                                print(f"    - {error['error']}")
                            if len(errors) > 3:
                                print(f"    ... and {len(errors) - 3} more")
                        else:
                            print(f"  ✓ Saved to {output_file.name}")

                        if raw_section:
                            print(f"  Entities: {raw_section.total_entities}")
                            print(
                                f"  Changes: +{improvement_result.entities_added} "
                                f"±{improvement_result.entities_modified} "
                                f"-{improvement_result.entities_deleted}"
                            )
                        else:
                            print("  Original data preserved (no changes)")
                        print()

                except Exception as e:
                    # Critical failure in improvement workflow - PRESERVE ORIGINAL DATA
                    print(f"  ✗ Critical error in improvement workflow: {e}")
                    if verbose:
                        import traceback

                        traceback.print_exc()

                    # Save original data with error metadata
                    save_data = existing_data.copy()

                    if "_improvement_attempts" not in save_data:
                        save_data["_improvement_attempts"] = []

                    save_data["_improvement_attempts"].append({
                        "status": "critical_error",
                        "error": str(e),
                        "traceback": traceback.format_exc() if verbose else None,
                    })

                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)

                    print("  ✓ Original data preserved despite error")
                    print()
            else:
                # existing_data is None - fall through to extract mode
                # Will be handled by extract workflow below
                pass

        # EXTRACT WORKFLOW (for new files OR when existing data couldn't be loaded)
        if not output_file.exists() or (output_file.exists() and existing_data is None):
            # EXTRACT WORKFLOW
            if verbose:
                print(f"  → EXTRACT mode (new file, strategy: {extraction_mode})")

            try:
                # Extract section ID
                section_id = extract_section_id(chapter_text, i)
                if verbose:
                    print(f"  Section: {section_id}")

                # Run extraction workflow (choose strategy based on extraction_mode)
                if extraction_mode == "single_label":
                    # Single-label extraction: iterate over each label individually
                    raw_section, errors = extract_chapter_by_labels(
                        chapter_text=chapter_text,
                        chapter_number=i,
                        file_path=str(markdown_file),
                        article_id=article_id,
                        max_iters_per_label=max_iters,
                        max_retries=max_retries,
                        fallback_model=fallback_model,
                        verbose=verbose,
                    )
                else:
                    # Batch extraction: extract all labels at once (default)
                    raw_section, errors = extract_chapter(
                        chapter_text=chapter_text,
                        chapter_number=i,
                        file_path=str(markdown_file),
                        article_id=article_id,
                        max_iters=max_iters,
                        max_retries=max_retries,
                        fallback_model=fallback_model,
                        verbose=verbose,
                    )

                # Prepare save data
                if raw_section:
                    save_data = raw_section.model_dump()
                    if errors:
                        save_data["_extraction_errors"] = errors
                else:
                    save_data = {
                        "status": "extraction_failed",
                        "chapter_number": i,
                        "section_id": section_id,
                        "errors": errors,
                    }

                # Save extracted data
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)

                if verbose:
                    if errors:
                        print(f"  ⚠ Saved to {output_file.name} (with {len(errors)} error(s))")
                        for error in errors[:3]:  # Show first 3 errors
                            print(f"    - {error['error']}")
                        if len(errors) > 3:
                            print(f"    ... and {len(errors) - 3} more")
                    else:
                        print(f"  ✓ Saved to {output_file.name}")

                    if raw_section:
                        print(f"  Extracted: {raw_section.total_entities} entities")
                        print(f"    - Definitions: {len(raw_section.definitions)}")
                        print(f"    - Theorems: {len(raw_section.theorems)}")
                        print(f"    - Proofs: {len(raw_section.proofs)}")
                        print(f"    - Axioms: {len(raw_section.axioms)}")
                        print(f"    - Parameters: {len(raw_section.parameters)}")
                        print(f"    - Remarks: {len(raw_section.remarks)}")
                        print(f"    - Citations: {len(raw_section.citations)}")
                    print()

            except Exception as e:
                # Critical failure in extraction workflow
                print(f"  ✗ Critical error in extraction workflow: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()

                # Save error report
                error_data = {
                    "status": "failed",
                    "chapter_number": i,
                    "error": str(e),
                    "traceback": traceback.format_exc() if verbose else None,
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False)
                print(f"  ✗ Saved error report to {output_file.name}")
                print()

    if verbose:
        print("=" * 80)
        print("✓ Processing complete!")
        print("=" * 80)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """Command-line interface for the DSPy pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract/improve mathematical concepts from markdown documents using DSPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh extraction (batch mode - fast, all labels at once)
  python -m mathster.parsing.dspy_pipeline \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md

  # Single-label extraction (slower but more accurate)
  python -m mathster.parsing.dspy_pipeline \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --extraction-mode single_label

  # Re-run to improve existing extractions
  python -m mathster.parsing.dspy_pipeline \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md

  # Specify custom output directory
  python -m mathster.parsing.dspy_pipeline \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --output-dir custom/output/path

  # Skip certain chapters
  python -m mathster.parsing.dspy_pipeline \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --skip-chapters 0 1

  # Use Sonnet instead of Haiku
  python -m mathster.parsing.dspy_pipeline \\
      docs/source/1_euclidean_gas/01_fragile_gas_framework.md \\
      --model anthropic/claude-sonnet-4-20250514
        """,
    )

    parser.add_argument("markdown_file", type=str, help="Path to markdown file to process")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for chapter JSON files (default: <file_dir>/parser/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-flash-lite-latest",  # "anthropic/claude-haiku-4-5",
        help="DSPy model identifier (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum ReAct iterations per chapter (default: 3)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts on extraction failure (default: 3)",
    )
    parser.add_argument(
        "--fallback-model",
        type=str,
        default="anthropic/claude-haiku-4-5",
        help="Model to use after first extraction failure (default: anthropic/claude-haiku-4-5)",
    )
    parser.add_argument(
        "--skip-chapters",
        type=int,
        nargs="+",
        default=None,
        help="Chapter indices to skip (e.g., --skip-chapters 0 1)",
    )
    parser.add_argument(
        "--extraction-mode",
        type=str,
        choices=["batch", "single_label"],
        default="batch",
        help="Extraction strategy: 'batch' (all labels at once, fast) or "
        "'single_label' (one label at a time, slower but more accurate, default: batch)",
    )
    parser.add_argument(
        "--improvement-mode",
        type=str,
        choices=["batch", "single_label"],
        default="batch",
        help="Improvement strategy: 'batch' (all missed labels at once) or "
        "'single_label' (one missed label at a time with per-label retry, default: batch)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

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
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
