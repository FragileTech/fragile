#!/usr/bin/env python3
"""
Hybrid parameter extraction: Automated regex + DSPy agent refinement.

This script runs a two-stage parameter extraction pipeline:

Stage 1 (Fast): Automated regex-based extraction
  - Searches for formal definitions ("Let X be", "X := ", etc.)
  - Finds first mentions in LaTeX formulas
  - Success rate: ~86% (398/463 parameters)
  - Time: ~1 second

Stage 2 (Slow): DSPy agent refinement for failures
  - Uses AI to find remaining parameters at line 1
  - Analyzes context and usage to locate definitions
  - Success rate: ~77% of failures (50/65 parameters)
  - Time: ~2-3 minutes (AI API calls)

Total accuracy: ~97% (448/463 parameters)

Usage:
    python examples/run_parameter_extraction_hybrid.py docs/source/1_euclidean_gas/07_mean_field.md
"""

import json
from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mathster.dspy_integration import configure_dspy
from mathster.dspy_integration.text_utils import split_markdown_by_chapters_with_line_numbers
from mathster.parameter_extraction import refine_parameters as refine_parameter_line_numbers


# Import from the simple extraction script
sys.path.insert(0, str(Path(__file__).parent))
from run_parameter_extraction_simple import extract_parameters_simple


# Configure DSPy with Gemini (fast and cheap)
configure_dspy(
    model="gemini/gemini-2.0-flash-exp",
    temperature=0.0,
    max_tokens=10000,  # Reduce tokens (don't need full 20k for line finding)
)


def count_params_at_line_1(chapter_file: Path) -> int:
    """Count how many parameters are at line 1."""
    try:
        with open(chapter_file, encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for param in data.get("parameters", []):
            line_range = param.get("source", {}).get("line_range", {}).get("lines", [[1, 1]])
            if line_range[0][0] == 1:
                count += 1
        return count
    except:
        return 0


def run_stage1_automated(
    doc_path: Path, chapters, full_document_text: str, article_id: str, parser_dir: Path
):
    """Run Stage 1: Automated regex extraction."""
    print("\n" + "=" * 80)
    print("STAGE 1: Automated Regex Extraction")
    print("=" * 80)
    print("Using 6 search patterns:")
    print("  1. Let X be...")
    print("  2. X denotes/represents/:=/=...")
    print("  3. Throughout, X is...")
    print("  4. Parameters: X, Y, Z")
    print("  5. First $X$ in LaTeX")
    print("  6. First mention in text")
    print()

    total_params_extracted = 0
    total_at_line_1 = 0

    for i in range(len(chapters)):
        chapter_file = parser_dir / f"chapter_{i}.json"

        if not chapter_file.exists():
            continue

        # Load existing extraction
        with open(chapter_file, encoding="utf-8") as f:
            chapter_data = json.load(f)

        # Skip if already has parameters
        existing_params = chapter_data.get("parameters", [])
        if existing_params:
            at_line_1 = count_params_at_line_1(chapter_file)
            total_at_line_1 += at_line_1
            total_params_extracted += len(existing_params)
            print(f"Chapter {i}: {len(existing_params)} parameters ({at_line_1} at line 1)")
            continue

        # Extract parameters
        chapter_text = chapters[i]
        raw_parameters = extract_parameters_simple(
            chapter_text=chapter_text,
            full_document_text=full_document_text,
            existing_extraction=chapter_data,
            file_path=str(doc_path),
            article_id=article_id,
        )

        if raw_parameters:
            chapter_data["parameters"] = raw_parameters

            with open(chapter_file, "w", encoding="utf-8") as f:
                json.dump(chapter_data, f, indent=2)

            at_line_1 = count_params_at_line_1(chapter_file)
            total_at_line_1 += at_line_1
            total_params_extracted += len(raw_parameters)

            print(
                f"Chapter {i}: ✓ Extracted {len(raw_parameters)} parameters ({at_line_1} at line 1)"
            )

    print("\n✓ Stage 1 Complete:")
    print(f"  Total parameters: {total_params_extracted}")
    print(
        f"  Correct lines: {total_params_extracted - total_at_line_1} ({(total_params_extracted - total_at_line_1) / total_params_extracted * 100 if total_params_extracted else 0:.1f}%)"
    )
    print(f"  Need refinement: {total_at_line_1}")

    return total_at_line_1


def run_stage2_dspy_refinement(
    doc_path: Path, full_document_text: str, article_id: str, parser_dir: Path
):
    """Run Stage 2: DSPy agent refinement."""
    print("\n" + "=" * 80)
    print("STAGE 2: DSPy Agent Refinement")
    print("=" * 80)
    print("Using AI to find parameters that regex couldn't locate...")
    print("This may take 2-3 minutes for ~65 parameters")
    print()

    total_updated = 0
    total_failed = 0
    all_errors = []

    for chapter_file in sorted(parser_dir.glob("chapter_*.json")):
        updated, failed, errors = refine_parameter_line_numbers(
            chapter_file=chapter_file,
            full_document_text=full_document_text,
            file_path=str(doc_path),
            article_id=article_id,
        )

        total_updated += updated
        total_failed += failed
        all_errors.extend(errors)

    print("\n✓ Stage 2 Complete:")
    print(f"  Updated: {total_updated} parameters")
    print(f"  Failed: {total_failed} parameters")
    if all_errors:
        print(f"  Errors: {len(all_errors)}")
        for err in all_errors[:5]:
            print(f"    - {err}")

    return total_updated, total_failed


def run_hybrid_extraction(document_path: str):
    """Run two-stage hybrid parameter extraction."""
    doc_path = Path(document_path)

    if not doc_path.exists():
        print(f"Error: Document not found: {document_path}")
        return 1

    print(f"Processing: {document_path}")
    print("=" * 80)

    # Setup
    parent_dir = doc_path.parent
    doc_name = doc_path.stem
    parser_dir = parent_dir / "parser"

    if not parser_dir.exists():
        print(f"Error: Parser directory not found: {parser_dir}")
        print("Run main extraction first:")
        print(f"  python -m mathster.parsing.cli {document_path}")
        return 1

    # Load document
    chapters = split_markdown_by_chapters_with_line_numbers(str(doc_path))
    print(f"Found {len(chapters)} chapters")

    # Load full document with line numbers
    with open(doc_path, encoding="utf-8") as f:
        full_doc_lines = f.readlines()
    full_document_text = "\n".join(
        f"{i + 1:03d}: {line.rstrip()}" for i, line in enumerate(full_doc_lines)
    )
    print(f"Document size: {len(full_doc_lines)} lines")

    article_id = doc_name

    # STAGE 1: Automated extraction
    params_needing_refinement = run_stage1_automated(
        doc_path, chapters, full_document_text, article_id, parser_dir
    )

    if params_needing_refinement == 0:
        print("\n✓ All parameters found by automated extraction!")
        print("  No DSPy refinement needed.")
        return 0

    # STAGE 2: DSPy refinement
    updated, failed = run_stage2_dspy_refinement(
        doc_path, full_document_text, article_id, parser_dir
    )

    # Final summary
    print("\n" + "=" * 80)
    print("HYBRID EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Stage 1 (Regex):  {params_needing_refinement} parameters at line 1")
    print(f"Stage 2 (DSPy):   Updated {updated}, Failed {failed}")
    print("\nFinal accuracy:")
    # Rough estimate (would need to recount from files for exact number)
    total_params = params_needing_refinement + updated + failed
    correct = total_params - failed
    print(
        f"  Correct line numbers: ~{correct}/{total_params} (~{correct / total_params * 100 if total_params else 0:.1f}%)"
    )
    print()

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/run_parameter_extraction_hybrid.py <document_path>")
        print()
        print("Example:")
        print(
            "  python examples/run_parameter_extraction_hybrid.py docs/source/1_euclidean_gas/07_mean_field.md"
        )
        print()
        print("This runs:")
        print("  1. Fast regex extraction (86% success)")
        print("  2. DSPy agent refinement for failures (targets 95%+ total)")
        sys.exit(1)

    document_path = sys.argv[1]
    sys.exit(run_hybrid_extraction(document_path))
