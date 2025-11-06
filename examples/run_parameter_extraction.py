#!/usr/bin/env python3
"""
Example script to run parameter extraction on a document.

This demonstrates how to use the parameter extraction workflow to populate
the parameters array in chapter_N.json files.

Usage:
    python examples/run_parameter_extraction.py docs/source/1_euclidean_gas/07_mean_field.md
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mathster.parsing.config import configure_dspy
from mathster.parsing.text_processing.splitting import split_markdown_by_chapters_with_line_numbers
from mathster.parsing.workflows.extract_parameters import extract_parameters_from_chapter

# Configure DSPy with Gemini
configure_dspy(
    model="gemini/gemini-2.0-flash-exp",
    temperature=0.0,
    max_tokens=20000,
)


def run_parameter_extraction(document_path: str):
    """
    Run parameter extraction on a document.

    Args:
        document_path: Path to markdown document
    """
    doc_path = Path(document_path)

    if not doc_path.exists():
        print(f"Error: Document not found: {document_path}")
        return 1

    print(f"Processing: {document_path}")
    print("=" * 80)

    # Determine paths
    parent_dir = doc_path.parent
    doc_name = doc_path.stem
    parser_dir = parent_dir / "parser"

    if not parser_dir.exists():
        print(f"Error: Parser directory not found: {parser_dir}")
        print("Please run main extraction first:")
        print(f"  python -m mathster.parsing.cli {document_path}")
        return 1

    # Split document into chapters
    chapters = split_markdown_by_chapters_with_line_numbers(str(doc_path))
    print(f"Found {len(chapters)} chapters\n")

    # Extract article_id from path
    # docs/source/1_euclidean_gas/07_mean_field.md → 07_mean_field
    article_id = doc_name

    total_params_extracted = 0
    total_errors = 0

    # Process each chapter
    for i in range(len(chapters)):
        chapter_file = parser_dir / f"chapter_{i}.json"

        if not chapter_file.exists():
            print(f"Chapter {i}: Skipped (no chapter_{i}.json)")
            continue

        print(f"Chapter {i}:")

        # Load existing extraction
        with open(chapter_file) as f:
            chapter_data = json.load(f)

        # Check if parameters already extracted
        existing_params = chapter_data.get("parameters", [])
        if existing_params:
            print(f"  Already has {len(existing_params)} parameters, skipping")
            total_params_extracted += len(existing_params)
            continue

        # Get chapter text
        chapter_text = chapters[i]

        # Extract parameters
        raw_parameters, errors = extract_parameters_from_chapter(
            chapter_text=chapter_text,
            existing_extraction=chapter_data,
            chapter_number=i,
            file_path=str(doc_path),
            article_id=article_id,
        )

        # Update chapter data
        if raw_parameters:
            chapter_data["parameters"] = raw_parameters

            # Save updated chapter
            with open(chapter_file, "w") as f:
                json.dump(chapter_data, f, indent=2)

            print(f"  ✓ Extracted {len(raw_parameters)} parameters")
            for param in raw_parameters:
                symbol = param.get("symbol", "?")
                meaning = param.get("meaning", "")[:60]
                print(f"    - {symbol}: {meaning}...")

            total_params_extracted += len(raw_parameters)
        else:
            print(f"  No parameters extracted")

        if errors:
            print(f"  ✗ {len(errors)} errors")
            total_errors += len(errors)
            for error in errors[:3]:  # Show first 3
                print(f"    - {error.get('error', error)}")

        print()

    # Summary
    print("=" * 80)
    print("Summary:")
    print(f"  ✓ Total parameters extracted: {total_params_extracted}")
    if total_errors:
        print(f"  ✗ Total errors: {total_errors}")
    print()

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/run_parameter_extraction.py <document_path>")
        print()
        print("Example:")
        print("  python examples/run_parameter_extraction.py docs/source/1_euclidean_gas/07_mean_field.md")
        sys.exit(1)

    document_path = sys.argv[1]
    exit(run_parameter_extraction(document_path))
