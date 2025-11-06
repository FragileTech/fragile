#!/usr/bin/env python3
"""
Simplified parameter extraction without DSPy (much faster).

This demonstrates the parameter extraction pipeline using direct regex matching
instead of AI agents. Suitable for quick parameter extraction.

Usage:
    python examples/run_parameter_extraction_simple.py docs/source/1_euclidean_gas/07_mean_field.md
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mathster.parsing.conversion.labels import sanitize_label
from mathster.parsing.conversion.sources import create_source_location
from mathster.parsing.text_processing.analysis import (
    collect_parameters_from_extraction,
    find_parameter_declarations,
)
from mathster.parsing.text_processing.splitting import split_markdown_by_chapters_with_line_numbers


def extract_parameters_simple(
    chapter_text: str,
    existing_extraction: dict,
    file_path: str,
    article_id: str,
) -> list[dict]:
    """
    Simple parameter extraction without AI.

    Args:
        chapter_text: Chapter text with line numbers
        existing_extraction: Existing chapter extraction
        file_path: Source file path
        article_id: Article ID

    Returns:
        List of RawParameter dicts
    """
    # Step 1: Collect parameters mentioned
    parameters_mentioned = collect_parameters_from_extraction(existing_extraction)

    if not parameters_mentioned:
        return []

    # Step 2: Find declarations in text
    declarations = find_parameter_declarations(chapter_text, list(parameters_mentioned))

    # Step 3: Create Parameter objects
    raw_parameters = []

    for symbol in parameters_mentioned:
        decl = declarations.get(symbol)

        if not decl:
            # No declaration found - create minimal parameter
            # Infer meaning from symbol name
            meaning = infer_meaning_from_symbol(symbol)
            label = f"param-{sanitize_label(symbol)}"

            # Use first line as placeholder
            raw_param = {
                "label": label,
                "symbol": symbol,
                "meaning": meaning,
                "scope": "global",  # Default
                "full_text": "",
                "source": create_source_location(
                    label, 1, 1, file_path, article_id
                ).model_dump(),
            }
        else:
            # Use declaration info
            meaning = extract_meaning_from_context(decl["context"], symbol)
            label = f"param-{sanitize_label(symbol)}"

            raw_param = {
                "label": label,
                "symbol": symbol,
                "meaning": meaning,
                "scope": "global",  # Default (could infer from pattern)
                "full_text": decl["context"],
                "source": create_source_location(
                    label, decl["line_start"], decl["line_end"], file_path, article_id
                ).model_dump(),
            }

        raw_parameters.append(raw_param)

    return raw_parameters


def infer_meaning_from_symbol(symbol: str) -> str:
    """Infer basic meaning from symbol name."""
    # Common mappings
    meanings = {
        "tau": "Time step size",
        "alpha": "Exploitation weight parameter",
        "beta": "Diversity weight parameter",
        "gamma": "Friction coefficient",
        "gamma_fric": "Friction coefficient",
        "epsilon": "Regularization parameter",
        "sigma": "Noise scale parameter",
        "N": "Number of walkers",
        "m": "Mass parameter",
        "d": "Dimension",
        "T": "Temperature",
        "Theta": "Temperature parameter",
    }

    return meanings.get(symbol, f"Parameter {symbol}")


def extract_meaning_from_context(context: str, symbol: str) -> str:
    """Extract meaning from declaration context."""
    # Simple extraction: take text after "is", "denotes", "represents"
    import re

    # Try to find "X is/denotes/represents Y"
    pattern = rf"{re.escape(symbol)}\s+(is|denotes?|represents?)\s+(.+?)(\.|$)"
    match = re.search(pattern, context, re.IGNORECASE)

    if match:
        meaning = match.group(2).strip()
        # Clean up
        meaning = meaning.split(".")[0]  # Take first sentence
        return meaning if meaning else infer_meaning_from_symbol(symbol)

    # Fallback: use inferred meaning
    return infer_meaning_from_symbol(symbol)


def run_parameter_extraction(document_path: str):
    """Run simplified parameter extraction."""
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
        return 1

    # Split document into chapters
    chapters = split_markdown_by_chapters_with_line_numbers(str(doc_path))
    print(f"Found {len(chapters)} chapters\n")

    article_id = doc_name
    total_params_extracted = 0

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

        # Simple extraction (no AI)
        raw_parameters = extract_parameters_simple(
            chapter_text=chapter_text,
            existing_extraction=chapter_data,
            file_path=str(doc_path),
            article_id=article_id,
        )

        # Update and save
        if raw_parameters:
            chapter_data["parameters"] = raw_parameters

            with open(chapter_file, "w") as f:
                json.dump(chapter_data, f, indent=2)

            print(f"  ✓ Extracted {len(raw_parameters)} parameters")
            for param in raw_parameters[:5]:  # Show first 5
                symbol = param.get("symbol", "?")
                meaning = param.get("meaning", "")[:60]
                print(f"    - {symbol}: {meaning}")
            if len(raw_parameters) > 5:
                print(f"    ... and {len(raw_parameters) - 5} more")

            total_params_extracted += len(raw_parameters)
        else:
            print(f"  No parameters found")

        print()

    # Summary
    print("=" * 80)
    print("Summary:")
    print(f"  ✓ Total parameters extracted: {total_params_extracted}")
    print()

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/run_parameter_extraction_simple.py <document_path>")
        print()
        print("Example:")
        print("  python examples/run_parameter_extraction_simple.py docs/source/1_euclidean_gas/07_mean_field.md")
        sys.exit(1)

    document_path = sys.argv[1]
    exit(run_parameter_extraction(document_path))
