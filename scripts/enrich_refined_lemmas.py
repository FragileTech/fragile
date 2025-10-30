#!/usr/bin/env python3
"""
Custom enrichment script for refined lemma files.

Refined lemmas have different structure than raw entities:
- Use 'label' instead of 'label_text'
- Use 'natural_language_statement' instead of 'full_statement_text'
- Use 'name' for the lemma title

This script:
1. Searches for lemma by title (name field) in markdown
2. Finds the :label: line below the title
3. Extracts line_range for the lemma block
4. Parses section number from surrounding headers
5. Updates source_location in JSON files
"""

import json
from pathlib import Path
import re


def find_section_for_line(markdown_lines: list[str], target_line: int) -> str | None:
    """
    Find the section number (X.Y.Z format) for a given line by searching backwards for headers.

    Args:
        markdown_lines: All lines of the markdown file
        target_line: Line number (1-indexed) to find section for

    Returns:
        Section number in X.Y.Z format, or None if not found
    """
    # Search backwards from target line
    for i in range(target_line - 1, -1, -1):
        line = markdown_lines[i]
        # Match patterns like "### 12.2.1." or "#### 12.2.1." etc.
        # Extract just the number part
        match = re.match(r"^#+\s*(\d+(?:\.\d+)*)", line)
        if match:
            return match.group(1).rstrip(".")
    return None


def find_lemma_in_markdown(
    markdown_content: str, lemma_name: str, lemma_label: str
) -> tuple[int, int, str | None] | None:
    """
    Find a lemma in markdown by searching for its label (most reliable).

    Strategy:
    1. Search directly for ":label: {lemma_label}" line
    2. Find the lemma start (heading or directive before label)
    3. Find the lemma end (next heading/directive at same level)

    Args:
        markdown_content: Full markdown file content
        lemma_name: Lemma title (e.g., "Boundedness of the Fitness Potential")
        lemma_label: Lemma label (e.g., "lem-potential-boundedness")

    Returns:
        Tuple of (start_line, end_line, section) or None if not found
    """
    lines = markdown_content.splitlines()

    # STRATEGY 1: Search directly for the label line (most reliable)
    label_line_idx = None
    for i, line in enumerate(lines):
        if f":label: {lemma_label}" in line:
            label_line_idx = i
            break

    if label_line_idx is None:
        return None

    # Found the label! Now find the start of this lemma
    # Work backwards to find the heading or directive that starts this lemma
    start_line_idx = None
    start_level = None

    for i in range(label_line_idx - 1, -1, -1):
        line = lines[i].strip()

        # Check for heading (# Title)
        if line.startswith("#"):
            start_line_idx = i
            start_level = ("heading", len(line) - len(line.lstrip("#")))
            break

        # Check for MyST directive (:::{prf:lemma})
        if line.startswith(":::") and "{prf:" in line:
            start_line_idx = i
            start_level = ("directive", 0)
            break

    if start_line_idx is None:
        # Couldn't find start, use label line as start
        start_line_idx = label_line_idx

    # Find the end of this lemma block
    end_line_idx = len(lines)  # Default to end of file

    if start_level and start_level[0] == "heading":
        # For headings: next heading at same or higher level
        title_level = start_level[1]
        for j in range(start_line_idx + 1, len(lines)):
            next_line = lines[j]
            if next_line.strip().startswith("#"):
                next_level = len(next_line) - len(next_line.lstrip("#"))
                if next_level <= title_level:
                    end_line_idx = j
                    break

    elif start_level and start_level[0] == "directive":
        # For directives: find matching closing :::
        nesting = 0
        for j in range(start_line_idx, len(lines)):
            line = lines[j].strip()
            if line.startswith(":::") and "{prf:" in line:
                nesting += 1
            elif line == ":::":
                nesting -= 1
                if nesting == 0:
                    end_line_idx = j + 1
                    break

    # Get section number
    section = find_section_for_line(lines, start_line_idx + 1)

    return (start_line_idx + 1, end_line_idx, section)


def enrich_lemma_file(lemma_file: Path, markdown_file: Path, markdown_content: str) -> bool:
    """
    Enrich a single lemma JSON file with source location data.

    Args:
        lemma_file: Path to lemma JSON file
        markdown_file: Path to source markdown file
        markdown_content: Content of markdown file

    Returns:
        True if enrichment succeeded, False otherwise
    """
    try:
        # Load lemma data
        with open(lemma_file, encoding="utf-8") as f:
            lemma_data = json.load(f)

        lemma_name = lemma_data.get("name")
        lemma_label = lemma_data.get("label")

        if not lemma_name or not lemma_label:
            print(f"❌ {lemma_file.name}: Missing name or label")
            return False

        # Find lemma in markdown
        result = find_lemma_in_markdown(markdown_content, lemma_name, lemma_label)

        if not result:
            print(f"❌ {lemma_file.name}: Not found in markdown (name: {lemma_name})")
            return False

        start_line, end_line, section = result

        # Update source_location
        source_loc = lemma_data.get("source_location", {})
        if source_loc is None:
            source_loc = {}

        # Preserve existing fields, update what we found
        source_loc["line_range"] = [start_line, end_line]
        source_loc["directive_label"] = lemma_label

        if section:
            source_loc["section"] = section

        lemma_data["source_location"] = source_loc

        # Write back
        with open(lemma_file, "w", encoding="utf-8") as f:
            json.dump(lemma_data, f, indent=2, ensure_ascii=False)

        print(f"✓ {lemma_file.name}: lines {start_line}-{end_line}, section {section or 'N/A'}")
        return True

    except Exception as e:
        print(f"❌ {lemma_file.name}: Error - {e}")
        return False


def main():
    """Enrich all lemma files in the 01_fragile_gas_framework directory."""
    lemmas_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/lemmas")
    markdown_file = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    if not lemmas_dir.exists():
        print(f"❌ Lemmas directory not found: {lemmas_dir}")
        return

    if not markdown_file.exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        return

    # Load markdown content once
    with open(markdown_file, encoding="utf-8") as f:
        markdown_content = f.read()

    # Process all lemma JSON files
    lemma_files = sorted(lemmas_dir.glob("*.json"))

    print(f"Enriching {len(lemma_files)} lemma files...")
    print("=" * 80)

    succeeded = 0
    failed = 0

    for lemma_file in lemma_files:
        if enrich_lemma_file(lemma_file, markdown_file, markdown_content):
            succeeded += 1
        else:
            failed += 1

    print("=" * 80)
    print(f"Results: {succeeded} succeeded, {failed} failed out of {len(lemma_files)} total")

    if failed > 0:
        print(f"\n⚠ {failed} lemmas need manual attention")
    else:
        print(f"\n✓ All {succeeded} lemmas enriched successfully!")


if __name__ == "__main__":
    main()
