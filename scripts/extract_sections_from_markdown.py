#!/usr/bin/env python3
"""
Extract Section Numbers from Markdown Headers.

Parses markdown file headers to build a mapping of line ranges to section numbers,
then enriches entity JSON files by filling in missing section fields based on their line_range.

Usage:
    python scripts/extract_sections_from_markdown.py \
        --markdown docs/source/1_euclidean_gas/01_fragile_gas_framework.md \
        --raw-data docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data
"""

import argparse
import json
from pathlib import Path
import re
import sys


def parse_section_from_header(header_text: str) -> str | None:
    """
    Extract section number from header text.

    Examples:
        "# 14. The Perturbation Operator" → "14"
        "## 14.2 Subsection" → "14.2"
        "### 2.1.3 Deep Subsection" → "2.1.3"
        "# Introduction" → None (no number)
    """
    # Remove leading # symbols and whitespace
    text = header_text.lstrip("#").strip()

    # Try to extract leading number (with optional dots for subsections)
    match = re.match(r"^(\d+(?:\.\d+)*)", text)
    if match:
        return match.group(1)

    return None


def build_section_map(markdown_path: Path) -> list[tuple[int, int, str]]:
    """
    Build a map of (start_line, end_line, section_number) from markdown headers.

    Args:
        markdown_path: Path to markdown file

    Returns:
        List of (start_line, end_line, section_number) tuples sorted by start_line
    """
    content = markdown_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    sections = []
    current_section = None
    current_start = 1

    for i, line in enumerate(lines, start=1):
        # Check if this is a header line
        if line.startswith("#"):
            section_num = parse_section_from_header(line)

            if section_num:
                # Close previous section
                if current_section:
                    sections.append((current_start, i - 1, current_section))

                # Start new section
                current_section = section_num
                current_start = i

    # Close final section
    if current_section:
        sections.append((current_start, len(lines), current_section))

    return sections


def find_section_for_line_range(
    line_range: tuple[int, int], section_map: list[tuple[int, int, str]]
) -> str | None:
    """
    Find which section a line range falls within.

    Args:
        line_range: (start, end) tuple
        section_map: List of (start, end, section) tuples

    Returns:
        Section number or None
    """
    start, _end = line_range

    # Find section containing the start of the entity
    for sec_start, sec_end, section in section_map:
        if sec_start <= start <= sec_end:
            return section

    return None


def enrich_entity_with_section(
    entity_path: Path, section_map: list[tuple[int, int, str]], dry_run: bool = False
) -> tuple[bool, str | None, str | None]:
    """
    Enrich a single entity with section number based on its line_range.

    Args:
        entity_path: Path to entity JSON
        section_map: Section mapping from markdown
        dry_run: If True, don't write changes

    Returns:
        (success, old_section, new_section)
    """
    try:
        data = json.loads(entity_path.read_text(encoding="utf-8"))

        # Check if has source_location with line_range
        if "source_location" not in data or not data["source_location"]:
            return (False, None, None)

        loc = data["source_location"]

        # Check if already has section
        old_section = loc.get("section")
        if old_section:
            return (True, old_section, old_section)  # Already has section

        # Check if has line_range
        line_range = loc.get("line_range")
        if not line_range or len(line_range) != 2:
            return (False, None, None)  # No line_range to work with

        # Find section for this line range
        new_section = find_section_for_line_range(tuple(line_range), section_map)

        if new_section:
            if not dry_run:
                loc["section"] = new_section
                entity_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            return (True, None, new_section)

        return (False, None, None)

    except Exception as e:
        print(f"Error processing {entity_path}: {e}")
        return (False, None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Extract section numbers from markdown and enrich entity JSON files"
    )

    parser.add_argument("--markdown", "-m", type=Path, required=True, help="Path to markdown file")
    parser.add_argument(
        "--raw-data", "-r", type=Path, required=True, help="Path to raw_data directory"
    )
    parser.add_argument(
        "--dry-run", "-d", action="store_true", help="Preview changes without writing"
    )

    args = parser.parse_args()

    if not args.markdown.exists():
        print(f"Error: Markdown file not found: {args.markdown}")
        sys.exit(1)

    if not args.raw_data.exists():
        print(f"Error: Raw data directory not found: {args.raw_data}")
        sys.exit(1)

    # Build section map from markdown
    print(f"Parsing markdown headers from {args.markdown}...")
    section_map = build_section_map(args.markdown)

    print(f"\nFound {len(section_map)} sections:")
    for start, end, section in section_map[:20]:  # Show first 20
        print(f"  Section {section:5} (lines {start:5}-{end:5})")
    if len(section_map) > 20:
        print(f"  ... and {len(section_map) - 20} more")

    # Enrich all entity files
    print(f"\nEnriching entities in {args.raw_data}...")

    stats = {"total": 0, "enriched": 0, "already_had_section": 0, "no_line_range": 0, "failed": 0}

    changes = []

    for json_file in args.raw_data.rglob("*.json"):
        # Skip report files
        if "report" in json_file.name.lower():
            continue

        stats["total"] += 1
        success, old_section, new_section = enrich_entity_with_section(
            json_file, section_map, args.dry_run
        )

        if not success:
            if old_section is None and new_section is None:
                stats["no_line_range"] += 1
            else:
                stats["failed"] += 1
        elif old_section == new_section and old_section is not None:
            stats["already_had_section"] += 1
        else:
            stats["enriched"] += 1
            changes.append((json_file, new_section))

    # Print report
    print(f"\n{'=' * 80}")
    print("SECTION ENRICHMENT REPORT")
    print(f"{'=' * 80}")
    print(f"Total files processed:     {stats['total']}")
    print(f"  ✓ Enriched with section: {stats['enriched']}")
    print(f"  ✓ Already had section:   {stats['already_had_section']}")
    print(f"  - No line_range:         {stats['no_line_range']}")
    print(f"  ✗ Failed:                {stats['failed']}")

    if changes:
        print(f"\n{'─' * 80}")
        print("CHANGES MADE:")
        print(f"{'─' * 80}")
        for file_path, section in changes[:20]:
            print(f"✓ {file_path.name:50} → Section {section}")
        if len(changes) > 20:
            print(f"\n... and {len(changes) - 20} more files enriched")

    if args.dry_run:
        print(f"\n{'=' * 80}")
        print("DRY RUN: No files were modified")
        print(f"{'=' * 80}")

    sys.exit(0 if stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
