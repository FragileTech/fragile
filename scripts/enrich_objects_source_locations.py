#!/usr/bin/env python3
"""
Enrich object entities with complete source locations.

This script:
1. Copies section and line_range from 'source' field to 'source_location' when available
2. For objects without 'source', uses find_source_location.py to find the location
3. Validates all results with validate_all_source_locations.py
"""

import json
from pathlib import Path
import sys
from typing import Any


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import re

from fragile.proofs.tools.line_finder import (
    get_file_line_count,
    validate_line_range,
)


def extract_section_from_line_range(
    markdown_content: str, line_range: tuple[int, int]
) -> str | None:
    """Extract section number (X.Y.Z format) from markdown headers before line_range."""
    lines = markdown_content.split("\n")

    # Search backwards from start of line_range to find nearest section header
    for i in range(line_range[0] - 1, -1, -1):
        if i >= len(lines):
            continue
        line = lines[i].strip()

        # Match section headers like "## 4.2.1 Title" or "### 4.2.1.1 Title"
        if line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                # Check if second part looks like section number
                section_candidate = parts[1]
                # Section format: digits separated by dots
                if all(c.isdigit() or c == "." for c in section_candidate):
                    return section_candidate.rstrip(".")

    return None


def enrich_object_from_source_field(obj_data: dict[str, Any]) -> bool:
    """
    Copy section and line_range from 'source' to 'source_location' if available.
    Returns True if enrichment was successful (has both section AND line_range).
    """
    source = obj_data.get("source")
    if not source:
        return False

    source_location = obj_data.get("source_location", {})

    # Check if source has BOTH section and line_range
    has_section = "section" in source and source["section"]
    has_line_range = "line_range" in source and source["line_range"]

    if has_section and has_line_range:
        source_location["section"] = str(source["section"])
        source_location["line_range"] = source["line_range"]
        obj_data["source_location"] = source_location
        return True

    # Partial data from source (only section) - return False to try other methods
    return False


def find_directive_by_label(markdown_content: str, label: str) -> tuple[int, int] | None:
    """
    Find a Jupyter Book directive by its :label: field.
    Handles MyST format where label is on separate line.

    Example:
        :::{prf:definition} Title
        :label: def-my-label
        Content
        :::
    """
    lines = markdown_content.split("\n")

    # Search for :label: line
    for i, line in enumerate(lines):
        if f":label: {label}" in line:
            # Found label line, search backwards for directive opening
            for j in range(i - 1, max(0, i - 10), -1):
                if re.match(r"^:::\{prf:\w+\}", lines[j].strip()):
                    # Found directive opening, now find closing
                    start_line = j + 1  # 1-indexed

                    # Search forward for closing :::
                    for k in range(i + 1, min(len(lines), i + 100)):
                        if lines[k].strip() == ":::":
                            end_line = k + 1  # 1-indexed
                            return (start_line, end_line)

                    # No closing found, use reasonable default
                    return (start_line, min(len(lines), i + 20))

    return None


def enrich_object_from_label(
    obj_data: dict[str, Any],
    markdown_content: str,
    markdown_file: Path,
) -> bool:
    """
    Find source location using definition_label directive matching.
    Returns True if enrichment was successful.
    """
    # Get the definition label
    def_label = obj_data.get("definition_label")
    if not def_label:
        label = obj_data.get("label")
        if label and label.startswith("obj-"):
            # Try converting obj-X to def-X
            def_label = label.replace("obj-", "def-", 1)
        else:
            return False

    # Find directive in markdown using custom function
    line_range = find_directive_by_label(markdown_content, def_label)
    if not line_range:
        return False

    # Validate line range
    max_lines = get_file_line_count(markdown_content)
    if not validate_line_range(line_range, max_lines):
        return False

    # Extract section from line range
    section = extract_section_from_line_range(markdown_content, line_range)

    # Update source_location
    source_location = obj_data.get("source_location", {})
    source_location["line_range"] = list(line_range)
    if section:
        source_location["section"] = section
    source_location["directive_label"] = def_label
    source_location["url_fragment"] = f"#{def_label}"

    obj_data["source_location"] = source_location

    return True


def enrich_object_from_name_search(
    obj_data: dict[str, Any],
    markdown_content: str,
) -> bool:
    """
    Find source location by searching for the object name in markdown.
    This is a fallback for objects without proper definition labels.
    """
    name = obj_data.get("name")
    if not name:
        return False

    # Search for name in markdown (case-insensitive)
    lines = markdown_content.split("\n")
    name_lower = name.lower()

    for i, line in enumerate(lines):
        if name_lower in line.lower():
            # Found name, check if it's near a definition directive
            # Search backwards for directive
            for j in range(max(0, i - 5), i + 1):
                if re.match(r"^:::\{prf:definition\}", lines[j].strip()):
                    # Found definition, find its label and closing
                    label = None
                    for k in range(j + 1, min(len(lines), j + 10)):
                        if ":label:" in lines[k]:
                            label_match = re.search(r":label:\s+(\S+)", lines[k])
                            if label_match:
                                label = label_match.group(1)
                                break

                    # Find closing
                    start_line = j + 1  # 1-indexed
                    end_line = start_line + 20  # default
                    for k in range(i + 1, min(len(lines), i + 100)):
                        if lines[k].strip() == ":::":
                            end_line = k + 1  # 1-indexed
                            break

                    # Extract section
                    section = extract_section_from_line_range(
                        markdown_content, (start_line, end_line)
                    )

                    # Update source_location
                    source_location = obj_data.get("source_location", {})
                    source_location["line_range"] = [start_line, end_line]
                    if section:
                        source_location["section"] = section
                    if label:
                        source_location["directive_label"] = label
                        source_location["url_fragment"] = f"#{label}"

                    obj_data["source_location"] = source_location
                    return True

    return False


def enrich_objects_directory(
    objects_dir: Path,
    markdown_file: Path,
) -> tuple[int, int, int]:
    """
    Enrich all object JSON files in directory.
    Returns (success_count, failed_count, total_count).
    """
    if not objects_dir.exists():
        print(f"ERROR: Directory does not exist: {objects_dir}")
        return (0, 0, 0)

    if not markdown_file.exists():
        print(f"ERROR: Markdown file does not exist: {markdown_file}")
        return (0, 0, 0)

    # Read markdown content once
    with open(markdown_file, encoding="utf-8") as f:
        markdown_content = f.read()

    # Process all JSON files
    json_files = sorted(objects_dir.glob("*.json"))
    total = len(json_files)
    success = 0
    failed = 0

    print(f"\nProcessing {total} object files in {objects_dir.name}/")
    print("=" * 70)

    for json_file in json_files:
        try:
            # Read object data
            with open(json_file, encoding="utf-8") as f:
                obj_data = json.load(f)

            enriched = False
            method = ""

            # Try method 1: Copy from 'source' field
            if enrich_object_from_source_field(obj_data):
                enriched = True
                method = "source_field"

            # Try method 2: Find using definition_label
            elif enrich_object_from_label(obj_data, markdown_content, markdown_file):
                enriched = True
                method = "directive_label"

            # Try method 3: Search by name
            elif enrich_object_from_name_search(obj_data, markdown_content):
                enriched = True
                method = "name_search"

            if enriched:
                # Write updated data
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(obj_data, f, indent=2, ensure_ascii=False)
                    f.write("\n")

                success += 1
                sl = obj_data["source_location"]
                section = sl.get("section", "?")
                lr = sl.get("line_range", [0, 0])
                print(f"✓ {json_file.name:50} [{method:15}] §{section} L{lr[0]}-{lr[1]}")
            else:
                failed += 1
                print(f"✗ {json_file.name:50} [no_match] Could not find location")

        except Exception as e:
            failed += 1
            print(f"✗ {json_file.name:50} [error] {e}")

    print("=" * 70)
    print(f"\nResults: {success} succeeded, {failed} failed, {total} total")
    print(f"Success rate: {100 * success / total:.1f}%")

    return (success, failed, total)


def main():
    """Main entry point."""
    # Define paths
    repo_root = Path(__file__).parent.parent
    objects_dir = (
        repo_root / "docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/objects"
    )
    markdown_file = repo_root / "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"

    print("Object Source Location Enrichment")
    print("=" * 70)
    print(f"Objects directory: {objects_dir}")
    print(f"Markdown file: {markdown_file}")

    # Enrich objects
    _success, failed, _total = enrich_objects_directory(objects_dir, markdown_file)

    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)

    print("\n✓ All objects enriched successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
