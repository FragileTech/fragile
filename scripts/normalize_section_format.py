#!/usr/bin/env python3
"""
Section Format Normalization Tool.

Normalizes section fields to strict X.Y.Z numeric format.
Rules:
- Remove § symbol
- Remove "Section " prefix (case-insensitive)
- Remove titles after dot or colon
- Keep only numeric dotted notation (e.g., "3", "2.1", "2.1.3")
- Return None if cannot normalize to valid format

Examples:
    "§3" → "3"
    "§2.1.3" → "2.1.3"
    "9. Rescale Transformation" → "9"
    "Section 12: Title" → "12"
    "3.1" → "3.1" (already valid)
    "Invalid text" → None

Usage:
    python scripts/normalize_section_format.py --section "§3.2"
    python scripts/normalize_section_format.py --file entity.json
    python scripts/normalize_section_format.py --batch raw_data/
"""

import argparse
import json
from pathlib import Path
import re
import sys


def normalize_section(section: str | None) -> str | None:
    """
    Normalize section to strict X.Y.Z numeric format.

    Args:
        section: Section string to normalize

    Returns:
        Normalized section in X.Y.Z format, or None if invalid

    Examples:
        >>> normalize_section("§3")
        '3'
        >>> normalize_section("9. Rescale Transformation")
        '9'
        >>> normalize_section("Section 12: Title")
        '12'
        >>> normalize_section("2.1.3")
        '2.1.3'
        >>> normalize_section("Invalid text")
        None
    """
    if section is None or not isinstance(section, str):
        return None

    section = section.strip()

    if not section:
        return None

    # Remove § symbol (Unicode section sign)
    if section.startswith("§"):
        section = section[1:].strip()

    # Remove "Section " prefix (case-insensitive)
    if section.lower().startswith("section "):
        section = section[8:].strip()

    # Extract leading number before space (e.g., "17 The Revival State" → "17")
    match = re.match(r"^(\d+(?:\.\d+)*)\s+", section)
    if match:
        section = match.group(1)

    # Remove title after ". " (e.g., "9. Title" → "9")
    if ". " in section:
        parts = section.split(". ", 1)
        # Check if first part is numeric
        if parts[0].replace(".", "").isdigit():
            section = parts[0]

    # Remove title after ": " (e.g., "12: Title" → "12")
    if ": " in section:
        parts = section.split(": ", 1)
        # Check if first part is numeric
        if parts[0].replace(".", "").isdigit():
            section = parts[0]

    # Handle formats with comma (e.g., "§9 Rescale Transformation, §8.2.2.4 Lemma: ...")
    # Take first section number only
    if "," in section:
        parts = section.split(",", 1)
        # Try to extract number from first part
        match = re.match(r"^(\d+(?:\.\d+)*)", parts[0].strip())
        if match:
            section = match.group(1)

    # Handle formats like "§1-introduction" → "1"
    if "-" in section and not section.replace("-", "").replace(".", "").isdigit():
        parts = section.split("-", 1)
        if parts[0].replace(".", "").isdigit():
            section = parts[0]

    # Final validation: must match X.Y.Z format (only digits and dots)
    if not re.match(r"^\d+(\.\d+)*$", section):
        return None  # Invalid format

    return section


def normalize_entity_file(
    file_path: Path, dry_run: bool = False
) -> tuple[bool, str | None, str | None]:
    """
    Normalize section in a single entity JSON file.

    Args:
        file_path: Path to JSON file
        dry_run: If True, don't write changes

    Returns:
        Tuple of (success, old_section, new_section)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            entity = json.load(f)

        # Check multiple possible locations for section info
        old_section = None
        location = None

        # Check source_location.section
        if entity.get("source_location"):
            if "section" in entity["source_location"]:
                old_section = entity["source_location"]["section"]
                location = "source_location.section"

        # Check source_section (top-level field in raw entities)
        if not old_section and "source_section" in entity:
            old_section = entity["source_section"]
            location = "source_section"

        # Check source.section (older format)
        if not old_section and "source" in entity and entity["source"]:
            if "section" in entity["source"]:
                old_section = entity["source"]["section"]
                location = "source.section"

        if old_section is None:
            return (False, None, None)  # No section to normalize

        new_section = normalize_section(old_section)

        if new_section == old_section:
            return (True, old_section, old_section)  # Already normalized

        if new_section is None:
            return (False, old_section, None)  # Cannot normalize

        # Update the entity
        if not dry_run:
            if location == "source_location.section":
                entity["source_location"]["section"] = new_section
            elif location == "source_section":
                entity["source_section"] = new_section
            elif location == "source.section":
                entity["source"]["section"] = new_section

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entity, f, indent=2, ensure_ascii=False)

        return (True, old_section, new_section)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (False, None, None)


def batch_normalize(directory: Path, dry_run: bool = False):
    """
    Normalize all entity JSON files in a directory recursively.

    Args:
        directory: Directory to scan
        dry_run: If True, don't write changes
    """
    stats = {
        "total": 0,
        "normalized": 0,
        "already_valid": 0,
        "invalid": 0,
        "no_section": 0,
        "errors": 0,
    }

    changes = []

    for json_file in directory.rglob("*.json"):
        stats["total"] += 1
        success, old_section, new_section = normalize_entity_file(json_file, dry_run=dry_run)

        if not success:
            if old_section is None:
                stats["no_section"] += 1
            elif new_section is None:
                stats["invalid"] += 1
                changes.append((json_file, old_section, "INVALID"))
            else:
                stats["errors"] += 1
        elif old_section == new_section:
            stats["already_valid"] += 1
        else:
            stats["normalized"] += 1
            changes.append((json_file, old_section, new_section))

    # Print report
    print(f"\n{'=' * 70}")
    print("SECTION NORMALIZATION REPORT")
    print(f"{'=' * 70}")
    print(f"Total files processed: {stats['total']}")
    print(f"  ✓ Already valid:     {stats['already_valid']}")
    print(f"  ✓ Normalized:        {stats['normalized']}")
    print(f"  ✗ Invalid format:    {stats['invalid']}")
    print(f"  - No section:        {stats['no_section']}")
    print(f"  ✗ Errors:            {stats['errors']}")

    if changes:
        print(f"\n{'─' * 70}")
        print("CHANGES MADE:")
        print(f"{'─' * 70}")
        for file_path, old, new in changes[:20]:  # Show first 20
            status = "✗ INVALID" if new == "INVALID" else "✓"
            print(f"{status} {file_path.name}")
            print(f"     Old: '{old}'")
            if new != "INVALID":
                print(f"     New: '{new}'")
            print()

        if len(changes) > 20:
            print(f"... and {len(changes) - 20} more changes")

    if dry_run:
        print(f"\n{'=' * 70}")
        print("DRY RUN: No files were modified")
        print(f"{'=' * 70}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Normalize section format to strict X.Y.Z numeric notation"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--section", "-s", help="Normalize a single section string")
    group.add_argument("--file", "-f", type=Path, help="Normalize section in a JSON file")
    group.add_argument(
        "--batch", "-b", type=Path, help="Batch normalize all JSON files in directory"
    )

    parser.add_argument(
        "--dry-run", "-d", action="store_true", help="Don't write changes (preview only)"
    )

    args = parser.parse_args()

    if args.section:
        # Single section normalization
        normalized = normalize_section(args.section)
        print(f"Input:  '{args.section}'")
        print(f"Output: '{normalized}'")
        if normalized is None:
            print("Status: INVALID (cannot normalize)")
            sys.exit(1)
        elif normalized == args.section:
            print("Status: Already valid")
        else:
            print("Status: Normalized")

    elif args.file:
        # Single file normalization
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        success, old_section, new_section = normalize_entity_file(args.file, dry_run=args.dry_run)

        if not success:
            if old_section is None:
                print(f"No section field found in {args.file}")
            else:
                print(f"Cannot normalize section '{old_section}' in {args.file}")
            sys.exit(1)

        print(f"File: {args.file}")
        print(f"Old:  '{old_section}'")
        print(f"New:  '{new_section}'")

        if old_section == new_section:
            print("Status: Already valid")
        else:
            print(f"Status: {'Normalized (preview)' if args.dry_run else 'Normalized'}")

    elif args.batch:
        # Batch normalization
        if not args.batch.exists():
            print(f"Error: Directory not found: {args.batch}")
            sys.exit(1)

        stats = batch_normalize(args.batch, dry_run=args.dry_run)

        # Exit with error if any invalid sections found
        if stats["invalid"] > 0 or stats["errors"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
