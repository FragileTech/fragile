#!/usr/bin/env python3
"""Fix SourceLocation format in refined_data entities.

Converts old format:
  {"file": "...", "line_start": ..., "line_end": ..., "section": "..."}

To new format:
  {"document_id": "...", "file_path": "...", "section": "...",
   "directive_label": "...", "line_range": [...]}
"""

import json
from pathlib import Path
import sys
from typing import Any


def convert_source_location(
    source: dict[str, Any] | None, entity_label: str, chapter: str, document: str
) -> dict[str, Any] | None:
    """Convert old SourceLocation format to new format.

    Args:
        source: Old source dict or None
        entity_label: Entity label (e.g., "obj-euclidean-gas")
        chapter: Chapter (e.g., "1_euclidean_gas")
        document: Document (e.g., "01_fragile_gas_framework")

    Returns:
        New source dict or None
    """
    if source is None:
        # Create minimal valid source with available info
        return {
            "document_id": document,
            "file_path": f"docs/source/{chapter}/{document}.md",
            "section": None,
            "directive_label": None,
            "line_range": None,
        }

    # Always use chapter/document metadata as source of truth for document_id
    # (don't trust potentially corrupted "file" field or existing "document_id")
    new_source = {}
    new_source["document_id"] = document
    new_source["file_path"] = f"docs/source/{chapter}/{document}.md"

    # Convert section
    if "section" in source:
        section = source["section"]
        if section is None:
            new_source["section"] = None
        elif isinstance(section, str):
            # Clean up section format if needed
            # e.g., "Section 16: ..., subsection 15.2.1" → "15.2.1"
            if "subsection" in section:
                # Extract subsection number
                parts = section.split("subsection")
                if len(parts) > 1:
                    subsection = parts[1].strip().rstrip(".")
                    new_source["section"] = subsection
                else:
                    new_source["section"] = section
            else:
                new_source["section"] = section
        else:
            new_source["section"] = section
    else:
        new_source["section"] = None

    # Directive label: preserve if exists, otherwise infer from entity_label
    if source.get("directive_label"):
        new_source["directive_label"] = source["directive_label"]
    else:
        # Infer from entity_label: obj-X → def-X, axiom-X → axiom-X, thm-X → thm-X
        if entity_label.startswith("obj-"):
            directive_label = entity_label.replace("obj-", "def-", 1)
        else:
            directive_label = entity_label
        new_source["directive_label"] = directive_label

    # Line range: preserve if exists in new format, otherwise convert from old format
    if source.get("line_range"):
        new_source["line_range"] = source["line_range"]
    elif "line_start" in source and "line_end" in source:
        line_start = source["line_start"]
        line_end = source["line_end"]
        new_source["line_range"] = [line_start, line_end]
    else:
        new_source["line_range"] = None

    return new_source


def fix_entity_file(file_path: Path) -> tuple[bool, str]:
    """Fix SourceLocation in a single entity file.

    Args:
        file_path: Path to entity JSON file

    Returns:
        (success, message)
    """
    try:
        # Read entity
        with open(file_path, encoding="utf-8") as f:
            entity = json.load(f)

        # Extract metadata
        label = entity.get("label", "")
        chapter = entity.get("chapter", "")
        document = entity.get("document", "")

        if not label or not chapter or not document:
            return (
                False,
                f"Missing required fields: label={label}, chapter={chapter}, document={document}",
            )

        # Get current source
        old_source = entity.get("source")

        # Convert source
        new_source = convert_source_location(old_source, label, chapter, document)

        # Update entity
        entity["source"] = new_source

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entity, f, indent=2, ensure_ascii=False)

        return True, "Fixed"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main execution."""
    refined_data_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data")

    if not refined_data_dir.exists():
        print(f"❌ Directory not found: {refined_data_dir}")
        sys.exit(1)

    print("=" * 70)
    print("FIXING SOURCE LOCATIONS IN REFINED DATA")
    print("=" * 70)
    print()

    # Process objects
    objects_dir = refined_data_dir / "objects"
    axioms_dir = refined_data_dir / "axioms"

    total_processed = 0
    total_success = 0
    total_failed = 0

    for entity_dir, entity_type in [(objects_dir, "Objects"), (axioms_dir, "Axioms")]:
        if not entity_dir.exists():
            print(f"⚠️  {entity_type} directory not found: {entity_dir}")
            continue

        print(f"\n{entity_type}:")
        print("-" * 70)

        files = sorted(entity_dir.glob("*.json"))

        for file_path in files:
            total_processed += 1
            success, message = fix_entity_file(file_path)

            if success:
                total_success += 1
                print(f"  ✓ {file_path.name}")
            else:
                total_failed += 1
                print(f"  ✗ {file_path.name}: {message}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed: {total_processed}")
    print(f"Success: {total_success}")
    print(f"Failed: {total_failed}")
    print()

    if total_failed > 0:
        print("⚠️  Some files failed to process")
        sys.exit(1)
    else:
        print("✅ All files processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
