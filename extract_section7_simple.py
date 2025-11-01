#!/usr/bin/env python3
"""
Extract Section 7 (Swarm Measuring) from 01_fragile_gas_framework.md

This script extracts mathematical entities from lines 1242-1480 by manually
parsing the directive structure without requiring LLM calls.
"""

from datetime import datetime
import json
from pathlib import Path

from fragile.proofs.tools import extract_jupyter_directives


def extract_directive_content(
    markdown_text: str, start_marker: str, label: str
) -> tuple[str, int, int]:
    """Extract content between directive markers."""
    lines = markdown_text.split("\n")

    # Find start line
    start_idx = None
    for i, line in enumerate(lines):
        if start_marker in line and label in lines[i : i + 3]:  # Check label within next 3 lines
            start_idx = i
            break

    if start_idx is None:
        return "", 0, 0

    # Find end marker (:::)
    end_idx = None
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip() == ":::":
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(lines)

    content = "\n".join(lines[start_idx : end_idx + 1])
    return content, start_idx + 1, end_idx + 1


def extract_section_7_simple():
    """Extract Section 7 entities using simple parsing."""

    # Paths
    source_file = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
    )
    output_base = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework"
    )

    print(f"Reading section from {source_file}")

    # Read the specific section (lines 1242-1480)
    with open(source_file, encoding="utf-8") as f:
        lines = f.readlines()

    section_lines = lines[1241:1480]  # 0-indexed, so subtract 1
    section_content = "".join(section_lines)

    print(f"Extracted {len(section_lines)} lines from Section 7")
    print(f"Section length: {len(section_content)} characters")

    # Extract directive hints
    directives = extract_jupyter_directives(section_content, section_id="§7")
    print(f"\nFound {len(directives)} directive hints:")

    # Create output directories
    output_dirs = {
        "definitions": output_base / "definitions",
        "theorems": output_base / "theorems",
        "lemmas": output_base / "lemmas",
        "mathster": output_base / "mathster",
        "remarks": output_base / "remarks",
    }

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_file": str(source_file),
        "section": "Section 7 (Swarm Measuring)",
        "lines": "1242-1480",
        "processing_stage": "manual_extraction",
        "entities_extracted": {},
        "extraction_time": datetime.now().isoformat(),
        "directives": [],
    }

    # Process each directive
    for idx, directive in enumerate(directives):
        print(f"\n{idx + 1}. {directive.directive_type}: {directive.label}")
        print(f"   Lines {directive.start_line}-{directive.end_line}")
        print(f"   Content preview: {directive.content[:100]}...")

        entity_data = {
            "label": directive.label,
            "type": directive.directive_type,
            "content": directive.content,
            "start_line": directive.start_line + 1241,  # Adjust to document line numbers
            "end_line": directive.end_line + 1241,
            "section": "§7",
        }

        stats["directives"].append({
            "label": directive.label,
            "type": directive.directive_type,
            "lines": f"{entity_data['start_line']}-{entity_data['end_line']}",
        })

        # Save to appropriate directory
        if directive.directive_type == "definition":
            filename = f"{directive.label}.json"
            file_path = output_dirs["definitions"] / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entity_data, f, indent=2)
            stats["entities_extracted"]["definitions"] = (
                stats["entities_extracted"].get("definitions", 0) + 1
            )

        elif directive.directive_type in {"theorem", "proposition", "corollary"}:
            filename = f"{directive.label}.json"
            file_path = output_dirs["theorems"] / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entity_data, f, indent=2)
            stats["entities_extracted"]["theorems"] = (
                stats["entities_extracted"].get("theorems", 0) + 1
            )

        elif directive.directive_type == "lemma":
            filename = f"{directive.label}.json"
            file_path = output_dirs["lemmas"] / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entity_data, f, indent=2)
            stats["entities_extracted"]["lemmas"] = (
                stats["entities_extracted"].get("lemmas", 0) + 1
            )

        elif directive.directive_type == "proof":
            filename = f"{directive.label}.json"
            file_path = output_dirs["mathster"] / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entity_data, f, indent=2)
            stats["entities_extracted"]["mathster"] = (
                stats["entities_extracted"].get("mathster", 0) + 1
            )

        elif directive.directive_type == "remark":
            filename = f"{directive.label}.json"
            file_path = output_dirs["remarks"] / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entity_data, f, indent=2)
            stats["entities_extracted"]["remarks"] = (
                stats["entities_extracted"].get("remarks", 0) + 1
            )

    # Calculate total
    stats["total_entities"] = sum(stats["entities_extracted"].values())

    # Save statistics
    stats_file = output_base / "section7_extraction_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total entities extracted: {stats['total_entities']}")
    for entity_type, count in stats["entities_extracted"].items():
        print(f"  - {entity_type}: {count}")
    print(f"\nOutput directory: {output_base}")
    print(f"Statistics saved to: {stats_file}")
    print(f"{'=' * 60}\n")

    return stats


if __name__ == "__main__":
    try:
        stats = extract_section_7_simple()
        print("\n✓ Extraction completed successfully!")
        print(f"✓ Total entities: {stats['total_entities']}")
        print("\nEntities by type:")
        for entity_type, count in stats["entities_extracted"].items():
            print(f"  - {entity_type}: {count} files")
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback

        traceback.print_exc()
        raise
