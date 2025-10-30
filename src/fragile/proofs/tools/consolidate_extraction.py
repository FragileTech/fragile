"""Consolidate extraction results from document-parser agents.

This tool aggregates statistics and generates summary reports from multiple
section-level extractions performed by document-parser agents.
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List


def collect_section_statistics(base_dir: Path) -> list[dict[str, Any]]:
    """Collect all section extraction statistics from a directory.

    Args:
        base_dir: Base directory containing extraction outputs

    Returns:
        List of section statistics dictionaries
    """
    stats_dir = base_dir / "statistics"
    if not stats_dir.exists():
        return []

    section_stats = []
    for stats_file in sorted(stats_dir.glob("section*_*.json")):
        try:
            with open(stats_file) as f:
                stats = json.load(f)
                stats["source_file"] = stats_file.name
                section_stats.append(stats)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read {stats_file}: {e}")

    return section_stats


def count_entity_files(base_dir: Path) -> dict[str, int]:
    """Count JSON entity files by type.

    Args:
        base_dir: Base directory containing entity subdirectories

    Returns:
        Dictionary mapping entity type to file count
    """
    entity_types = [
        "axioms",
        "definitions",
        "theorems",
        "lemmas",
        "propositions",
        "proofs",
        "objects",
        "parameters",
        "remarks",
        "citations",
        "equations",
    ]

    counts = {}
    for entity_type in entity_types:
        entity_dir = base_dir / entity_type
        if entity_dir.exists():
            json_files = list(entity_dir.glob("*.json"))
            if json_files:
                counts[entity_type] = len(json_files)

    return counts


def generate_summary_report(base_dir: Path, output_file: Path | None = None) -> dict[str, Any]:
    """Generate consolidated summary report.

    Args:
        base_dir: Base directory containing all extraction outputs
        output_file: Optional path to save JSON report

    Returns:
        Summary report dictionary
    """
    section_stats = collect_section_statistics(base_dir)
    entity_counts = count_entity_files(base_dir)

    # Aggregate by section
    sections_summary = []
    total_entities = 0

    for stats in section_stats:
        section_num = stats.get("section", "unknown")
        section_title = stats.get("section_title", "")
        entity_count = stats.get("total_entities", 0)

        total_entities += entity_count

        sections_summary.append({
            "section": section_num,
            "title": section_title,
            "entities": entity_count,
            "source_file": stats.get("source_file", ""),
        })

    # Build summary
    summary = {
        "document": str(base_dir.name),
        "total_sections_processed": len(section_stats),
        "total_entities_extracted": total_entities,
        "entity_counts_by_type": entity_counts,
        "sections": sections_summary,
    }

    # Save if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary report saved to: {output_file}")

    return summary


def print_summary_table(summary: dict[str, Any]) -> None:
    """Print human-readable summary table.

    Args:
        summary: Summary report from generate_summary_report()
    """
    print(f"\n{'=' * 80}")
    print(f"EXTRACTION SUMMARY: {summary['document']}")
    print(f"{'=' * 80}\n")

    print(f"Total Sections Processed: {summary['total_sections_processed']}")
    print(f"Total Entities Extracted: {summary['total_entities_extracted']}\n")

    print("Entity Counts by Type:")
    print("-" * 40)
    for entity_type, count in sorted(summary["entity_counts_by_type"].items()):
        print(f"  {entity_type:20s}: {count:4d}")

    print(f"\n{'=' * 80}")
    print("Section Breakdown:")
    print(f"{'=' * 80}")
    print(f"{'Section':<10} {'Title':<40} {'Entities':>10}")
    print("-" * 80)

    for section in summary["sections"]:
        section_id = section["section"]
        title = section["title"][:38]  # Truncate long titles
        entities = section["entities"]
        print(f"{section_id:<10} {title:<40} {entities:>10}")

    print(f"{'=' * 80}\n")


def main():
    """Main entry point for consolidation script."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python consolidate_extraction.py <extraction_dir> [output.json]")
        print("\nExample:")
        print(
            "  python consolidate_extraction.py docs/source/1_euclidean_gas/01_fragile_gas_framework/"
        )
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    summary = generate_summary_report(base_dir, output_file)
    print_summary_table(summary)


if __name__ == "__main__":
    main()
