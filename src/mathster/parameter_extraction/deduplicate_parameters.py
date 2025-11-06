#!/usr/bin/env python3
"""
Parameter deduplication script.

Removes duplicate parameter entries from the unified registry that have the same label.
Uses intelligent scoring to select the best version when duplicates exist.

Usage:
    python src/mathster/parameter_extraction/deduplicate_parameters.py
    python src/mathster/parameter_extraction/deduplicate_parameters.py --dry-run
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def score_parameter(param: dict) -> float:
    """Score a parameter to determine which duplicate to keep."""
    score = 0.0

    # Priority 1: Correct attribution (+1000)
    doc_id = param.get("_document_id", "")
    source_article_id = param.get("source", {}).get("article_id", "")
    if doc_id == source_article_id:
        score += 1000

    # Priority 2: File path match (+100)
    file_path = param.get("source", {}).get("file_path", "")
    if doc_id in file_path:
        score += 100

    # Priority 3: Valid line number (+10)
    line_range = param.get("source", {}).get("line_range", {}).get("lines", [[1, 1]])
    if line_range and len(line_range) > 0:
        line_start = line_range[0][0]
        if line_start != 1:
            score += 10
        score -= line_start / 10000.0

    # Priority 4: Has confidence (+1)
    if param.get("_dspy_confidence"):
        score += 1

    return score


def deduplicate(parameters: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicate parameters by label."""
    grouped = {}
    for param in parameters:
        label = param.get("label", "")
        if label:
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(param)

    # Find duplicates
    duplicates = {label: entries for label, entries in grouped.items() if len(entries) > 1}
    
    logger.info(f"Found {len(duplicates)} labels with duplicates")

    # Select best version
    deduplicated = []
    
    # Add unique parameters
    for param in parameters:
        label = param.get("label", "")
        if label not in duplicates:
            deduplicated.append(param)
    
    # For duplicates, select best
    total_removed = 0
    for label, entries in duplicates.items():
        scored = [(score_parameter(e), e) for e in entries]
        scored.sort(reverse=True)
        best = scored[0][1]
        deduplicated.append(best)
        total_removed += len(entries) - 1
        logger.info(f"  {label}: {len(entries)} copies → kept 1 (score: {scored[0][0]:.1f})")

    report = {
        "original": len(parameters),
        "deduplicated": len(deduplicated),
        "removed": total_removed,
    }

    return deduplicated, report


def main():
    parser = argparse.ArgumentParser(description="Deduplicate parameters")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--registry-root", type=Path, default=Path("unified_registry"))
    args = parser.parse_args()

    params_file = args.registry_root / "parameters.json"
    
    if not params_file.exists():
        logger.error(f"Not found: {params_file}")
        return 1

    with open(params_file) as f:
        parameters = json.load(f)

    deduplicated, report = deduplicate(parameters)

    if not args.dry_run:
        with open(params_file, "w") as f:
            json.dump(deduplicated, f, indent=2)
        logger.info(f"\n✓ Saved to {params_file}")

    print(f"\nOriginal: {report['original']}")
    print(f"Deduplicated: {report['deduplicated']}")
    print(f"Removed: {report['removed']}")

    return 0


if __name__ == "__main__":
    exit(main())
