#!/usr/bin/env python3
"""
Integration script for applying improvement metadata to raw_data structure.

This script reads _improvement_metadata from chapter_N.json files and applies
the changes (ADD/MODIFY/DELETE) to the per-entity files in raw_data/ directories.

Usage:
    python -m mathster.tools.integrate_improvements docs/source/1_euclidean_gas/07_mean_field.md
    python -m mathster.tools.integrate_improvements docs/source/1_euclidean_gas/07_mean_field.md --dry-run
    python -m mathster.tools.integrate_improvements docs/source/1_euclidean_gas/07_mean_field.md --validate --verbose
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from mathster.parsing.models.changes import ChangeOperation, EntityChange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class IntegrationStatistics:
    """Track integration statistics."""

    def __init__(self):
        self.entities_added = 0
        self.entities_modified = 0
        self.entities_deleted = 0
        self.errors = []
        self.warnings = []

    def record_add(self, label: str):
        """Record ADD operation."""
        self.entities_added += 1
        logger.info(f"  [ADD] {label}")

    def record_modify(self, label: str):
        """Record MODIFY operation."""
        self.entities_modified += 1
        logger.info(f"  [MODIFY] {label}")

    def record_delete(self, label: str):
        """Record DELETE operation."""
        self.entities_deleted += 1
        logger.info(f"  [DELETE] {label}")

    def record_error(self, message: str):
        """Record error."""
        self.errors.append(message)
        logger.error(f"  ERROR: {message}")

    def record_warning(self, message: str):
        """Record warning."""
        self.warnings.append(message)
        logger.warning(f"  WARNING: {message}")

    def get_summary(self) -> str:
        """Get summary string."""
        lines = [
            "\nIntegration Summary:",
            f"  ✓ Added: {self.entities_added} entities",
            f"  ✓ Modified: {self.entities_modified} entities",
            f"  ✓ Deleted: {self.entities_deleted} entities",
            f"  ✓ Total operations: {self.entities_added + self.entities_modified + self.entities_deleted}",
        ]

        if self.errors:
            lines.append(f"\n  ✗ Errors: {len(self.errors)}")
            for err in self.errors[:5]:  # Show first 5
                lines.append(f"    - {err}")
            if len(self.errors) > 5:
                lines.append(f"    ... and {len(self.errors) - 5} more")

        if self.warnings:
            lines.append(f"\n  ⚠ Warnings: {len(self.warnings)}")
            for warn in self.warnings[:5]:
                lines.append(f"    - {warn}")
            if len(self.warnings) > 5:
                lines.append(f"    ... and {len(self.warnings) - 5} more")

        return "\n".join(lines)


def find_parser_directory(document_path: Path) -> Path | None:
    """
    Find the parser directory for a document.

    Args:
        document_path: Path to markdown document

    Returns:
        Path to parser directory or None if not found

    Examples:
        docs/source/1_euclidean_gas/07_mean_field.md
        → docs/source/1_euclidean_gas/parser/
    """
    if not document_path.exists():
        logger.error(f"Document not found: {document_path}")
        return None

    # Get parent directory
    parent_dir = document_path.parent

    # Parser directory should be in same parent
    parser_dir = parent_dir / "parser"

    if not parser_dir.exists():
        logger.error(f"Parser directory not found: {parser_dir}")
        return None

    return parser_dir


def find_raw_data_directory(document_path: Path) -> Path:
    """
    Find or create the raw_data directory for a document.

    Args:
        document_path: Path to markdown document

    Returns:
        Path to raw_data directory (created if needed)

    Examples:
        docs/source/1_euclidean_gas/07_mean_field.md
        → docs/source/1_euclidean_gas/07_mean_field/raw_data/
    """
    # Get document name without extension
    doc_name = document_path.stem

    # raw_data directory should be {doc_name}/raw_data/
    parent_dir = document_path.parent
    raw_data_dir = parent_dir / doc_name / "raw_data"

    # Create if doesn't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    return raw_data_dir


def load_chapter_files(parser_dir: Path) -> list[dict[str, Any]]:
    """
    Load all chapter_N.json files from parser directory.

    Args:
        parser_dir: Path to parser directory

    Returns:
        List of chapter data dictionaries (sorted by chapter number)
    """
    chapter_files = sorted(parser_dir.glob("chapter_*.json"))

    if not chapter_files:
        logger.warning(f"No chapter files found in {parser_dir}")
        return []

    logger.info(f"Found {len(chapter_files)} chapter files")

    chapters = []
    for chapter_file in chapter_files:
        try:
            with open(chapter_file) as f:
                data = json.load(f)
                chapters.append(data)
        except Exception as e:
            logger.error(f"Failed to load {chapter_file}: {e}")

    return chapters


def extract_improvements(chapter_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract improvement changes from chapter data.

    Args:
        chapter_data: Chapter JSON data

    Returns:
        List of changes (empty if no improvements)
    """
    metadata = chapter_data.get("_improvement_metadata")

    if not metadata:
        return []

    changes = metadata.get("changes", [])
    return changes


def get_entity_file_path(
    raw_data_dir: Path,
    entity_type: str,
    label: str,
) -> Path:
    """
    Get file path for an entity in raw_data structure.

    Args:
        raw_data_dir: Path to raw_data directory
        entity_type: Type of entity (definition, theorem, etc.)
        label: Entity label

    Returns:
        Path to entity JSON file

    Examples:
        raw_data_dir = docs/source/.../raw_data/
        entity_type = "definition"
        label = "def-lipschitz"
        → docs/source/.../raw_data/definitions/def-lipschitz.json
    """
    # Map entity types to directory names (plural)
    type_dirs = {
        "definition": "definitions",
        "theorem": "theorems",
        "lemma": "lemmas",
        "proposition": "propositions",
        "corollary": "corollaries",
        "proof": "proofs",
        "axiom": "axioms",
        "assumption": "assumptions",
        "parameter": "parameters",
        "remark": "remarks",
        "citation": "citations",
    }

    dir_name = type_dirs.get(entity_type, f"{entity_type}s")
    entity_dir = raw_data_dir / dir_name

    # Create directory if needed
    entity_dir.mkdir(parents=True, exist_ok=True)

    return entity_dir / f"{label}.json"


def intelligent_merge(old_entity: dict, new_entity: dict) -> dict:
    """
    Intelligently merge old and new entity data.

    Strategy:
    - Preserve source.file_path from old (canonical location)
    - Update line_range if changed
    - Update content fields (term, statement, etc.)
    - Preserve _metadata if exists
    - Add _last_updated timestamp

    Args:
        old_entity: Original entity data
        new_entity: New entity data from improvement

    Returns:
        Merged entity data
    """
    merged = old_entity.copy()

    # Update line ranges if changed
    if "line_start" in new_entity:
        merged["line_start"] = new_entity["line_start"]
    if "line_end" in new_entity:
        merged["line_end"] = new_entity["line_end"]

    # Update full_text if changed
    if "full_text" in new_entity:
        merged["full_text"] = new_entity["full_text"]

    # Update content fields based on entity type
    content_fields = [
        "term",  # definitions
        "statement_type",  # theorems
        "conclusion_formula_latex",  # theorems
        "definition_references",  # theorems
        "theorem_references",  # theorems/proofs
        "proves_label",  # proofs
        "steps",  # proofs
        "parameters_mentioned",  # various
    ]

    for field in content_fields:
        if field in new_entity:
            merged[field] = new_entity[field]

    # Update source.line_range if changed
    if "source" in merged and "full_text" in new_entity:
        merged["source"]["line_range"] = new_entity["full_text"]

    # Add metadata about update
    if "_metadata" not in merged:
        merged["_metadata"] = {}

    merged["_metadata"]["last_updated"] = datetime.now().isoformat()
    merged["_metadata"]["updated_by"] = "integrate_improvements.py"

    return merged


def apply_add_operation(
    change: dict[str, Any],
    raw_data_dir: Path,
    stats: IntegrationStatistics,
    dry_run: bool = False,
) -> bool:
    """
    Apply ADD operation: create new entity file.

    Args:
        change: EntityChange dict with operation='ADD'
        raw_data_dir: Path to raw_data directory
        stats: Statistics tracker
        dry_run: If True, don't actually write files

    Returns:
        True if successful, False otherwise
    """
    entity_type = change["entity_type"]
    label = change["label"]
    new_data = change.get("new_data")

    if not new_data:
        stats.record_error(f"ADD operation missing new_data for {label}")
        return False

    # Get target file path
    file_path = get_entity_file_path(raw_data_dir, entity_type, label)

    # Check if file already exists
    if file_path.exists():
        stats.record_warning(f"File already exists for {label}, skipping ADD")
        return False

    # Save new entity
    if not dry_run:
        try:
            with open(file_path, "w") as f:
                json.dump(new_data, f, indent=2)
        except Exception as e:
            stats.record_error(f"Failed to write {label}: {e}")
            return False

    stats.record_add(label)
    return True


def apply_modify_operation(
    change: dict[str, Any],
    raw_data_dir: Path,
    stats: IntegrationStatistics,
    dry_run: bool = False,
) -> bool:
    """
    Apply MODIFY operation: update existing entity file with intelligent merge.

    Args:
        change: EntityChange dict with operation='MODIFY'
        raw_data_dir: Path to raw_data directory
        stats: Statistics tracker
        dry_run: If True, don't actually write files

    Returns:
        True if successful, False otherwise
    """
    entity_type = change["entity_type"]
    label = change["label"]
    new_data = change.get("new_data")

    if not new_data:
        stats.record_error(f"MODIFY operation missing new_data for {label}")
        return False

    # Get target file path
    file_path = get_entity_file_path(raw_data_dir, entity_type, label)

    # Load existing entity
    if not file_path.exists():
        stats.record_warning(f"File not found for MODIFY operation: {label}, treating as ADD")
        # Treat as ADD instead
        return apply_add_operation(change, raw_data_dir, stats, dry_run)

    try:
        with open(file_path) as f:
            old_data = json.load(f)
    except Exception as e:
        stats.record_error(f"Failed to load {label} for MODIFY: {e}")
        return False

    # Intelligent merge
    merged_data = intelligent_merge(old_data, new_data)

    # Save merged entity
    if not dry_run:
        try:
            with open(file_path, "w") as f:
                json.dump(merged_data, f, indent=2)
        except Exception as e:
            stats.record_error(f"Failed to write modified {label}: {e}")
            return False

    stats.record_modify(label)
    return True


def apply_delete_operation(
    change: dict[str, Any],
    raw_data_dir: Path,
    stats: IntegrationStatistics,
    dry_run: bool = False,
) -> bool:
    """
    Apply DELETE operation: remove entity file.

    Args:
        change: EntityChange dict with operation='DELETE'
        raw_data_dir: Path to raw_data directory
        stats: Statistics tracker
        dry_run: If True, don't actually delete files

    Returns:
        True if successful, False otherwise
    """
    entity_type = change["entity_type"]
    label = change["label"]

    # Get target file path
    file_path = get_entity_file_path(raw_data_dir, entity_type, label)

    # Check if file exists
    if not file_path.exists():
        stats.record_warning(f"File not found for DELETE operation: {label}")
        return False

    # Delete file
    if not dry_run:
        try:
            file_path.unlink()
        except Exception as e:
            stats.record_error(f"Failed to delete {label}: {e}")
            return False

    stats.record_delete(label)
    return True


def apply_change(
    change: dict[str, Any],
    raw_data_dir: Path,
    stats: IntegrationStatistics,
    dry_run: bool = False,
) -> bool:
    """
    Apply a single change operation.

    Args:
        change: EntityChange dict
        raw_data_dir: Path to raw_data directory
        stats: Statistics tracker
        dry_run: If True, preview changes without applying

    Returns:
        True if successful, False otherwise
    """
    operation = change.get("operation")

    if operation == "ADD":
        return apply_add_operation(change, raw_data_dir, stats, dry_run)
    elif operation == "MODIFY":
        return apply_modify_operation(change, raw_data_dir, stats, dry_run)
    elif operation == "DELETE":
        return apply_delete_operation(change, raw_data_dir, stats, dry_run)
    elif operation == "NO_CHANGE":
        # Skip NO_CHANGE operations
        return True
    else:
        stats.record_error(f"Unknown operation: {operation}")
        return False


def validate_raw_data_directory(raw_data_dir: Path) -> tuple[bool, list[str]]:
    """
    Validate all entities in raw_data directory.

    Args:
        raw_data_dir: Path to raw_data directory

    Returns:
        (is_valid, errors) tuple
    """
    errors = []

    # Check each entity type directory
    entity_dirs = [
        "definitions",
        "theorems",
        "lemmas",
        "propositions",
        "corollaries",
        "proofs",
        "axioms",
        "assumptions",
        "parameters",
        "remarks",
        "citations",
    ]

    total_entities = 0

    for entity_dir_name in entity_dirs:
        entity_dir = raw_data_dir / entity_dir_name

        if not entity_dir.exists():
            continue

        for entity_file in entity_dir.glob("*.json"):
            total_entities += 1

            try:
                with open(entity_file) as f:
                    entity_data = json.load(f)

                # Basic validation
                if "label" not in entity_data:
                    errors.append(f"{entity_file.name}: Missing 'label' field")

                if "source" not in entity_data:
                    errors.append(f"{entity_file.name}: Missing 'source' field")

            except json.JSONDecodeError as e:
                errors.append(f"{entity_file.name}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"{entity_file.name}: Validation error - {e}")

    logger.info(f"Validated {total_entities} entities")

    if errors:
        logger.error(f"Validation found {len(errors)} errors")
        return False, errors
    else:
        logger.info("✓ All entities valid")
        return True, []


def integrate_document_improvements(
    document_path: Path,
    dry_run: bool = False,
    validate: bool = False,
    verbose: bool = False,
) -> IntegrationStatistics:
    """
    Integrate improvement metadata from chapter files into raw_data structure.

    Args:
        document_path: Path to markdown document
        dry_run: If True, preview changes without applying
        validate: If True, validate raw_data after integration
        verbose: If True, enable verbose logging

    Returns:
        IntegrationStatistics with summary of operations
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Processing: {document_path}")

    if dry_run:
        logger.info("DRY-RUN MODE: No files will be modified")

    stats = IntegrationStatistics()

    # Find parser directory
    parser_dir = find_parser_directory(document_path)
    if not parser_dir:
        stats.record_error("Parser directory not found")
        return stats

    # Find raw_data directory
    raw_data_dir = find_raw_data_directory(document_path)
    logger.info(f"Raw data directory: {raw_data_dir}")

    # Load all chapter files
    chapters = load_chapter_files(parser_dir)

    if not chapters:
        stats.record_error("No chapter files found")
        return stats

    # Process each chapter
    for i, chapter_data in enumerate(chapters):
        logger.info(f"\nChapter {i}:")

        # Extract improvements
        changes = extract_improvements(chapter_data)

        if not changes:
            logger.info(f"  No changes")
            continue

        logger.info(f"  Processing {len(changes)} changes")

        # Apply each change
        for change in changes:
            try:
                apply_change(change, raw_data_dir, stats, dry_run)
            except Exception as e:
                stats.record_error(f"Failed to apply change: {e}")
                logger.debug(f"  Change: {change}")

    # Validate if requested
    if validate and not dry_run:
        logger.info("\nValidating integrated data...")
        is_valid, validation_errors = validate_raw_data_directory(raw_data_dir)

        if not is_valid:
            for err in validation_errors:
                stats.record_error(f"Validation: {err}")

    return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Integrate improvement metadata into raw_data structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Integrate improvements for a single document
  python -m mathster.tools.integrate_improvements docs/source/1_euclidean_gas/07_mean_field.md

  # Preview changes without applying (dry-run)
  python -m mathster.tools.integrate_improvements docs/source/1_euclidean_gas/07_mean_field.md --dry-run

  # Integrate and validate
  python -m mathster.tools.integrate_improvements docs/source/1_euclidean_gas/07_mean_field.md --validate

  # Verbose output for debugging
  python -m mathster.tools.integrate_improvements docs/source/1_euclidean_gas/07_mean_field.md --verbose
        """,
    )

    parser.add_argument(
        "document",
        type=Path,
        help="Path to markdown document",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate raw_data after integration",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Resolve document path
    document_path = args.document.resolve()

    if not document_path.exists():
        logger.error(f"Document not found: {document_path}")
        return 1

    # Run integration
    stats = integrate_document_improvements(
        document_path=document_path,
        dry_run=args.dry_run,
        validate=args.validate,
        verbose=args.verbose,
    )

    # Print summary
    print(stats.get_summary())

    # Exit with error code if there were errors
    if stats.errors:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())
