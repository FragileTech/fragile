"""
Source Location Enrichment Tool.

This module enriches raw JSON files with precise SourceLocation data, linking
mathematical entities to their exact line ranges in source markdown documents.

Key Features:
- Finds exact line ranges for extracted entities using text matching
- Creates SourceLocation objects with multiple precision levels (line → directive → section)
- Supports batch enrichment of entire directories
- Validates line ranges against markdown files
- Provides fallback strategies when exact matching fails

Workflow:
1. Read raw_data/*.json files
2. For each entity, extract identifying text (statement, definition, etc.)
3. Use line_finder to locate text in source markdown
4. Create SourceLocation with found line_range
5. Write enriched JSON with "source_location" field

Maps to Lean:
    namespace SourceLocationEnricher
      def enrich_single_entity : RawEntity → Path → String → Option SourceLocation
      def enrich_directory : Path → Path → String → IO Unit
      def batch_enrich_all_documents : Path → IO Unit
    end SourceLocationEnricher
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.tools.line_finder import (
    find_directive_lines,
    find_text_in_markdown,
    get_file_line_count,
    validate_line_range,
)
from fragile.proofs.utils.source_helpers import SourceLocationBuilder


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# ENTITY-SPECIFIC TEXT EXTRACTION
# =============================================================================


def extract_search_text_from_entity(entity_data: dict[str, Any]) -> str | None:
    """
    Extract the best text snippet to search for from a raw entity.

    Different entity types have different fields that make good search targets:
    - RawTheorem/RawLemma: full_statement_text
    - RawDefinition: term_being_defined + beginning of full_text
    - RawProof: strategy_text or beginning of full_body_text
    - RawAxiom: core_assumption_text
    - RawEquation: latex_content (may need special handling)

    Args:
        entity_data: Dictionary containing the raw entity data

    Returns:
        Text snippet suitable for searching, or None if entity lacks searchable text

    Examples:
        >>> entity = {"full_statement_text": "Let v > 0 and assume..."}
        >>> extract_search_text_from_entity(entity)
        "Let v > 0 and assume..."

    Maps to Lean:
        def extract_search_text_from_entity (entity : Json) : Option String
    """
    # Try theorem-like fields (theorems, lemmas, propositions, corollaries)
    if "full_statement_text" in entity_data:
        text = entity_data["full_statement_text"]
        # Truncate to first 200 chars for better matching
        return text[:200] if len(text) > 200 else text

    # Try definition fields
    if "full_text" in entity_data and "term_being_defined" in entity_data:
        term = entity_data["term_being_defined"]
        full_text = entity_data["full_text"]
        # Use term + beginning of definition (up to 200 chars)
        combined = f"{term}. {full_text}"
        return combined[:200] if len(combined) > 200 else combined

    # Try axiom fields
    if "core_assumption_text" in entity_data:
        text = entity_data["core_assumption_text"]
        return text[:200] if len(text) > 200 else text

    # Try proof fields
    if entity_data.get("strategy_text"):
        text = entity_data["strategy_text"]
        return text[:150] if len(text) > 150 else text

    if entity_data.get("full_body_text"):
        text = entity_data["full_body_text"]
        return text[:150] if len(text) > 150 else text

    # Try equation fields (special handling)
    if "latex_content" in entity_data:
        # For equations, we'll use directive label instead (equations are usually wrapped)
        return None  # Signal to use directive label fallback

    # Try generic full_text field
    if "full_text" in entity_data:
        text = entity_data["full_text"]
        return text[:200] if len(text) > 200 else text

    return None


def extract_directive_label_from_entity(entity_data: dict[str, Any]) -> str | None:
    """
    Extract directive label from entity data if present.

    Some entities may have been extracted from Jupyter Book directives
    and include the label in their temp_id or label_text fields.

    Args:
        entity_data: Dictionary containing the raw entity data

    Returns:
        Directive label if extractable, None otherwise

    Examples:
        >>> entity = {"label_text": "thm-keystone"}
        >>> extract_directive_label_from_entity(entity)
        "thm-keystone"

    Maps to Lean:
        def extract_directive_label_from_entity (entity : Json) : Option String
    """
    # Check for explicit label_text (used in refined data)
    if "label_text" in entity_data:
        label = entity_data["label_text"]
        # Check if it's a directive label format (e.g., "thm-keystone", "def-walker")
        if isinstance(label, str) and "-" in label:
            return label

    # Check temp_id for hints
    if "temp_id" in entity_data:
        temp_id = entity_data["temp_id"]
        # temp_id like "raw-thm-1" doesn't help, but sometimes contains actual label
        if isinstance(temp_id, str) and not temp_id.startswith("raw-"):
            return temp_id

    return None


# =============================================================================
# CORE ENRICHMENT LOGIC
# =============================================================================


def find_entity_location(
    entity_data: dict[str, Any],
    markdown_content: str,
    document_id: str,
    file_path: str,
) -> SourceLocation | None:
    """
    Find the source location for a raw entity using multiple strategies.

    Tries strategies in order of precision:
    1. Line-range matching: Find exact text in markdown → most precise
    2. Directive matching: Find by Jupyter Book directive label
    3. Section fallback: Use source_section from entity
    4. Minimal fallback: Just document_id + file_path

    Args:
        entity_data: Dictionary containing the raw entity data
        markdown_content: Full content of the markdown file
        document_id: Document ID (e.g., "03_cloning")
        file_path: Path to markdown file

    Returns:
        SourceLocation with best available precision, or None if document_id/file_path invalid

    Examples:
        >>> entity = {"full_statement_text": "Let v > 0..."}
        >>> loc = find_entity_location(
        ...     entity, content, "03_cloning", "docs/.../03_cloning.md"
        ... )
        >>> loc.line_range
        (142, 158)

    Maps to Lean:
        def find_entity_location
          (entity : Json)
          (content : String)
          (document_id : String)
          (file_path : String)
          : Option SourceLocation
    """
    # Validate document_id format
    if not document_id or not file_path:
        logger.warning("Invalid document_id or file_path")
        return None

    # Get total line count for validation
    max_lines = get_file_line_count(markdown_content)

    # STRATEGY 1: Line-range matching (most precise)
    search_text = extract_search_text_from_entity(entity_data)
    if search_text:
        line_range = find_text_in_markdown(markdown_content, search_text, case_sensitive=False)
        if line_range and validate_line_range(line_range, max_lines):
            logger.debug(f"Found entity at lines {line_range} using text matching")
            section = entity_data.get("source_section")
            return SourceLocationBuilder.from_markdown_location(
                document_id=document_id,
                file_path=file_path,
                start_line=line_range[0],
                end_line=line_range[1],
                section=section,
            )

    # STRATEGY 2: Directive label matching
    directive_label = extract_directive_label_from_entity(entity_data)
    if directive_label:
        line_range = find_directive_lines(markdown_content, directive_label)
        if line_range and validate_line_range(line_range, max_lines):
            logger.debug(
                f"Found entity at lines {line_range} using directive label '{directive_label}'"
            )
            section = entity_data.get("source_section")
            return SourceLocation(
                document_id=document_id,
                file_path=file_path,
                line_range=line_range,
                directive_label=directive_label,
                section=section,
                url_fragment=f"#{directive_label}",
            )

    # STRATEGY 3: Section fallback
    section = entity_data.get("source_section")
    if section:
        logger.debug(f"Using section fallback: {section}")
        return SourceLocationBuilder.from_section(
            document_id=document_id,
            file_path=file_path,
            section=section,
        )

    # STRATEGY 4: Minimal fallback
    logger.debug("Using minimal fallback (document-level only)")
    return SourceLocationBuilder.minimal(
        document_id=document_id,
        file_path=file_path,
    )


def enrich_single_entity(
    entity_json_path: Path,
    markdown_file: Path,
    document_id: str,
    output_path: Path | None = None,
) -> bool:
    """
    Enrich a single raw JSON file with source location data.

    Args:
        entity_json_path: Path to the raw JSON file to enrich
        markdown_file: Path to the source markdown file
        document_id: Document ID (e.g., "03_cloning")
        output_path: Optional output path; if None, overwrites input file

    Returns:
        True if enrichment succeeded, False otherwise

    Examples:
        >>> enrich_single_entity(
        ...     Path("raw_data/theorems/thm-keystone.json"),
        ...     Path("docs/source/1_euclidean_gas/03_cloning.md"),
        ...     "03_cloning",
        ... )
        True

    Maps to Lean:
        def enrich_single_entity
          (entity_path : Path)
          (markdown_path : Path)
          (document_id : String)
          (output_path : Option Path)
          : IO Bool
    """
    try:
        # Read entity JSON
        with open(entity_json_path, encoding="utf-8") as f:
            entity_data = json.load(f)

        # Read markdown content
        with open(markdown_file, encoding="utf-8") as f:
            markdown_content = f.read()

        # Find source location
        file_path = str(markdown_file)
        source_location = find_entity_location(
            entity_data=entity_data,
            markdown_content=markdown_content,
            document_id=document_id,
            file_path=file_path,
        )

        if source_location is None:
            logger.error(f"Failed to create source location for {entity_json_path}")
            return False

        # Add source location to entity data
        entity_data["source"] = source_location.model_dump(mode="json")

        # Write enriched JSON
        output = output_path or entity_json_path
        with open(output, "w", encoding="utf-8") as f:
            json.dump(entity_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Enriched {entity_json_path.name} → {source_location.get_display_location()}")
        return True

    except Exception as e:
        logger.error(f"Error enriching {entity_json_path}: {e}")
        return False


def enrich_directory(
    raw_data_dir: Path,
    markdown_file: Path,
    document_id: str,
    entity_types: list[str] | None = None,
) -> tuple[int, int]:
    """
    Enrich all raw JSON files in a directory with source locations.

    Args:
        raw_data_dir: Directory containing raw_data/ subdirectories (theorems/, definitions/, etc.)
        markdown_file: Path to the source markdown file
        document_id: Document ID
        entity_types: Optional list of entity types to enrich (e.g., ["theorems", "definitions"]).
                     If None, enriches all entity types.

    Returns:
        Tuple of (succeeded_count, total_count)

    Examples:
        >>> enrich_directory(
        ...     Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data"),
        ...     Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md"),
        ...     "01_fragile_gas_framework",
        ... )
        (47, 50)  # 47 out of 50 entities enriched successfully

    Maps to Lean:
        def enrich_directory
          (raw_dir : Path)
          (markdown_path : Path)
          (document_id : String)
          (entity_types : Option (List String))
          : IO (Nat × Nat)
    """
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_data_dir}")
        return (0, 0)

    if not markdown_file.exists():
        logger.error(f"Markdown file does not exist: {markdown_file}")
        return (0, 0)

    # Define entity type subdirectories
    all_entity_types = [
        "theorems",
        "definitions",
        "axioms",
        "proofs",
        "equations",
        "parameters",
        "remarks",
    ]

    types_to_process = entity_types or all_entity_types

    succeeded = 0
    total = 0

    for entity_type in types_to_process:
        entity_dir = raw_data_dir / entity_type
        if not entity_dir.exists():
            logger.debug(f"Skipping {entity_type}: directory does not exist")
            continue

        # Process all JSON files in this entity type directory
        json_files = list(entity_dir.glob("*.json"))
        logger.info(f"Processing {len(json_files)} {entity_type}...")

        for json_file in json_files:
            total += 1
            if enrich_single_entity(json_file, markdown_file, document_id):
                succeeded += 1

    logger.info(f"Enrichment complete: {succeeded}/{total} entities succeeded")
    return (succeeded, total)


def batch_enrich_all_documents(
    docs_source_dir: Path,
    entity_types: list[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Batch enrich all documents in the docs/source/ directory.

    Automatically discovers all documents with raw_data/ subdirectories
    and enriches them. Useful for processing the entire corpus.

    Args:
        docs_source_dir: Path to docs/source/ directory containing chapters
        entity_types: Optional list of entity types to enrich

    Returns:
        Dictionary mapping document_id → (succeeded, total) for each document

    Examples:
        >>> results = batch_enrich_all_documents(Path("docs/source"))
        >>> results["03_cloning"]
        (12, 12)
        >>> results["01_fragile_gas_framework"]
        (47, 50)

    Maps to Lean:
        def batch_enrich_all_documents
          (docs_dir : Path)
          (entity_types : Option (List String))
          : IO (Map String (Nat × Nat))
    """
    if not docs_source_dir.exists():
        logger.error(f"Docs source directory does not exist: {docs_source_dir}")
        return {}

    results = {}

    # Discover all chapter directories (1_euclidean_gas, 2_geometric_gas, etc.)
    chapter_dirs = [d for d in docs_source_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]

    for chapter_dir in chapter_dirs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing chapter: {chapter_dir.name}")
        logger.info(f"{'=' * 60}")

        # Find all document subdirectories with raw_data/
        for doc_dir in chapter_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            raw_data_dir = doc_dir / "raw_data"
            if not raw_data_dir.exists():
                continue

            # Extract document_id from directory name
            document_id = doc_dir.name

            # Find corresponding markdown file
            markdown_file = doc_dir.parent / f"{document_id}.md"
            if not markdown_file.exists():
                logger.warning(f"Markdown file not found for {document_id}: {markdown_file}")
                continue

            logger.info(f"\nEnriching document: {document_id}")
            succeeded, total = enrich_directory(
                raw_data_dir=raw_data_dir,
                markdown_file=markdown_file,
                document_id=document_id,
                entity_types=entity_types,
            )

            results[document_id] = (succeeded, total)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("BATCH ENRICHMENT SUMMARY")
    logger.info(f"{'=' * 60}")

    total_succeeded = sum(s for s, _ in results.values())
    total_count = sum(t for _, t in results.values())

    for doc_id, (succeeded, total) in sorted(results.items()):
        status = "✓" if succeeded == total else "⚠"
        logger.info(f"{status} {doc_id}: {succeeded}/{total}")

    logger.info(f"\nOverall: {total_succeeded}/{total_count} entities enriched successfully")
    logger.info(f"Success rate: {100 * total_succeeded / total_count:.1f}%")

    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Enrich raw JSON files with precise source locations"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single entity command
    single_parser = subparsers.add_parser("single", help="Enrich a single JSON file")
    single_parser.add_argument("json_file", type=Path, help="Path to raw JSON file")
    single_parser.add_argument("markdown_file", type=Path, help="Path to source markdown file")
    single_parser.add_argument("document_id", help="Document ID (e.g., '03_cloning')")
    single_parser.add_argument(
        "--output", "-o", type=Path, help="Output path (default: overwrite input)"
    )

    # Directory command
    dir_parser = subparsers.add_parser(
        "directory", help="Enrich all files in a raw_data directory"
    )
    dir_parser.add_argument("raw_data_dir", type=Path, help="Path to raw_data directory")
    dir_parser.add_argument("markdown_file", type=Path, help="Path to source markdown file")
    dir_parser.add_argument("document_id", help="Document ID")
    dir_parser.add_argument(
        "--types", "-t", nargs="+", help="Entity types to process (default: all)"
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Batch enrich all documents in docs/source/"
    )
    batch_parser.add_argument("docs_source_dir", type=Path, help="Path to docs/source directory")
    batch_parser.add_argument(
        "--types", "-t", nargs="+", help="Entity types to process (default: all)"
    )

    args = parser.parse_args()

    if args.command == "single":
        success = enrich_single_entity(
            args.json_file, args.markdown_file, args.document_id, args.output
        )
        sys.exit(0 if success else 1)

    elif args.command == "directory":
        succeeded, total = enrich_directory(
            args.raw_data_dir, args.markdown_file, args.document_id, args.types
        )
        sys.exit(0 if succeeded == total else 1)

    elif args.command == "batch":
        results = batch_enrich_all_documents(args.docs_source_dir, args.types)
        # Exit with error if any document failed
        all_success = all(s == t for s, t in results.values())
        sys.exit(0 if all_success else 1)

    else:
        parser.print_help()
        sys.exit(1)
