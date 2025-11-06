"""
Text extraction utilities for enriching raw data entities.

Populates full_text fields in RawDocumentSection entities by reading source files
and extracting text content from line ranges. This is pure Python file I/O - no LLM required.

Usage:
    from mathster.enrichment import extract_full_text

    # Load chapter_N.json (has empty full_text fields)
    with open("chapter_0.json") as f:
        section_data = json.load(f)

    # Parse as RawDocumentSection
    from mathster.core.raw_data import RawDocumentSection
    section = RawDocumentSection(**section_data)

    # Extract text
    enriched = extract_full_text(section)

    # Now enriched.full_text and all entity full_text fields are populated
"""

import json
import logging
from pathlib import Path

from mathster.core.raw_data import RawDocumentSection

logger = logging.getLogger(__name__)


def extract_full_text(section: RawDocumentSection) -> RawDocumentSection:
    """
    Populate all full_text fields for a RawDocumentSection.

    Reads the source markdown file and extracts text content for:
    - Section-level full_text
    - All entity full_text fields (definitions, theorems, proofs, etc.)

    This is a pure Python utility - reads file once, extracts all text.

    Args:
        section: RawDocumentSection with empty full_text fields

    Returns:
        RawDocumentSection with all full_text fields populated

    Raises:
        FileNotFoundError: If source file doesn't exist
        ValueError: If line ranges are invalid
    """
    logger.info(f"Extracting text for section: {section.section_id}")

    # Check if already enriched
    if section.full_text != "":
        logger.warning("Section already has full_text, skipping enrichment")
        return section

    # Read source file once
    file_path = Path(section.source.file_path)
    if not file_path.exists():
        # Try relative to project root
        from pathlib import Path as P

        project_root = P(__file__).parent.parent.parent.parent
        file_path = project_root / section.source.file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {section.source.file_path}")

    logger.debug(f"Reading file: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        all_lines = f.readlines()

    # Extract section-level text
    section_text = _extract_text_from_source(section.source, all_lines)

    # Create updated section with populated text
    enriched_data = {
        "section_id": section.section_id,
        "full_text": section_text,
        "source": section.source,
        # Enrich each entity type
        "definitions": [_enrich_entity_text(d, all_lines) for d in section.definitions],
        "theorems": [_enrich_entity_text(t, all_lines) for t in section.theorems],
        "proofs": [_enrich_entity_text(p, all_lines) for p in section.proofs],
        "axioms": [_enrich_entity_text(a, all_lines) for a in section.axioms],
        "assumptions": [_enrich_entity_text(a, all_lines) for a in section.assumptions],
        "parameters": [_enrich_entity_text(p, all_lines) for p in section.parameters],
        "remarks": [_enrich_entity_text(r, all_lines) for r in section.remarks],
        "citations": [_enrich_entity_text(c, all_lines) for c in section.citations],
    }

    # Create new RawDocumentSection with enriched text
    enriched_section = RawDocumentSection(**enriched_data)

    logger.info(f"✓ Enriched {section.total_entities} entities")
    return enriched_section


def _enrich_entity_text(entity, all_lines: list[str]):
    """
    Enrich a single entity's full_text field.

    Args:
        entity: RawDataModel subclass instance
        all_lines: All lines from source file

    Returns:
        Entity dict with full_text populated
    """
    entity_dict = entity.model_dump()

    # Extract text if full_text is empty
    if entity_dict.get("full_text") == "":
        source = entity.source
        full_text = _extract_text_from_source(source, all_lines)
        entity_dict["full_text"] = full_text

    return entity_dict


def _extract_text_from_source(source, all_lines: list[str]) -> str:
    """
    Extract text content from SourceLocation line ranges.

    Args:
        source: SourceLocation object
        all_lines: All lines from the source file

    Returns:
        Extracted text content
    """
    line_range = source.line_range
    text_blocks = []

    for start, end in line_range.lines:
        # Validate bounds
        if start < 1 or end > len(all_lines):
            raise ValueError(
                f"Line range [{start}, {end}] out of bounds (file has {len(all_lines)} lines)"
            )

        # Extract lines (1-indexed inclusive)
        block = "".join(all_lines[start - 1 : end])
        text_blocks.append(block.rstrip())

    # Join discontinuous blocks
    return "\n[...]\n".join(text_blocks)


def enrich_chapter_file(
    chapter_file: Path,
    output_file: Path | None = None,
) -> Path:
    """
    Enrich a chapter_N.json file with full text content.

    Works with raw JSON dicts to avoid Pydantic validation issues.

    Args:
        chapter_file: Path to chapter_N.json
        output_file: Optional output path (default: chapter_N_enriched.json)

    Returns:
        Path to enriched file
    """
    logger.info(f"Enriching: {chapter_file}")

    # Load chapter data as dict (avoid Pydantic validation issues)
    with open(chapter_file, encoding="utf-8") as f:
        section_data = json.load(f)

    # Extract text using dict-based approach
    enriched_data = extract_full_text_from_dict(section_data)

    # Determine output file
    if output_file is None:
        output_file = chapter_file.parent / f"{chapter_file.stem}_enriched.json"

    # Save enriched data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2)

    logger.info(f"✓ Saved to: {output_file}")
    return output_file


def extract_full_text_from_dict(section_data: dict) -> dict:
    """
    Extract full text from a chapter dict (avoids Pydantic validation).

    Args:
        section_data: Chapter JSON as dict

    Returns:
        Enriched chapter dict with full_text populated
    """
    from mathster.core.article_system import SourceLocation

    # Get source location
    source_dict = section_data.get("source", {})
    file_path = source_dict.get("file_path")

    if not file_path:
        logger.warning("No file_path in source, skipping enrichment")
        return section_data

    # Read file
    path = Path(file_path)
    if not path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        path = project_root / file_path

    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return section_data

    with open(path, encoding="utf-8") as f:
        all_lines = f.readlines()

    # Extract section-level text
    if section_data.get("full_text") == "":
        section_lines = source_dict.get("line_range", {}).get("lines", [])
        if section_lines:
            section_text = _extract_text_from_lines(section_lines, all_lines)
            section_data["full_text"] = section_text

    # Enrich each entity type
    for entity_type in ["definitions", "theorems", "proofs", "axioms", "assumptions", "parameters", "remarks", "citations"]:
        entities = section_data.get(entity_type, [])
        for entity in entities:
            full_text = entity.get("full_text")

            # Check if full_text needs extraction
            # It can be: "" (empty string), dict/TextLocation, or already extracted text
            needs_extraction = False

            if full_text == "":
                needs_extraction = True
            elif isinstance(full_text, dict):
                # full_text is a TextLocation dict
                entity_lines = full_text.get("lines", [])
                if entity_lines:
                    entity_text = _extract_text_from_lines(entity_lines, all_lines)
                    entity["full_text"] = entity_text
                continue

            if needs_extraction:
                entity_source = entity.get("source", {})
                entity_lines = entity_source.get("line_range", {}).get("lines", [])
                if entity_lines:
                    entity_text = _extract_text_from_lines(entity_lines, all_lines)
                    entity["full_text"] = entity_text

    logger.info(f"✓ Enriched section + {sum(len(section_data.get(t, [])) for t in ['definitions', 'theorems', 'proofs', 'axioms', 'assumptions', 'parameters', 'remarks', 'citations'])} entities")
    return section_data


def _extract_text_from_lines(line_ranges: list[list[int]], all_lines: list[str]) -> str:
    """Extract text from line ranges."""
    text_blocks = []

    for start, end in line_ranges:
        if start < 1 or end > len(all_lines):
            logger.warning(f"Line range [{start}, {end}] out of bounds (file has {len(all_lines)} lines)")
            continue

        # Extract lines (1-indexed inclusive)
        block = "".join(all_lines[start - 1 : end])
        text_blocks.append(block.rstrip())

    return "\n[...]\n".join(text_blocks)
