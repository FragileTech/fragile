#!/usr/bin/env python3
"""
Load entities from enriched/ directories.

This loader reads chapter_N.json files from per-document enriched/ folders
and flattens them into the unified registry format, similar to ParserLoader
but loading from the enriched data source.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EnrichedDataLoader:
    """Load and flatten entities from {document}/enriched/chapter_N.json files."""

    def __init__(self):
        self.entities_loaded = 0
        self.chapters_processed = 0
        self.errors = []

    def load_from_enriched_directory(self, enriched_dir: Path, document_id: str) -> list[dict]:
        """
        Load all entities from enriched directory.

        Args:
            enriched_dir: Path to {document}/enriched/ directory
            document_id: Document identifier (e.g., "07_mean_field")

        Returns:
            List of flattened entity dicts
        """
        logger.info(f"Loading from enriched directory: {enriched_dir}")

        if not enriched_dir.exists():
            logger.warning(f"Enriched directory not found: {enriched_dir}")
            return []

        all_entities = []
        chapter_files = sorted(enriched_dir.glob("chapter_*.json"))

        if not chapter_files:
            logger.warning(f"No chapter files found in {enriched_dir}")
            return []

        for chapter_file in chapter_files:
            # Extract chapter number from filename
            try:
                chapter_num = int(chapter_file.stem.split("_")[1])
            except (IndexError, ValueError):
                logger.warning(f"Invalid chapter filename: {chapter_file.name}")
                continue

            entities = self._load_chapter_file(chapter_file, chapter_num, document_id)
            all_entities.extend(entities)
            self.chapters_processed += 1

        logger.info(f"Loaded {len(all_entities)} entities from {len(chapter_files)} enriched chapters")
        return all_entities

    def _load_chapter_file(
        self, chapter_file: Path, chapter_num: int, document_id: str
    ) -> list[dict]:
        """
        Load and flatten a single enriched chapter_N.json file.

        Args:
            chapter_file: Path to enriched chapter JSON file
            chapter_num: Chapter number
            document_id: Document identifier

        Returns:
            List of flattened entities from this chapter
        """
        try:
            with open(chapter_file, encoding="utf-8") as f:
                section_data = json.load(f)
        except Exception as e:
            error = f"Failed to load {chapter_file}: {e}"
            logger.error(error)
            self.errors.append(error)
            return []

        entities = []

        # Entity types to process
        entity_types = [
            "definitions",
            "theorems",
            "proofs",
            "axioms",
            "assumptions",
            "parameters",
            "remarks",
            "citations",
        ]

        section_id = section_data.get("section_id", "")

        # Flatten each entity type
        for entity_type in entity_types:
            entity_list = section_data.get(entity_type, [])

            for entity in entity_list:
                # Convert to dict if it's a Pydantic model
                if hasattr(entity, "model_dump"):
                    entity_dict = entity.model_dump()
                elif isinstance(entity, dict):
                    entity_dict = entity.copy()
                else:
                    logger.warning(f"Unexpected entity format: {type(entity)}")
                    continue

                # Add dashboard metadata
                entity_dict["_entity_type"] = entity_type.rstrip("s")  # "definitions" → "definition"
                entity_dict["_source_type"] = "enriched"
                entity_dict["_section_id"] = section_id
                entity_dict["_chapter_number"] = chapter_num
                entity_dict["_enriched_file"] = str(chapter_file.name)
                entity_dict["_document_id"] = document_id
                entity_dict["_chapter"] = None  # Will be set by builder

                # Extract line number for sorting
                source = entity_dict.get("source", {})
                line_range = source.get("line_range")

                if line_range and isinstance(line_range, dict):
                    lines = line_range.get("lines", [])
                    if lines and isinstance(lines, list) and len(lines) > 0:
                        if isinstance(lines[0], list) and len(lines[0]) >= 1:
                            entity_dict["_line_number"] = lines[0][0]
                        else:
                            entity_dict["_line_number"] = float("inf")
                    else:
                        entity_dict["_line_number"] = float("inf")
                else:
                    entity_dict["_line_number"] = float("inf")

                entities.append(entity_dict)
                self.entities_loaded += 1

        return entities
