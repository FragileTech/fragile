"""
DSPy tool wrappers for validation, comparison, and label checking.

Provides tool functions that ReAct agents can call during extraction
and improvement workflows for self-validation and comparison.
"""

import json

from mathster.parsing.validation.validators import validate_extraction


def validate_extraction_tool(extraction_json: str, context: str) -> str:
    """
    Tool for ReAct agent to validate its extraction.

    Args:
        extraction_json: JSON string representing ChapterExtraction
        context: Context JSON dict as a string containing file_path, article_id, chapter_text

    Returns:
        Validation feedback string for the agent
    """
    try:
        # Parse extraction
        extraction_dict = json.loads(extraction_json)

        # Parse context
        context_dict = json.loads(context)
        file_path = context_dict.get("file_path", "")
        article_id = context_dict.get("article_id", "")
        chapter_text = context_dict.get("chapter_text", "")

        # Validate
        result = validate_extraction(
            extraction_dict, file_path=file_path, article_id=article_id, chapter_text=chapter_text
        )

        return result.get_feedback()

    except Exception as e:
        return f"Validation tool error: {e!s}"


def compare_labels_tool(extraction_json: str, context: str) -> str:
    """
    Tool to compare extracted labels against source document.

    Args:
        extraction_json: JSON string representing ChapterExtraction
        context: Context containing chapter text and expected labels

    Returns:
        Comparison feedback string
    """
    try:
        from mathster.parsing.text_processing.analysis import compare_extraction_with_source

        extraction_dict = json.loads(extraction_json)
        context_dict = json.loads(context)

        chapter_text = context_dict.get("chapter_text", "")

        # Compare using text processing tools
        return compare_extraction_with_source(extraction_dict, chapter_text)

    except Exception as e:
        return f"Comparison tool error: {e!s}"


def validate_single_entity_tool(entity_json: str, context: str) -> str:
    """
    Tool to validate a single entity extraction.

    Args:
        entity_json: JSON string representing a single entity
        context: Context containing entity type and validation parameters

    Returns:
        Validation feedback for the single entity
    """
    try:
        entity_dict = json.loads(entity_json)
        context_dict = json.loads(context)

        entity_type = context_dict.get("entity_type", "")

        # Basic validation checks
        if "label" not in entity_dict:
            return "Error: Entity missing 'label' field"

        if "line_start" not in entity_dict or "line_end" not in entity_dict:
            return "Error: Entity missing line range fields"

        # Validate label pattern
        label = entity_dict["label"]
        expected_prefixes = {
            "definitions": ["def-"],
            "theorems": ["thm-", "lem-", "prop-", "cor-"],
            "proofs": ["proof-"],
            "axioms": ["axiom-", "ax-", "def-axiom-"],
            "parameters": ["param-"],
            "remarks": ["remark-"],
            "assumptions": ["assumption-"],
        }

        prefixes = expected_prefixes.get(entity_type, [])
        if not any(label.startswith(p) for p in prefixes):
            return f"Error: Label '{label}' doesn't match expected pattern for {entity_type}"

        return f"✓ Entity '{label}' validated successfully"

    except Exception as e:
        return f"Entity validation error: {e!s}"


def compare_extractions_tool(existing_json: str, improved_json: str) -> str:
    """
    Tool to compare existing extraction with improved version.

    Used during improvement workflow to show what changed.

    Args:
        existing_json: JSON string of original extraction
        improved_json: JSON string of improved extraction

    Returns:
        Comparison summary showing added/modified/deleted entities
    """
    try:
        existing = json.loads(existing_json)
        improved = json.loads(improved_json)

        changes = []

        # Compare each entity type
        entity_types = [
            "definitions",
            "theorems",
            "proofs",
            "axioms",
            "parameters",
            "remarks",
            "citations",
            "assumptions",
        ]

        for etype in entity_types:
            existing_labels = {e.get("label") for e in existing.get(etype, [])}
            improved_labels = {e.get("label") for e in improved.get(etype, [])}

            added = improved_labels - existing_labels
            removed = existing_labels - improved_labels

            if added:
                changes.append(f"Added {len(added)} {etype}: {', '.join(sorted(added))}")
            if removed:
                changes.append(f"Removed {len(removed)} {etype}: {', '.join(sorted(removed))}")

        if not changes:
            return "No changes detected between extractions"

        return "\n".join(changes)

    except Exception as e:
        return f"Comparison error: {e!s}"


def validate_improvement_tool(improved_json: str, context: str) -> str:
    """
    Tool to validate improved extraction.

    Args:
        improved_json: JSON string of improved extraction
        context: Context with file_path, article_id, chapter_text

    Returns:
        Validation feedback for improved extraction
    """
    # Reuse validate_extraction_tool
    return validate_extraction_tool(improved_json, context)
