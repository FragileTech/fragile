"""
Semantic validation workflow for enriched mathematical entities.

Validates that extracted text matches entity types and definitions using DSPy agents.
Especially important for parameters which lack directive markers.

Usage:
    from mathster.enrichment.workflows import validate_enriched_chapter

    report = validate_enriched_chapter(
        "docs/source/1_euclidean_gas/parser/chapter_3_enriched.json",
        entity_types=["parameters"],  # Focus on parameters
    )

    print(f"Valid: {report['valid']}, Invalid: {report['invalid']}")
"""

import json
import logging
from pathlib import Path

from mathster.dspy_integration import configure_dspy
from mathster.enrichment.dspy_components.validators import SemanticValidator

logger = logging.getLogger(__name__)


def validate_enriched_chapter(
    enriched_file: Path,
    entity_types: list[str] | None = None,
    confidence_threshold: str = "medium",
    max_entities: int | None = None,
) -> dict:
    """
    Semantically validate entities in an enriched chapter using DSPy.

    Args:
        enriched_file: Path to chapter_N_enriched.json
        entity_types: Which types to validate (None = all)
        confidence_threshold: Minimum confidence: "high", "medium", "low"
        max_entities: Max entities to validate (None = all, for testing use small number)

    Returns:
        Validation report dict with statistics and errors
    """
    logger.info(f"Validating: {enriched_file}")

    # Load enriched data
    with open(enriched_file, encoding="utf-8") as f:
        data = json.load(f)

    # Configure DSPy with gemini-flash-lite (fast and cheap)
    configure_dspy(
        model="gemini/gemini-2.5-flash-lite",
        temperature=0.0,
        max_tokens=5000,  # Don't need much for validation
    )

    # Initialize validator
    validator = SemanticValidator()

    # Default: validate all entity types
    if entity_types is None:
        entity_types = ["definitions", "theorems", "proofs", "axioms", "assumptions", "parameters", "remarks"]

    # Track results
    results = {
        "file": str(enriched_file),
        "total_validated": 0,
        "valid": 0,
        "invalid": 0,
        "low_confidence": 0,
        "errors": [],
        "by_type": {},
    }

    # Validate each entity type
    for entity_type in entity_types:
        entities = data.get(entity_type, [])

        if not entities:
            continue

        logger.info(f"Validating {len(entities)} {entity_type}...")

        type_stats = {"valid": 0, "invalid": 0, "low_confidence": 0}

        # Limit if max_entities specified
        entities_to_validate = entities[:max_entities] if max_entities else entities

        for entity in entities_to_validate:
            label = entity.get("label", "unknown")
            full_text = entity.get("full_text", "")

            # Skip if no text
            if not full_text or full_text == "":
                logger.warning(f"  {label}: No full_text, skipping")
                continue

            # Prepare metadata based on entity type
            metadata = {}
            if entity_type in ["definitions"]:
                metadata["term"] = entity.get("term")
            elif entity_type in ["theorems", "lemmas", "propositions", "corollaries"]:
                metadata["statement_type"] = entity.get("statement_type")
                metadata["conclusion_formula"] = entity.get("conclusion_formula_latex")
            elif entity_type == "parameters":
                metadata["symbol"] = entity.get("symbol")
                metadata["meaning"] = entity.get("meaning")
                metadata["scope"] = entity.get("scope")
            elif entity_type == "proofs":
                metadata["proves_label"] = entity.get("proves_label")

            # Get line range
            line_range = entity.get("source", {}).get("line_range", {}).get("lines", [])

            try:
                # Validate with DSPy agent
                result = validator(
                    entity_type=entity_type.rstrip("s"),  # "definitions" → "definition"
                    entity_label=label,
                    entity_metadata=metadata,
                    extracted_text=full_text,
                    line_range=line_range,
                )

                # Process result
                is_valid = result.get("is_valid", False)
                confidence = result.get("confidence", "low")
                errors = result.get("validation_errors", [])

                if is_valid and confidence in ["high", "medium"]:
                    results["valid"] += 1
                    type_stats["valid"] += 1
                    logger.info(f"  ✓ {label} ({confidence})")

                elif confidence == "low":
                    results["low_confidence"] += 1
                    type_stats["low_confidence"] += 1
                    logger.warning(f"  ⚠ {label} (low confidence)")

                    if errors:
                        results["errors"].append(
                            {"label": label, "type": entity_type, "errors": errors, "confidence": confidence}
                        )

                else:
                    results["invalid"] += 1
                    type_stats["invalid"] += 1
                    logger.error(f"  ✗ {label} (invalid)")

                    results["errors"].append(
                        {"label": label, "type": entity_type, "errors": errors, "suggestions": result.get("suggestions", "")}
                    )

                results["total_validated"] += 1

            except Exception as e:
                logger.error(f"  ✗ {label}: Validation failed - {e}")
                results["errors"].append({"label": label, "type": entity_type, "errors": [str(e)]})

        results["by_type"][entity_type] = type_stats
        logger.info(
            f"  {entity_type}: {type_stats['valid']} valid, "
            f"{type_stats['invalid']} invalid, "
            f"{type_stats['low_confidence']} low confidence"
        )

    # Summary
    logger.info(f"\n✓ Validation complete:")
    logger.info(f"  Total validated: {results['total_validated']}")
    logger.info(f"  Valid: {results['valid']}")
    logger.info(f"  Invalid: {results['invalid']}")
    logger.info(f"  Low confidence: {results['low_confidence']}")

    return results
