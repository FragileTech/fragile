#!/usr/bin/env python3
"""
Validate a single JSON file from raw_data/, refined_data/, or pipeline_data/.

This script automatically detects the data stage and entity type based on the file path,
then validates against the appropriate Pydantic schema.

The code is the single source of truth: data must conform to the schemas, not vice versa.

Usage as CLI:
    python scripts/validate_json.py <path-to-json-file>

Usage as library:
    from scripts.validate_json import validate_json_file
    report = validate_json_file(Path("my-file.json"))

Examples:
    # Validate raw extraction output
    python scripts/validate_json.py docs/source/.../raw_data/theorems/raw-thm-001.json

    # Validate refined/enriched data
    python scripts/validate_json.py docs/source/.../refined_data/theorems/thm-convergence.json

    # Validate pipeline-ready data
    python scripts/validate_json.py docs/source/.../pipeline_data/axioms/axiom-confining.json
"""

import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ValidationError

# Import enriched types (refined_data)
from fragile.proofs.core.enriched_types import (
    EnrichedAxiom,
    EnrichedDefinition,
    EnrichedObject,
    EnrichedTheorem,
    EquationBox,
    ParameterBox,
    RemarkBox,
)

# Import core/pipeline types (pipeline_data)
from fragile.proofs.core.math_types import Axiom, MathematicalObject, Parameter, TheoremBox
from fragile.proofs.core.proof_system import ProofBox

# Import staging types (raw_data)
from fragile.proofs.staging_types import (
    RawAxiom,
    RawDefinition,
    RawEquation,
    RawParameter,
    RawProof,
    RawRemark,
    RawTheorem,
)


# Type aliases
DataStage = Literal["raw", "refined", "pipeline"]
EntityType = Literal[
    "theorem", "axiom", "definition", "object", "parameter", "proof", "remark", "equation"
]


# =============================================================================
# SCHEMA MAPS
# =============================================================================

# Raw data schemas (staging_types) - direct LLM extraction output
RAW_SCHEMA_MAP: dict[EntityType, type[BaseModel]] = {
    "theorem": RawTheorem,
    "axiom": RawAxiom,
    "definition": RawDefinition,
    "object": RawDefinition,  # Objects defined as definitions in raw stage
    "parameter": RawParameter,
    "proof": RawProof,
    "remark": RawRemark,
    "equation": RawEquation,
}

# Refined data schemas (enriched_types) - partially enriched
REFINED_SCHEMA_MAP: dict[EntityType, type[BaseModel]] = {
    "theorem": EnrichedTheorem,
    "axiom": EnrichedAxiom,
    "definition": EnrichedDefinition,
    "object": EnrichedObject,
    "parameter": ParameterBox,
    "proof": ProofBox,
    "remark": RemarkBox,
    "equation": EquationBox,
}

# Pipeline data schemas (core math_types) - fully validated, execution-ready
PIPELINE_SCHEMA_MAP: dict[EntityType, type[BaseModel]] = {
    "theorem": TheoremBox,
    "axiom": Axiom,
    "definition": Axiom,  # Definitions become axioms in pipeline
    "object": MathematicalObject,
    "parameter": Parameter,
    "proof": ProofBox,
    "remark": RemarkBox,
    "equation": EquationBox,
}


# =============================================================================
# PATH ANALYSIS
# =============================================================================


def detect_data_stage(file_path: Path) -> DataStage | None:
    """Detect the data stage from file path.

    Args:
        file_path: Path to JSON file

    Returns:
        Data stage or None if cannot detect
    """
    path_str = str(file_path)

    if "/raw_data/" in path_str:
        return "raw"
    if "/refined_data/" in path_str:
        return "refined"
    if "/pipeline_data/" in path_str:
        return "pipeline"

    return None


def detect_entity_type(file_path: Path, data: dict, stage: DataStage) -> EntityType | None:
    """Detect entity type from file path and/or label.

    Args:
        file_path: Path to JSON file
        data: JSON data dictionary
        stage: Data stage (raw/refined/pipeline)

    Returns:
        Entity type or None if cannot detect
    """
    # Check parent directory name
    parent_dir = file_path.parent.name

    # Map directory names to entity types
    dir_mapping = {
        "theorems": "theorem",
        "lemmas": "theorem",
        "propositions": "theorem",
        "corollaries": "theorem",
        "axioms": "axiom",
        "definitions": "definition",
        "objects": "object",
        "parameters": "parameter",
        "proofs": "proof",
        "remarks": "remark",
        "equations": "equation",
    }

    if parent_dir in dir_mapping:
        return dir_mapping[parent_dir]

    # Fall back to label prefix detection
    label = data.get("label", "") or data.get("temp_id", "")

    if label:
        if label.startswith(("thm-", "lem-", "prop-", "cor-")):
            return "theorem"
        if label.startswith(("ax-", "axiom-")):
            return "axiom"
        if label.startswith("def-"):
            return "definition"
        if label.startswith("obj-"):
            return "object"
        if label.startswith("param-"):
            return "parameter"
        if label.startswith("proof-"):
            return "proof"
        if label.startswith("remark-"):
            return "remark"
        if label.startswith("eq-"):
            return "equation"
        if stage == "raw" and label.startswith("raw-"):
            # Raw staging IDs
            if "def" in label:
                return "definition"
            if "thm" in label or "lem" in label:
                return "theorem"

    return None


def get_schema_for_entity(
    stage: DataStage, entity_type: EntityType
) -> tuple[type[BaseModel], str]:
    """Get the appropriate Pydantic schema for validation.

    Args:
        stage: Data stage (raw/refined/pipeline)
        entity_type: Entity type

    Returns:
        (schema_class, schema_name) tuple

    Raises:
        ValueError: If no schema mapping exists
    """
    if stage == "raw":
        schema_map = RAW_SCHEMA_MAP
        stage_name = "Raw"
    elif stage == "refined":
        schema_map = REFINED_SCHEMA_MAP
        stage_name = "Refined"
    else:  # pipeline
        schema_map = PIPELINE_SCHEMA_MAP
        stage_name = "Pipeline"

    schema_class = schema_map.get(entity_type)
    if not schema_class:
        raise ValueError(f"No {stage_name} schema mapping for entity type '{entity_type}'")

    return schema_class, schema_class.__name__


def _should_show_error_input(error_type: str) -> bool:
    """Determine if we should show input value for this error type.

    Only show input when it provides actionable insight, not for missing fields.

    Args:
        error_type: Pydantic error type string

    Returns:
        True if input should be displayed for this error type
    """
    # Type mismatches - actionable (shows what type was provided)
    if error_type in {
        "string_type",
        "int_type",
        "int_parsing",
        "float_type",
        "float_parsing",
        "bool_type",
        "bool_parsing",
        "dict_type",
        "list_type",
        "tuple_type",
        "set_type",
    }:
        return True

    # Pattern/format violations - actionable (shows what failed validation)
    if error_type in {
        "string_pattern_mismatch",
        "string_too_short",
        "string_too_long",
        "value_error",
        "literal_error",
        "enum",
    }:
        return True

    # Missing fields - NOT actionable (input is just the parent dict)
    if error_type == "missing":
        return False

    # Default: don't show (reduces noise)
    return False


def format_validation_report(
    file_path: Path,
    stage: DataStage,
    entity_type: EntityType,
    schema_name: str,
    validated: BaseModel | None = None,
    validation_error: ValidationError | None = None,
    full_data: dict | None = None,
) -> str:
    """Format validation report as a string.

    Args:
        file_path: Path to JSON file
        stage: Data stage
        entity_type: Entity type
        schema_name: Schema class name
        validated: Validated Pydantic model (if successful)
        validation_error: ValidationError (if failed)
        full_data: Full JSON data (shown once at top if validation fails)

    Returns:
        Formatted report string
    """
    lines = []

    # Header
    lines.extend((
        f"📄 File: {file_path.name}",
        f"📂 Stage: {stage}",
        f"🏷️  Type: {entity_type}",
        f"📋 Schema: {schema_name}",
        "",
    ))

    if validated:
        # Success case
        label = getattr(validated, "label", None) or getattr(validated, "temp_id", None)
        lines.append(f"✅ VALID - {schema_name}")
        if label:
            lines.append(f"   Label: {label}")

        # Print additional info
        if hasattr(validated, "statement_type"):
            lines.append(f"   Statement Type: {validated.statement_type}")
        if hasattr(validated, "object_type"):
            lines.append(f"   Object Type: {validated.object_type}")

        lines.extend(("", "🎉 Validation successful!"))

    elif validation_error:
        # Failure case
        lines.extend((f"❌ VALIDATION FAILED - {schema_name}", ""))

        # Show full JSON input once at the top
        if full_data:
            lines.extend((
                "Full JSON input:",
                "```json",
                json.dumps(full_data, indent=2),
                "```",
                "",
            ))

        lines.append("Errors:")

        for i, error in enumerate(validation_error.errors(), 1):
            loc = " → ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_type = error["type"]

            lines.extend((
                f"\n  {i}. Field: {loc}",
                f"     Error: {msg}",
                f"     Type: {error_type}",
            ))

            # Only show input for actionable error types
            if _should_show_error_input(error_type) and "input" in error:
                input_val = error["input"]
                if input_val is not None:
                    str_val = str(input_val)
                    # Show full value if short, truncate if long
                    if len(str_val) < 100:
                        lines.append(f"     Input: {str_val}")
                    else:
                        lines.append(f"     Input: {str_val[:100]}... (truncated)")

        lines.extend(("", f"💥 Found {len(validation_error.errors())} validation error(s)"))

    return "\n".join(lines)


# =============================================================================
# VALIDATION
# =============================================================================


def validate_json_file(file_path: Path, print_report: bool = False) -> str:
    """Validate a single JSON file and return a report.

    The code is the single source of truth: data MUST conform to schemas.
    If validation fails, the data is incorrect, not the schema.

    Args:
        file_path: Path to JSON file
        print_report: If True, print the report to stdout

    Returns:
        Formatted validation report string

    Raises:
        SystemExit: If validation fails (only when print_report=True)
    """
    # Check file exists
    if not file_path.exists():
        report = f"❌ File not found: {file_path}"
        if print_report:
            print(report)
            sys.exit(1)
        return report

    # Load JSON data
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        report = f"❌ Invalid JSON: {e}"
        if print_report:
            print(report)
            sys.exit(1)
        return report

    # Detect data stage
    stage = detect_data_stage(file_path)
    if not stage:
        report = (
            f"❌ Cannot detect data stage from path: {file_path}\n"
            "   Expected path to contain: /raw_data/, /refined_data/, or /pipeline_data/"
        )
        if print_report:
            print(report)
            sys.exit(1)
        return report

    # Check for invalid raw_data/objects/ directory (should not exist)
    if stage == "raw" and "/raw_data/objects/" in str(file_path):
        report = (
            f"❌ INVALID LOCATION: raw_data/objects/ should not exist\n"
            f"   File: {file_path}\n"
            "\n"
            "   The raw_data/ stage uses staging types (RawDefinition, RawTheorem, etc.)\n"
            "   that capture both concepts and instances without semantic distinction.\n"
            "\n"
            "   Objects (obj-*) are created during Stage 2 refinement through semantic\n"
            "   analysis. They should only exist in:\n"
            "   - refined_data/objects/ (semantic enrichment)\n"
            "   - pipeline_data/objects/ (execution-ready)\n"
            "\n"
            "   This file belongs in refined_data/objects/ or pipeline_data/objects/,\n"
            "   not raw_data/objects/."
        )
        if print_report:
            print(report)
            sys.exit(1)
        return report

    # Detect entity type
    entity_type = detect_entity_type(file_path, data, stage)
    if not entity_type:
        label = data.get("label") or data.get("temp_id", "MISSING")
        report = (
            f"❌ Cannot detect entity type from path or label\n"
            f"   File: {file_path}\n"
            f"   Label/ID: {label}"
        )
        if print_report:
            print(report)
            sys.exit(1)
        return report

    # Get appropriate schema
    try:
        schema_class, schema_name = get_schema_for_entity(stage, entity_type)
    except ValueError as e:
        report = f"❌ {e}"
        if print_report:
            print(report)
            sys.exit(1)
        return report

    # Validate against schema
    try:
        validated = schema_class.model_validate(data)
        report = format_validation_report(
            file_path, stage, entity_type, schema_name, validated=validated
        )

        if print_report:
            print(report)

        return report

    except ValidationError as e:
        report = format_validation_report(
            file_path, stage, entity_type, schema_name, validation_error=e, full_data=data
        )

        if print_report:
            print(report)
            sys.exit(1)

        return report

    except Exception as e:
        report = f"❌ Unexpected error during validation:\n   {type(e).__name__}: {e}"

        if print_report:
            print(report)
            sys.exit(1)

        return report


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main CLI entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_json.py <path-to-json-file>")
        print()
        print("Examples:")
        print("  # Validate raw extraction")
        print(
            "  python scripts/validate_json.py docs/source/.../raw_data/theorems/raw-thm-001.json"
        )
        print()
        print("  # Validate refined data")
        print(
            "  python scripts/validate_json.py docs/source/.../refined_data/theorems/thm-convergence.json"
        )
        print()
        print("  # Validate pipeline data")
        print(
            "  python scripts/validate_json.py docs/source/.../pipeline_data/axioms/ax-confining.json"
        )
        sys.exit(1)

    file_path = Path(sys.argv[1])
    validate_json_file(file_path, print_report=True)


if __name__ == "__main__":
    main()
