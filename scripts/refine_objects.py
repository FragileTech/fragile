#!/usr/bin/env python3
"""
Refine all object entities in the Fragile Gas Framework.

This script:
1. Reads all JSON files in the objects directory
2. Validates each object against the MathematicalObject schema
3. Applies corrections and enrichments
4. Generates a comprehensive validation report
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fragile.proofs.core.math_types import MathematicalObject, ObjectType


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def save_json_file(file_path: Path, data: dict[str, Any]) -> None:
    """Save a JSON file with pretty formatting."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def detect_object_type(data: dict[str, Any]) -> ObjectType:
    """Infer object type from the data."""
    # Check explicit object_type field
    if "object_type" in data:
        type_str = data["object_type"].lower()
        try:
            return ObjectType(type_str)
        except ValueError:
            pass

    # Check applies_to_object_type field (from definitions)
    if "applies_to_object_type" in data:
        type_str = data["applies_to_object_type"].lower()
        try:
            return ObjectType(type_str)
        except ValueError:
            pass

    # Infer from name/description
    name_lower = data.get("name", "").lower()
    data.get("description", "").lower()
    data.get("mathematical_expression", "")

    # Check for operator keywords
    if any(word in name_lower for word in ["operator", "update", "aggregator", "projection"]):
        return ObjectType.OPERATOR

    # Check for measure keywords
    if any(word in name_lower for word in ["measure", "distribution", "probability"]):
        if "distribution" in name_lower:
            return ObjectType.DISTRIBUTION
        return ObjectType.MEASURE

    # Check for function keywords
    if any(word in name_lower for word in ["function", "map", "mapping"]):
        return ObjectType.FUNCTION

    # Check for space keywords
    if any(word in name_lower for word in ["space", "domain"]):
        return ObjectType.SPACE

    # Check for structure keywords
    if any(word in name_lower for word in ["structure", "system", "algorithm", "instantiation"]):
        return ObjectType.STRUCTURE

    # Check for constants/coefficients (treat as structure)
    if any(word in name_lower for word in ["constant", "coefficient", "bound", "parameter"]):
        return ObjectType.STRUCTURE

    # Default to set
    return ObjectType.SET


def normalize_object_data(
    data: dict[str, Any], file_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """
    Normalize object data to match MathematicalObject schema.

    Returns:
        Tuple of (normalized_data, corrections_applied)
    """
    corrections = []
    normalized = {}

    # Extract label
    if "label" in data:
        label = data["label"]
        # Fix label format if needed
        if not label.startswith("obj-"):
            if label.startswith("def-"):
                # This is a definition, convert to object label
                label = label.replace("def-", "obj-", 1)
                corrections.append(f"Converted definition label to object label: {label}")
            elif label.startswith("raw-obj-"):
                # This is a raw object, needs proper label
                corrections.append(f"Raw object needs proper label (currently: {label})")
        normalized["label"] = label
    else:
        # Generate label from filename
        label = file_path.stem
        if not label.startswith("obj-"):
            label = f"obj-{label}"
        normalized["label"] = label
        corrections.append(f"Generated label from filename: {label}")

    # Extract name
    if "name" in data:
        normalized["name"] = data["name"]
    elif "term" in data:
        normalized["name"] = data["term"]
        corrections.append("Used 'term' field as name")
    else:
        corrections.append("ERROR: Missing name field")
        normalized["name"] = "MISSING NAME"

    # Extract mathematical expression
    if "mathematical_expression" in data:
        normalized["mathematical_expression"] = data["mathematical_expression"]
    elif "formal_statement" in data:
        # For definitions, use formal_statement as expression
        normalized["mathematical_expression"] = data["formal_statement"]
        corrections.append("Used 'formal_statement' as mathematical_expression")
    elif "description" in data:
        # Use description if no better option
        normalized["mathematical_expression"] = data["description"]
        corrections.append("Used 'description' as mathematical_expression")
    else:
        corrections.append("ERROR: Missing mathematical_expression")
        normalized["mathematical_expression"] = "MISSING EXPRESSION"

    # Detect object type
    normalized["object_type"] = detect_object_type(data)
    if "object_type" not in data and "applies_to_object_type" not in data:
        corrections.append(f"Inferred object_type: {normalized['object_type'].value}")

    # Extract attributes (currently empty for raw objects)
    normalized["current_attributes"] = data.get("current_attributes", [])
    normalized["attribute_history"] = data.get("attribute_history", [])

    # Extract tags
    if "tags" in data:
        normalized["tags"] = data["tags"]
    else:
        normalized["tags"] = []
        corrections.append("No tags found, using empty list")

    # Extract source information
    normalized["source"] = data.get("source")
    normalized["chapter"] = data.get("chapter")
    normalized["document"] = data.get("document")

    # Extract definition label
    if "definition_label" in data:
        normalized["definition_label"] = data["definition_label"]
    elif normalized["label"].startswith("obj-"):
        # Generate definition label from object label
        normalized["definition_label"] = normalized["label"].replace("obj-", "def-", 1)

    return normalized, corrections


def validate_object(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate object data against MathematicalObject schema.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        MathematicalObject(**data)
        return True, []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            errors.append(f"  - {field}: {message}")
        return False, errors


def process_objects_directory(objects_dir: Path) -> dict[str, Any]:
    """
    Process all object files in the directory.

    Returns:
        Dictionary with processing results
    """
    results = {
        "total_files": 0,
        "valid_objects": 0,
        "invalid_objects": 0,
        "corrected_objects": 0,
        "objects": [],
    }

    # Get all JSON files
    json_files = sorted(objects_dir.glob("*.json"))
    results["total_files"] = len(json_files)

    print(f"\n{'=' * 80}")
    print("OBJECT REFINEMENT REPORT")
    print(f"{'=' * 80}")
    print(f"Directory: {objects_dir}")
    print(f"Total files: {len(json_files)}")
    print(f"{'=' * 80}\n")

    for file_path in json_files:
        print(f"\nProcessing: {file_path.name}")
        print(f"{'-' * 80}")

        # Load original data
        try:
            original_data = load_json_file(file_path)
        except Exception as e:
            print(f"ERROR: Failed to load JSON: {e}")
            results["objects"].append({
                "file": file_path.name,
                "status": "error",
                "error": f"Failed to load JSON: {e}",
            })
            results["invalid_objects"] += 1
            continue

        # Normalize data
        normalized_data, corrections = normalize_object_data(original_data, file_path)

        # Validate
        is_valid, validation_errors = validate_object(normalized_data)

        # Report results
        obj_result = {
            "file": file_path.name,
            "label": normalized_data.get("label", "UNKNOWN"),
            "name": normalized_data.get("name", "UNKNOWN"),
            "object_type": normalized_data.get("object_type", "UNKNOWN"),
            "corrections": corrections,
            "validation_errors": validation_errors,
            "status": "valid" if is_valid else "invalid",
        }

        if corrections:
            results["corrected_objects"] += 1
            print("Corrections applied:")
            for correction in corrections:
                print(f"  - {correction}")

        if is_valid:
            results["valid_objects"] += 1
            print("Status: VALID")
            if corrections:
                print(f"Note: Object is valid after {len(corrections)} correction(s)")
        else:
            results["invalid_objects"] += 1
            print("Status: INVALID")
            print("Validation errors:")
            for error in validation_errors:
                print(error)

        results["objects"].append(obj_result)

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print summary report."""
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files processed: {results['total_files']}")
    print(f"Valid objects: {results['valid_objects']}")
    print(f"Invalid objects: {results['invalid_objects']}")
    print(f"Objects with corrections: {results['corrected_objects']}")
    print(f"{'=' * 80}\n")

    # Group by object type
    type_counts = {}
    for obj in results["objects"]:
        obj_type = obj.get("object_type", "UNKNOWN")
        if isinstance(obj_type, ObjectType):
            obj_type = obj_type.value
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

    print("Objects by type:")
    for obj_type, count in sorted(type_counts.items()):
        print(f"  {obj_type:20} : {count:3}")
    print()

    # List invalid objects
    if results["invalid_objects"] > 0:
        print("\nInvalid objects:")
        for obj in results["objects"]:
            if obj["status"] == "invalid":
                print(f"  - {obj['file']} ({obj['label']})")
                for error in obj.get("validation_errors", []):
                    print(f"    {error}")

    # List objects needing manual review
    needs_review = [
        obj
        for obj in results["objects"]
        if any(
            "ERROR" in c or "MISSING" in str(obj.get("name", ""))
            for c in obj.get("corrections", [])
        )
    ]

    if needs_review:
        print(f"\n{'=' * 80}")
        print("Objects needing manual review:")
        print(f"{'=' * 80}")
        for obj in needs_review:
            print(f"\n{obj['file']} ({obj['label']}):")
            for correction in obj.get("corrections", []):
                if "ERROR" in correction or "MISSING" in correction:
                    print(f"  - {correction}")


def main():
    """Main entry point."""
    objects_dir = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/objects"
    )

    if not objects_dir.exists():
        print(f"ERROR: Directory not found: {objects_dir}")
        return

    results = process_objects_directory(objects_dir)
    print_summary(results)

    # Save results to JSON
    report_path = objects_dir.parent / "object_refinement_report.json"
    save_json_file(report_path, results)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
