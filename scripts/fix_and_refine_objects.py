#!/usr/bin/env python3
"""
Fix and refine all object entities in the Fragile Gas Framework.

This script:
1. Reads all JSON files in the objects directory
2. Fixes structural issues (source, missing fields, etc.)
3. Validates and enriches each object
4. Writes corrected files back
5. Generates a comprehensive validation report
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
        # Handle custom types by mapping to standard ones
        type_mapping = {
            "constant_collection": "structure",
            "constant": "structure",
            "algorithm": "structure",
            "instantiation": "structure",
        }
        type_str = type_mapping.get(type_str, type_str)
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


def extract_mathematical_expression(data: dict[str, Any]) -> str:
    """Extract or construct mathematical expression from data."""
    # Priority 1: explicit mathematical_expression
    if data.get("mathematical_expression"):
        return data["mathematical_expression"]

    # Priority 2: formal_statement (for definitions)
    if data.get("formal_statement"):
        return data["formal_statement"]

    # Priority 3: statement field
    if data.get("statement"):
        return data["statement"]

    # Priority 4: description
    if data.get("description"):
        return data["description"]

    # Priority 5: definition field
    if data.get("definition"):
        return data["definition"]

    # Priority 6: Try to construct from components
    if "components" in data:
        components = data["components"]
        if isinstance(components, dict):
            # Try signature first
            if "signature" in components:
                return components["signature"]
            # Try combining available info
            parts = []
            if "symbol" in components:
                parts.append(components["symbol"])
            if "specification" in components:
                parts.append(components["specification"])
            if parts:
                return " : ".join(parts)

    # Priority 7: Construct from name
    name = data.get("name", "")
    if name:
        return f"Mathematical object: {name}"

    return "MISSING EXPRESSION - NEEDS MANUAL REVIEW"


def fix_source_location(data: dict[str, Any], file_path: Path) -> tuple[dict[str, Any], list[str]]:
    """
    Fix source location to match SourceLocation schema.

    Returns:
        Tuple of (fixed_source_or_None, corrections_applied)
    """
    corrections = []
    source = data.get("source")

    if source is None:
        return None, []

    if isinstance(source, str):
        # Source is just a string, convert to proper structure
        chapter = data.get("chapter", "1_euclidean_gas")
        document = data.get("document", "01_fragile_gas_framework")

        fixed_source = {
            "document_id": document,
            "file_path": f"docs/source/{chapter}/{document}.md",
            "section": source,
        }
        corrections.append("Converted string source to SourceLocation structure")
        return fixed_source, corrections

    if isinstance(source, dict):
        # Check required fields
        if "document_id" not in source:
            document = data.get("document", "01_fragile_gas_framework")
            source["document_id"] = document
            corrections.append(f"Added missing document_id: {document}")

        if "file_path" not in source:
            chapter = data.get("chapter", "1_euclidean_gas")
            document = source.get("document_id", data.get("document", "01_fragile_gas_framework"))
            # Handle case where document field is used instead of document_id
            if "document" in source and "document_id" not in source:
                document = source["document"]
                source["document_id"] = document
                corrections.append("Used 'document' field as document_id")

            file_path_str = f"docs/source/{chapter}/{document}.md"
            source["file_path"] = file_path_str
            corrections.append(f"Added missing file_path: {file_path_str}")

        # Convert line_number to line_range if present
        if "line_number" in source and "line_range" not in source:
            line_num = source["line_number"]
            source["line_range"] = (line_num, line_num + 10)  # Approximate 10-line span
            del source["line_number"]
            corrections.append("Converted line_number to line_range")

        # Remove invalid fields
        invalid_fields = ["subsection"]
        for field in invalid_fields:
            if field in source:
                del source[field]
                corrections.append(f"Removed invalid field: {field}")

        return source, corrections

    return None, []


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
                # This is a raw object, generate proper label
                proper_label = label.replace("raw-obj-", "obj-swarm-update-operator-", 1)
                corrections.append(f"Converted raw label {label} to {proper_label}")
                label = proper_label
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
    normalized["mathematical_expression"] = extract_mathematical_expression(data)
    if normalized["mathematical_expression"].startswith("MISSING"):
        corrections.append("ERROR: Could not extract mathematical_expression")
    elif "mathematical_expression" not in data:
        corrections.append("Constructed mathematical_expression from other fields")

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
        if "tags" not in data:  # Only note if truly missing
            corrections.append("No tags found, using empty list")

    # Fix source location
    fixed_source, source_corrections = fix_source_location(data, file_path)
    normalized["source"] = fixed_source
    corrections.extend(source_corrections)

    # Extract chapter and document
    normalized["chapter"] = data.get("chapter", "1_euclidean_gas")
    normalized["document"] = data.get("document", "01_fragile_gas_framework")

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


def process_and_fix_objects(objects_dir: Path, write_back: bool = False) -> dict[str, Any]:
    """
    Process all object files, fix issues, and optionally write back.

    Args:
        objects_dir: Directory containing object JSON files
        write_back: If True, write corrected files back to disk

    Returns:
        Dictionary with processing results
    """
    results = {
        "total_files": 0,
        "valid_objects": 0,
        "fixed_objects": 0,
        "invalid_objects": 0,
        "objects": [],
    }

    # Get all JSON files
    json_files = sorted(objects_dir.glob("*.json"))
    results["total_files"] = len(json_files)

    print(f"\n{'=' * 80}")
    print("OBJECT FIX AND REFINEMENT REPORT")
    print(f"{'=' * 80}")
    print(f"Directory: {objects_dir}")
    print(f"Total files: {len(json_files)}")
    print(f"Write back: {write_back}")
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
            print("Corrections applied:")
            for correction in corrections:
                print(f"  - {correction}")

        if is_valid:
            results["valid_objects"] += 1
            if corrections:
                results["fixed_objects"] += 1
            print("Status: VALID")
            if corrections:
                print(f"Note: Object is valid after {len(corrections)} correction(s)")

            # Write back if requested
            if write_back and corrections:
                try:
                    save_json_file(file_path, normalized_data)
                    print("  ✓ Wrote corrected file back to disk")
                except Exception as e:
                    print(f"  ✗ Failed to write file: {e}")
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
    print(f"Fixed objects: {results['fixed_objects']}")
    print(f"Invalid objects: {results['invalid_objects']}")
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
                for error in obj.get("validation_errors", [])[:3]:  # Show first 3 errors
                    print(f"    {error}")

    # List objects with significant corrections
    significant_fixes = [
        obj
        for obj in results["objects"]
        if len(obj.get("corrections", [])) >= 3 and obj["status"] == "valid"
    ]

    if significant_fixes:
        print(f"\n{'=' * 80}")
        print(f"Objects with significant corrections ({len(significant_fixes)}):")
        print(f"{'=' * 80}")
        for obj in significant_fixes[:10]:  # Show first 10
            print(f"\n{obj['file']} ({obj['label']}):")
            for correction in obj.get("corrections", [])[:5]:
                print(f"  - {correction}")


def main():
    """Main entry point."""
    import sys

    objects_dir = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/objects"
    )

    if not objects_dir.exists():
        print(f"ERROR: Directory not found: {objects_dir}")
        return

    # Check for --write-back flag
    write_back = "--write-back" in sys.argv or "-w" in sys.argv

    if write_back:
        print("\n⚠️  WARNING: Files will be modified in place!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return

    results = process_and_fix_objects(objects_dir, write_back=write_back)
    print_summary(results)

    # Save results to JSON
    report_path = objects_dir.parent / "object_fix_report.json"
    save_json_file(report_path, results)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
