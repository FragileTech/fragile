#!/usr/bin/env python3
"""
Theorem Refinement Script - Stage 2 Enrichment

Transforms raw theorem JSON files into validated, enriched TheoremBox instances.
Follows the document-refiner agent protocol.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from pydantic import ValidationError

from fragile.proofs.core.math_types import TheoremBox


def load_json_file(filepath: Path) -> dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def save_json_file(filepath: Path, data: dict[str, Any]) -> None:
    """Save data to a JSON file with pretty formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_output_type(raw_type: str) -> str:
    """
    Normalize output type to match TheoremOutputType enum.

    Common mappings:
    - "General Result" → "Property"
    - "Bound" → "Bound"
    - "Continuity" → "Property"
    - etc.
    """
    # Map common raw types to enum values
    type_mapping = {
        "general result": "Property",
        "bound": "Bound",
        "continuity": "Property",
        "convergence": "Convergence",
        "equivalence": "Equivalence",
        "existence": "Existence",
        "property": "Property",
        "relation": "Relation",
        "construction": "Construction",
        "classification": "Classification",
        "uniqueness": "Uniqueness",
        "impossibility": "Impossibility",
        "embedding": "Embedding",
        "approximation": "Approximation",
        "decomposition": "Decomposition",
        "extension": "Extension",
        "reduction": "Reduction",
        "contraction": "Contraction",
    }

    normalized = raw_type.lower().strip()
    return type_mapping.get(normalized, "Property")  # Default to Property


def enrich_theorem(raw_data: dict[str, Any]) -> dict[str, Any]:
    """
    Enrich a raw theorem with semantic information.

    Transforms various raw formats into standardized TheoremBox format.
    """
    enriched = {}

    # Extract label (handle multiple possible field names)
    label = (
        raw_data.get("label")
        or raw_data.get("theorem_id")
        or raw_data.get("label_text")
        or raw_data.get("temp_id", "")
    )

    # Remove "raw-" prefix if present
    if label.startswith("raw-"):
        label = label.replace("raw-", "", 1)

    # Ensure label has correct prefix
    if not label.startswith(("thm-", "lem-", "prop-")):
        # Try to determine from context
        if "lemma" in str(raw_data.get("statement_type", "")).lower():
            label = f"lem-{label}" if not label.startswith("lem-") else label
        elif "proposition" in str(raw_data.get("statement_type", "")).lower():
            label = f"prop-{label}" if not label.startswith("prop-") else label
        else:
            label = f"thm-{label}" if not label.startswith("thm-") else label

    enriched["label"] = label

    # Extract name
    enriched["name"] = (
        raw_data.get("name")
        or raw_data.get("title")
        or raw_data.get("label_text")
        or label.replace("-", " ").title()
    )

    # Extract statement
    statement = (
        raw_data.get("statement")
        or raw_data.get("full_statement_text")
        or raw_data.get("natural_language_statement")
        or ""
    )

    # If statement is truncated, note it
    if statement and (statement.endswith("...") or len(statement) < 50):
        enriched["validation_errors"] = ["Statement appears truncated or incomplete"]

    # Natural language statement (preserve original)
    enriched["natural_language_statement"] = statement

    # Extract formal statement
    formal_statement = raw_data.get("formal_statement", "")
    if formal_statement and formal_statement != statement:
        # Could combine both
        enriched["natural_language_statement"] = f"{statement}\n\n{formal_statement}"

    # Extract output type
    raw_output_type = raw_data.get("output_type", "Property")
    enriched["output_type"] = normalize_output_type(raw_output_type)

    # Input objects
    enriched["input_objects"] = raw_data.get("input_objects", [])

    # Input axioms
    enriched["input_axioms"] = raw_data.get("input_axioms", [])

    # Input parameters
    enriched["input_parameters"] = raw_data.get("input_parameters", [])

    # Relations established
    relations = raw_data.get("relations_established", [])
    # Filter out empty or invalid relations
    relations = [r for r in relations if r and r.strip()]
    enriched["relations_established"] = []  # Will be populated by relationship builder

    # Dependencies (for future relationship building)
    dependencies = raw_data.get("dependencies", [])
    if dependencies:
        # Store for later relationship inference
        enriched["_raw_dependencies"] = dependencies

    # Used in (for future relationship building)
    used_in = raw_data.get("used_in", [])
    if used_in:
        enriched["_raw_used_in"] = used_in

    # Tags (for metadata)
    tags = raw_data.get("tags", [])
    if tags:
        enriched["_tags"] = tags

    # Proof sketch
    proof_sketch = raw_data.get("proof_sketch", "")
    if proof_sketch:
        enriched["_proof_sketch"] = proof_sketch

    # Proof status
    proof_status = raw_data.get("proof_status", "unproven")
    if proof_status in {"complete", "expanded", "verified"}:
        enriched["proof_status"] = "expanded"
    elif proof_status == "sketched":
        enriched["proof_status"] = "sketched"
    else:
        enriched["proof_status"] = "unproven"

    # Notes/importance
    notes = raw_data.get("notes", "")
    importance = raw_data.get("importance", "")
    if notes or importance:
        enriched["_metadata"] = {
            "notes": notes,
            "importance": importance,
        }

    # Preserve raw data for debugging
    enriched["raw_fallback"] = raw_data

    return enriched


def validate_theorem(enriched_data: dict[str, Any]) -> tuple[TheoremBox | None, list[str]]:
    """
    Validate enriched theorem data against TheoremBox schema.

    Returns:
        (theorem_box, validation_errors)
    """
    errors = []

    try:
        # Try to validate
        theorem = TheoremBox.model_validate(enriched_data)
        return theorem, []
    except ValidationError as e:
        # Collect validation errors
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            errors.append(f"{field}: {message}")

        # Try to create a minimal valid instance with error tracking
        try:
            # Ensure minimum required fields
            minimal_data = {
                "label": enriched_data.get("label", "thm-unknown"),
                "name": enriched_data.get("name", "Unknown Theorem"),
                "output_type": enriched_data.get("output_type", "Property"),
                "validation_errors": errors,
                "raw_fallback": enriched_data.get("raw_fallback"),
            }

            # Add optional fields if present and valid
            for field in [
                "natural_language_statement",
                "input_objects",
                "input_axioms",
                "input_parameters",
                "proof_status",
            ]:
                if field in enriched_data:
                    minimal_data[field] = enriched_data[field]

            theorem = TheoremBox.model_validate(minimal_data)
            return theorem, errors
        except ValidationError:
            # Complete failure
            return None, errors


def process_theorem_file(input_path: Path, output_dir: Path, stats: dict[str, Any]) -> None:
    """Process a single theorem file."""
    try:
        # Load raw data
        raw_data = load_json_file(input_path)

        # Enrich
        enriched_data = enrich_theorem(raw_data)

        # Validate
        theorem, errors = validate_theorem(enriched_data)

        if theorem:
            # Save enriched theorem
            output_path = output_dir / f"{theorem.label}.json"
            save_json_file(output_path, theorem.model_dump(mode="json"))

            if errors:
                stats["partial_success"] += 1
                stats["validation_errors"].append({
                    "file": str(input_path.name),
                    "label": theorem.label,
                    "errors": errors,
                })
            else:
                stats["success"] += 1
        else:
            stats["failed"] += 1
            stats["validation_errors"].append({
                "file": str(input_path.name),
                "label": enriched_data.get("label", "unknown"),
                "errors": errors,
            })

    except Exception as e:
        stats["failed"] += 1
        stats["validation_errors"].append({
            "file": str(input_path.name),
            "label": "unknown",
            "errors": [f"Exception: {e!s}"],
        })


def main():
    """Main refinement workflow."""
    start_time = time.time()

    # Paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/theorems"
    refined_dir = (
        base_dir / "docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/theorems"
    )
    stats_dir = (
        base_dir / "docs/source/1_euclidean_gas/01_fragile_gas_framework/reports/statistics"
    )

    # Initialize statistics
    stats = {
        "source_directory": str(raw_dir),
        "processing_stage": "semantic_enrichment",
        "mode": "quick",  # Not using LLM yet
        "success": 0,
        "partial_success": 0,
        "failed": 0,
        "validation_errors": [],
        "timestamp": datetime.now().isoformat(),
    }

    # Find all theorem files
    theorem_files = sorted(raw_dir.glob("*.json"))

    print("Document Refiner - Stage 2: Semantic Enrichment")
    print(f"  Source: {raw_dir}")
    print("  Mode: quick")
    print()

    print("Phase 2.1: Loading raw data...")
    print(f"  Found {len(theorem_files)} theorem files")
    print()

    print("Phase 2.4: Enriching theorems → TheoremBox...")

    # Process each file
    for theorem_file in theorem_files:
        process_theorem_file(theorem_file, refined_dir, stats)

    # Calculate totals
    total = stats["success"] + stats["partial_success"] + stats["failed"]
    stats["total_processed"] = total
    stats["enrichment_time_seconds"] = round(time.time() - start_time, 2)

    print(f"  Processed {total} theorems")
    print(f"    Success: {stats['success']}")
    print(f"    Partial Success: {stats['partial_success']}")
    print(f"    Failed: {stats['failed']}")
    print()

    # Save statistics
    print("Phase 2.10: Exporting statistics...")
    stats_file = stats_dir / "theorem_refinement_statistics.json"
    save_json_file(stats_file, stats)
    print(f"  Statistics: {stats_file}")

    # Save validation report
    validation_report = {
        "total_errors": len(stats["validation_errors"]),
        "by_severity": {
            "complete_failure": sum(
                1
                for e in stats["validation_errors"]
                if any("Exception" in err for err in e.get("errors", []))
            ),
            "partial_validation": stats["partial_success"],
        },
        "errors": stats["validation_errors"],
    }
    validation_file = stats_dir / "theorem_validation_report.json"
    save_json_file(validation_file, validation_report)
    print(f"  Validation Report: {validation_file}")
    print()

    # Summary
    print("Semantic enrichment complete!")
    print(f"  Output: {refined_dir}")
    print(f"  Reports: {stats_dir}")
    print(f"  Time: {stats['enrichment_time_seconds']} seconds")
    print(f"  Success Rate: {100 * stats['success'] / total:.1f}%")

    if stats["validation_errors"]:
        print()
        print("Validation Errors Summary:")
        for error_entry in stats["validation_errors"][:5]:  # Show first 5
            print(f"  - {error_entry['file']} ({error_entry['label']})")
            for err in error_entry["errors"][:2]:  # Show first 2 errors
                print(f"      {err}")

        if len(stats["validation_errors"]) > 5:
            print(f"  ... and {len(stats['validation_errors']) - 5} more")


if __name__ == "__main__":
    main()
