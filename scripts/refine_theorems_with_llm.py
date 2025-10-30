#!/usr/bin/env python3
"""
Enhanced Theorem Refinement Script - Stage 2 Enrichment with LLM

Uses Gemini 2.5 Pro to enrich theorems with:
- Decomposed assumptions and conclusions
- Inferred input objects/axioms/parameters
- Better output type classification
- Framework consistency validation
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


def create_gemini_enrichment_prompt(raw_data: dict[str, Any], label: str) -> str:
    """
    Create a structured prompt for Gemini to analyze and enrich a theorem.
    """
    statement = (
        raw_data.get("statement")
        or raw_data.get("full_statement_text")
        or raw_data.get("natural_language_statement")
        or ""
    )

    formal_statement = raw_data.get("formal_statement", "")

    return f"""Analyze this mathematical theorem and extract structured semantic information.

THEOREM: {label}
NAME: {raw_data.get("name", raw_data.get("title", "Unknown"))}

STATEMENT:
{statement}

{f"FORMAL STATEMENT:{formal_statement}" if formal_statement else ""}

Please provide a JSON response with the following structure:

{{
  "assumptions": [
    "assumption 1 (e.g., 'Let v > 0')",
    "assumption 2 (e.g., 'Assume U is Lipschitz')"
  ],
  "conclusion": "The main conclusion (e.g., 'the system converges exponentially')",
  "output_type": "One of: Property, Relation, Existence, Construction, Classification, Uniqueness, Impossibility, Embedding, Approximation, Equivalence, Decomposition, Extension, Reduction, Bound, Convergence, Contraction",
  "input_objects": [
    "obj-object1",
    "obj-object2"
  ],
  "input_axioms": [
    "axiom-axiom1"
  ],
  "input_parameters": [
    "param-param1"
  ],
  "attributes_required": {{
    "obj-object1": ["attr-lipschitz", "attr-bounded"],
    "obj-object2": ["attr-continuous"]
  }},
  "uses_definitions": [
    "def-definition1"
  ]
}}

IMPORTANT RULES:
1. Extract ALL explicit assumptions from the statement (Look for "Let", "Assume", "Suppose", "Given", etc.)
2. Extract the main conclusion (what the theorem proves/establishes)
3. Classify the output_type based on what the theorem accomplishes
4. For input_objects: Include mathematical objects that appear in the statement (use "obj-" prefix)
5. For input_axioms: Include any axioms the theorem explicitly relies on (use "axiom-" prefix)
6. For input_parameters: Include any parameters/constants (use "param-" prefix)
7. For attributes_required: Map each input object to the properties it must have
8. For uses_definitions: Include definitions needed to understand the statement (use "def-" prefix)

Only return the JSON, no additional text.
"""


def enrich_with_gemini(
    raw_data: dict[str, Any], label: str, use_gemini: bool = True
) -> dict[str, Any] | None:
    """
    Use Gemini 2.5 Pro to enrich theorem with semantic analysis.

    Returns enriched fields or None if LLM call fails.
    """
    if not use_gemini:
        return None

    try:
        # Create prompt
        create_gemini_enrichment_prompt(raw_data, label)

        # Note: In actual implementation, you would call the Gemini MCP here
        # For now, return None to indicate LLM enrichment should be done separately
        # This is where you'd use: mcp__gemini-cli__ask-gemini

        return None

    except Exception as e:
        print(f"  ! Gemini enrichment failed for {label}: {e}")
        return None


def normalize_output_type(raw_type: str) -> str:
    """Normalize output type to match TheoremOutputType enum."""
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
    return type_mapping.get(normalized, "Property")


def enrich_theorem(raw_data: dict[str, Any], use_gemini: bool = False) -> dict[str, Any]:
    """
    Enrich a raw theorem with semantic information.
    """
    enriched = {}

    # Extract label
    label = (
        raw_data.get("label")
        or raw_data.get("theorem_id")
        or raw_data.get("label_text")
        or raw_data.get("temp_id", "")
    )

    if label.startswith("raw-"):
        label = label.replace("raw-", "", 1)

    if not label.startswith(("thm-", "lem-", "prop-")):
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

    formal_statement = raw_data.get("formal_statement", "")
    if formal_statement and formal_statement != statement:
        enriched["natural_language_statement"] = f"{statement}\n\n{formal_statement}"
    else:
        enriched["natural_language_statement"] = statement

    # Try LLM enrichment
    llm_enrichment = enrich_with_gemini(raw_data, label, use_gemini)

    if llm_enrichment:
        # Use LLM-enriched fields
        enriched.update(llm_enrichment)
    else:
        # Fall back to existing data
        enriched["output_type"] = normalize_output_type(raw_data.get("output_type", "Property"))
        enriched["input_objects"] = raw_data.get("input_objects", [])
        enriched["input_axioms"] = raw_data.get("input_axioms", [])
        enriched["input_parameters"] = raw_data.get("input_parameters", [])

    # Proof status
    proof_status = raw_data.get("proof_status", "unproven")
    if proof_status in {"complete", "expanded", "verified"}:
        enriched["proof_status"] = "expanded"
    elif proof_status == "sketched":
        enriched["proof_status"] = "sketched"
    else:
        enriched["proof_status"] = "unproven"

    # Preserve raw data
    enriched["raw_fallback"] = raw_data

    # Store dependencies for relationship building
    dependencies = raw_data.get("dependencies", [])
    if dependencies:
        enriched["_raw_dependencies"] = dependencies

    used_in = raw_data.get("used_in", [])
    if used_in:
        enriched["_raw_used_in"] = used_in

    tags = raw_data.get("tags", [])
    if tags:
        enriched["_tags"] = tags

    return enriched


def validate_theorem(enriched_data: dict[str, Any]) -> tuple[TheoremBox | None, list[str]]:
    """Validate enriched theorem data against TheoremBox schema."""
    errors = []

    try:
        theorem = TheoremBox.model_validate(enriched_data)
        return theorem, []
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            errors.append(f"{field}: {message}")

        try:
            minimal_data = {
                "label": enriched_data.get("label", "thm-unknown"),
                "name": enriched_data.get("name", "Unknown Theorem"),
                "output_type": enriched_data.get("output_type", "Property"),
                "validation_errors": errors,
                "raw_fallback": enriched_data.get("raw_fallback"),
            }

            for field in [
                "natural_language_statement",
                "input_objects",
                "input_axioms",
                "input_parameters",
                "proof_status",
                "attributes_required",
            ]:
                if field in enriched_data:
                    minimal_data[field] = enriched_data[field]

            theorem = TheoremBox.model_validate(minimal_data)
            return theorem, errors
        except ValidationError:
            return None, errors


def process_theorem_file(
    input_path: Path, output_dir: Path, stats: dict[str, Any], use_gemini: bool = False
) -> None:
    """Process a single theorem file."""
    try:
        raw_data = load_json_file(input_path)
        enriched_data = enrich_theorem(raw_data, use_gemini)
        theorem, errors = validate_theorem(enriched_data)

        if theorem:
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
    import sys

    start_time = time.time()
    use_gemini = "--gemini" in sys.argv

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
        "mode": "full" if use_gemini else "quick",
        "success": 0,
        "partial_success": 0,
        "failed": 0,
        "validation_errors": [],
        "timestamp": datetime.now().isoformat(),
    }

    theorem_files = sorted(raw_dir.glob("*.json"))

    print("Document Refiner - Stage 2: Semantic Enrichment")
    print(f"  Source: {raw_dir}")
    print(f"  Mode: {'full (with Gemini 2.5 Pro)' if use_gemini else 'quick'}")
    print()

    if use_gemini:
        print("NOTE: Gemini enrichment requires manual MCP tool invocation.")
        print("      Run with --gemini flag when MCP is available.")
        print()

    print("Phase 2.1: Loading raw data...")
    print(f"  Found {len(theorem_files)} theorem files")
    print()

    print("Phase 2.4: Enriching theorems → TheoremBox...")

    for theorem_file in theorem_files:
        process_theorem_file(theorem_file, refined_dir, stats, use_gemini)

    total = stats["success"] + stats["partial_success"] + stats["failed"]
    stats["total_processed"] = total
    stats["enrichment_time_seconds"] = round(time.time() - start_time, 2)

    print(f"  Processed {total} theorems")
    print(f"    Success: {stats['success']}")
    print(f"    Partial Success: {stats['partial_success']}")
    print(f"    Failed: {stats['failed']}")
    print()

    print("Phase 2.10: Exporting statistics...")
    stats_file = stats_dir / "theorem_refinement_statistics.json"
    save_json_file(stats_file, stats)
    print(f"  Statistics: {stats_file}")

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

    print("Semantic enrichment complete!")
    print(f"  Output: {refined_dir}")
    print(f"  Reports: {stats_dir}")
    print(f"  Time: {stats['enrichment_time_seconds']} seconds")
    print(f"  Success Rate: {100 * stats['success'] / total:.1f}%")

    if stats["validation_errors"]:
        print()
        print("Validation Errors Summary:")
        for error_entry in stats["validation_errors"][:5]:
            print(f"  - {error_entry['file']} ({error_entry['label']})")
            for err in error_entry["errors"][:2]:
                print(f"      {err}")

        if len(stats["validation_errors"]) > 5:
            print(f"  ... and {len(stats['validation_errors']) - 5} more")


if __name__ == "__main__":
    main()
