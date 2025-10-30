#!/usr/bin/env python3
"""
Refine all lemma entities in Fragile Gas Framework.

This script validates and corrects all lemma JSON files according to framework conventions.
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any


LEMMAS_DIR = Path(
    "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/lemmas"
)


class LemmaRefiner:
    def __init__(self):
        self.corrections = defaultdict(list)
        self.validation_errors = defaultdict(list)
        self.lemmas_processed = []

    def load_lemma(self, filepath: Path) -> dict[str, Any]:
        """Load lemma JSON file."""
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def save_lemma(self, filepath: Path, data: dict[str, Any]):
        """Save lemma JSON file with formatting."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def validate_label(self, label: str) -> tuple[bool, str]:
        """Validate label follows framework convention (lem-* or sub-lem-*)."""
        if not label:
            return False, "Label is empty"
        if label.startswith(("lem-", "sub-lem-")):
            return True, "Valid"
        if label.startswith("unlabeled-"):
            return False, "Unlabeled lemma needs proper label"
        return False, "Invalid label format (should be lem-* or sub-lem-*)"

    def normalize_statement_type(self, data: dict[str, Any]) -> str:
        """Normalize statement_type field."""
        # Check both 'statement_type' and 'type' fields
        stmt_type = data.get("statement_type") or data.get("type")

        if stmt_type in {"lemma", "sub-lemma"}:
            return "lemma"

        # Infer from label
        label = data.get("label", "")
        if label.startswith("sub-lem-"):
            return "lemma"
        if label.startswith("lem-"):
            return "lemma"

        return "lemma"  # Default

    def validate_name(self, name: str, label: str) -> tuple[bool, str]:
        """Validate name is descriptive."""
        if not name:
            return False, "Name is missing"
        if name == label:
            return False, "Name is identical to label (should be descriptive)"
        if len(name) < 10:
            return False, "Name too short (should be descriptive)"
        return True, "Valid"

    def validate_output_type(self, output_type: str) -> tuple[bool, str]:
        """Validate output_type is accurate."""
        valid_types = [
            "General Result",
            "Property",
            "Bound",
            "Convergence",
            "Equivalence",
            "Existence",
            "Uniqueness",
            "Lipschitz Continuity",
            "Mean Square Continuity",
            "Regularity",
        ]
        if not output_type:
            return False, "Output type is missing"
        if output_type not in valid_types:
            return False, f"Output type '{output_type}' not in standard types"
        return True, "Valid"

    def extract_name_from_content(self, data: dict[str, Any]) -> str:
        """Extract descriptive name from content or statement."""
        # Try to get from existing name field
        if data.get("name"):
            return data["name"]

        # Try to extract from statement/content
        statement = (
            data.get("natural_language_statement")
            or data.get("statement")
            or data.get("content", "")
        )

        # Take first sentence or first 100 chars
        if statement:
            first_sentence = statement.split(".")[0].strip()
            if len(first_sentence) > 100:
                return first_sentence[:97] + "..."
            return first_sentence

        return ""

    def normalize_lemma_schema(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize lemma to consistent schema."""
        normalized = {}

        # Core fields
        normalized["label"] = data.get("label", "")
        normalized["name"] = data.get("name", "") or self.extract_name_from_content(data)
        normalized["statement_type"] = self.normalize_statement_type(data)

        # Source metadata
        normalized["source"] = data.get("source")
        normalized["chapter"] = data.get("chapter", "1_euclidean_gas")
        normalized["document"] = data.get("document", "01_fragile_gas_framework")

        # Proof information
        normalized["proof"] = data.get("proof")
        normalized["proof_status"] = data.get("proof_status", "expanded")

        # Input dependencies
        normalized["input_objects"] = data.get("input_objects", [])
        normalized["input_axioms"] = data.get("input_axioms", [])
        normalized["input_parameters"] = data.get("input_parameters", [])
        normalized["attributes_required"] = data.get("attributes_required", {})

        # Internal dependencies
        normalized["internal_lemmas"] = data.get("internal_lemmas", [])
        normalized["internal_propositions"] = data.get("internal_propositions", [])
        normalized["lemma_dag_edges"] = data.get("lemma_dag_edges", [])

        # Output information
        normalized["output_type"] = data.get("output_type", "General Result")
        normalized["attributes_added"] = data.get("attributes_added", [])
        normalized["relations_established"] = data.get("relations_established", [])

        # Statement content
        normalized["natural_language_statement"] = (
            data.get("natural_language_statement")
            or data.get("statement")
            or data.get("content", "")
        )

        # Additional fields
        normalized["assumptions"] = data.get("assumptions", [])
        normalized["conclusion"] = data.get("conclusion", "")
        normalized["equation_label"] = data.get("equation_label")
        normalized["uses_definitions"] = data.get("uses_definitions", [])
        normalized["validation_errors"] = data.get("validation_errors", [])
        normalized["raw_fallback"] = data.get("raw_fallback")

        # Preserve legacy fields for reference
        if "start_line" in data:
            normalized["_legacy_start_line"] = data["start_line"]
        if "end_line" in data:
            normalized["_legacy_end_line"] = data["end_line"]
        if "section" in data:
            normalized["_legacy_section"] = data["section"]

        return normalized

    def refine_lemma(self, filepath: Path) -> dict[str, Any]:
        """Refine a single lemma file."""
        filename = filepath.name
        print(f"\nProcessing: {filename}")

        # Load lemma
        data = self.load_lemma(filepath)
        original_data = data.copy()

        # Normalize schema
        data = self.normalize_lemma_schema(data)

        # Validate label
        label = data["label"]
        label_valid, label_msg = self.validate_label(label)
        if not label_valid:
            self.validation_errors[filename].append(f"Label: {label_msg}")
            print(f"  ERROR: {label_msg}")

        # Validate statement_type
        if data["statement_type"] != "lemma":
            self.corrections[filename].append(
                f"statement_type: '{original_data.get('statement_type')}' -> 'lemma'"
            )
            data["statement_type"] = "lemma"
            print("  CORRECTED: statement_type -> 'lemma'")

        # Validate name
        name_valid, name_msg = self.validate_name(data["name"], label)
        if not name_valid:
            self.validation_errors[filename].append(f"Name: {name_msg}")
            print(f"  WARNING: {name_msg}")

        # Validate output_type
        output_type_valid, output_type_msg = self.validate_output_type(data["output_type"])
        if not output_type_valid:
            self.validation_errors[filename].append(f"Output type: {output_type_msg}")
            print(f"  WARNING: {output_type_msg}")

        # Check completeness of input fields
        if not data["input_objects"] and not data["input_axioms"] and not data["input_parameters"]:
            self.validation_errors[filename].append(
                "Input dependencies: No input objects, axioms, or parameters specified"
            )
            print("  WARNING: No input dependencies specified")

        # Check relations_established
        if not data["relations_established"]:
            self.validation_errors[filename].append(
                "Relations: No relations_established specified"
            )
            print("  WARNING: No relations_established")

        # Check for missing statement
        if not data["natural_language_statement"]:
            self.validation_errors[filename].append(
                "Statement: Missing natural language statement"
            )
            print("  ERROR: Missing statement content")

        # Save if changes were made
        if data != original_data:
            self.save_lemma(filepath, data)
            print("  SAVED: Updated lemma file")
        else:
            print("  OK: No changes needed")

        self.lemmas_processed.append({
            "filename": filename,
            "label": label,
            "name": data["name"],
            "output_type": data["output_type"],
            "has_errors": len(self.validation_errors[filename]) > 0,
            "has_corrections": len(self.corrections[filename]) > 0,
        })

        return data

    def generate_report(self) -> str:
        """Generate summary report."""
        lines = []
        lines.extend((
            "=" * 80,
            "LEMMA REFINEMENT REPORT",
            "=" * 80,
            f"\nTotal lemmas processed: {len(self.lemmas_processed)}",
        ))

        # Count by status
        errors_count = sum(1 for l in self.lemmas_processed if l["has_errors"])
        corrections_count = sum(1 for l in self.lemmas_processed if l["has_corrections"])
        clean_count = len(self.lemmas_processed) - errors_count

        lines.append(f"  - Clean (no issues): {clean_count}")
        lines.append(f"  - With corrections: {corrections_count}")
        lines.append(f"  - With validation errors: {errors_count}")

        # List all lemmas
        lines.append("\n" + "-" * 80)
        lines.append("ALL LEMMAS")
        lines.append("-" * 80)
        for lemma in self.lemmas_processed:
            status = "OK"
            if lemma["has_errors"]:
                status = "ERRORS"
            elif lemma["has_corrections"]:
                status = "CORRECTED"

            lines.extend((
                f"\n{lemma['label']}",
                f"  File: {lemma['filename']}",
                f"  Name: {lemma['name'][:80]}",
                f"  Output Type: {lemma['output_type']}",
                f"  Status: {status}",
            ))

        # Corrections summary
        if self.corrections:
            lines.extend(("\n" + "-" * 80, "CORRECTIONS APPLIED", "-" * 80))
            for filename, corr_list in self.corrections.items():
                lines.append(f"\n{filename}:")
                for corr in corr_list:
                    lines.append(f"  - {corr}")

        # Validation errors summary
        if self.validation_errors:
            lines.extend(("\n" + "-" * 80, "VALIDATION ERRORS", "-" * 80))
            for filename, err_list in self.validation_errors.items():
                lines.append(f"\n{filename}:")
                for err in err_list:
                    lines.append(f"  - {err}")

        # Final validation status
        lines.extend(("\n" + "=" * 80, "FINAL VALIDATION STATUS", "=" * 80))
        if errors_count == 0:
            lines.extend(("\nSTATUS: ALL LEMMAS VALID", "All lemmas pass validation checks."))
        else:
            lines.extend((
                f"\nSTATUS: {errors_count} LEMMAS REQUIRE ATTENTION",
                "Some lemmas have validation errors that need manual review.",
            ))

        return "\n".join(lines)

    def run(self):
        """Run refinement on all lemma files."""
        lemma_files = sorted(LEMMAS_DIR.glob("*.json"))

        print(f"Found {len(lemma_files)} lemma files in {LEMMAS_DIR}")

        for filepath in lemma_files:
            self.refine_lemma(filepath)

        # Generate report
        report = self.generate_report()
        print("\n" + report)

        # Save report
        report_path = LEMMAS_DIR.parent / "lemma_refinement_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n\nReport saved to: {report_path}")


if __name__ == "__main__":
    refiner = LemmaRefiner()
    refiner.run()
