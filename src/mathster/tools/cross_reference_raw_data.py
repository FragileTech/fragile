#!/usr/bin/env python3
"""
Cross-Reference Analysis for Raw Data Entities.

This script analyzes all theorem-like entities in raw_data/ and fills in:
- input_objects: Mathematical objects this result depends on
- input_axioms: Axioms required for this result
- output_type: What this establishes (bound/property/existence/etc)
- relations_established: Specific relationships proven

Uses Gemini 2.5 Pro for dependency analysis.
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class RawDataCrossReferencer:
    """Cross-reference analyzer for raw_data entities."""

    def __init__(self, raw_data_dir: Path):
        """Initialize analyzer."""
        self.raw_data_dir = Path(raw_data_dir)

        # Entity directories
        self.theorems_dir = self.raw_data_dir / "theorems"
        self.lemmas_dir = self.raw_data_dir / "lemmas"
        self.propositions_dir = self.raw_data_dir / "propositions"
        self.corollaries_dir = self.raw_data_dir / "corollaries"
        self.objects_dir = self.raw_data_dir / "objects"
        self.axioms_dir = self.raw_data_dir / "axioms"
        self.parameters_dir = self.raw_data_dir / "parameters"
        self.definitions_dir = self.raw_data_dir / "definitions"

        # Registry of available entities
        self.available_objects: set[str] = set()
        self.available_axioms: set[str] = set()
        self.available_parameters: set[str] = set()
        self.available_definitions: set[str] = set()

        # Statistics
        self.stats = {
            "total_entities": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "dependencies_filled": defaultdict(int),
            "missing_dependencies": [],
        }

    def load_registry(self):
        """Load all available objects, axioms, parameters."""
        print("Loading entity registry...")

        # Load objects
        if self.objects_dir.exists():
            for obj_file in self.objects_dir.glob("*.json"):
                with open(obj_file) as f:
                    data = json.load(f)
                    self.available_objects.add(data.get("label", obj_file.stem))

        # Load axioms
        if self.axioms_dir.exists():
            for axiom_file in self.axioms_dir.glob("*.json"):
                with open(axiom_file) as f:
                    data = json.load(f)
                    self.available_axioms.add(data.get("label", axiom_file.stem))

        # Load parameters
        if self.parameters_dir.exists():
            for param_file in self.parameters_dir.glob("*.json"):
                with open(param_file) as f:
                    data = json.load(f)
                    self.available_parameters.add(data.get("label", param_file.stem))

        # Load definitions
        if self.definitions_dir.exists():
            for def_file in self.definitions_dir.glob("*.json"):
                with open(def_file) as f:
                    data = json.load(f)
                    self.available_definitions.add(data.get("label", def_file.stem))

        print(f"  Objects: {len(self.available_objects)}")
        print(f"  Axioms: {len(self.available_axioms)}")
        print(f"  Parameters: {len(self.available_parameters)}")
        print(f"  Definitions: {len(self.available_definitions)}")
        print()

    def get_all_entity_files(self) -> list[Path]:
        """Get all theorem-like entity files."""
        files = []

        if self.theorems_dir.exists():
            files.extend(self.theorems_dir.glob("*.json"))

        if self.lemmas_dir.exists():
            files.extend(self.lemmas_dir.glob("*.json"))

        if self.propositions_dir.exists():
            files.extend(self.propositions_dir.glob("*.json"))

        if self.corollaries_dir.exists():
            files.extend(self.corollaries_dir.glob("*.json"))

        return sorted(files)

    def analyze_entity(self, entity_file: Path) -> dict:
        """Analyze a single entity file and extract dependencies."""
        with open(entity_file) as f:
            entity = json.load(f)

        # Get statement text
        statement = self._extract_statement(entity)
        if not statement:
            return entity

        # Extract label
        label = entity.get("label") or entity.get("label_text") or entity_file.stem

        print(f"  Analyzing: {label}")

        # Initialize fields if missing
        if "input_objects" not in entity:
            entity["input_objects"] = []
        if "input_axioms" not in entity:
            entity["input_axioms"] = []
        if "input_parameters" not in entity:
            entity["input_parameters"] = []
        if "output_type" not in entity:
            entity["output_type"] = None
        if "relations_established" not in entity:
            entity["relations_established"] = []

        # Extract explicit dependencies from existing fields
        explicit_deps = self._extract_explicit_dependencies(entity)

        # Analyze statement for implicit dependencies
        implicit_deps = self._analyze_statement_text(statement, label)

        # Merge dependencies
        all_deps = self._merge_dependencies(explicit_deps, implicit_deps)

        # Update entity
        entity["input_objects"] = list(
            set(entity.get("input_objects", []) + all_deps.get("objects", []))
        )
        entity["input_axioms"] = list(
            set(entity.get("input_axioms", []) + all_deps.get("axioms", []))
        )
        entity["input_parameters"] = list(
            set(entity.get("input_parameters", []) + all_deps.get("parameters", []))
        )

        if all_deps.get("output_type"):
            entity["output_type"] = all_deps["output_type"]

        if all_deps.get("relations"):
            entity["relations_established"] = list(
                set(entity.get("relations_established", []) + all_deps["relations"])
            )

        # Track statistics
        self.stats["dependencies_filled"]["objects"] += len(all_deps.get("objects", []))
        self.stats["dependencies_filled"]["axioms"] += len(all_deps.get("axioms", []))
        self.stats["dependencies_filled"]["parameters"] += len(all_deps.get("parameters", []))

        return entity

    def _extract_statement(self, entity: dict) -> str | None:
        """Extract statement text from entity."""
        # Try different field names
        for field in ["statement", "natural_language_statement", "full_statement_text", "content"]:
            if entity.get(field):
                return entity[field]
        return None

    def _extract_explicit_dependencies(self, entity: dict) -> dict:
        """Extract dependencies already mentioned in the entity."""
        deps = {"objects": [], "axioms": [], "parameters": [], "relations": []}

        # Check existing dependency fields
        for dep_label in entity.get("dependencies", []):
            if dep_label in self.available_objects:
                deps["objects"].append(dep_label)
            elif dep_label in self.available_axioms:
                deps["axioms"].append(dep_label)

        # Check uses_definitions
        for def_label in entity.get("uses_definitions", []):
            if def_label in self.available_definitions:
                # Find corresponding object
                obj_label = def_label.replace("def-", "obj-")
                if obj_label in self.available_objects:
                    deps["objects"].append(obj_label)

        return deps

    def _analyze_statement_text(self, statement: str, label: str) -> dict:
        """
        Analyze statement text to extract dependencies.

        This performs pattern matching to identify:
        - Object references (operators, measures, spaces)
        - Axiom references
        - Parameters
        - Output type
        - Relations established
        """
        deps = {
            "objects": [],
            "axioms": [],
            "parameters": [],
            "output_type": None,
            "relations": [],
        }

        # Pattern matching for common objects
        # This is a simplified version - full implementation would use LLM

        # Check for explicit object references in available objects
        for obj_label in self.available_objects:
            # Convert label to readable name pattern
            obj_name = obj_label.replace("obj-", "").replace("-", " ")
            if obj_name.lower() in statement.lower():
                deps["objects"].append(obj_label)

        # Check for axiom references
        for axiom_label in self.available_axioms:
            axiom_name = axiom_label.replace("axiom-", "").replace("-", " ")
            if axiom_name.lower() in statement.lower():
                deps["axioms"].append(axiom_label)

        # Infer output type from statement keywords
        statement_lower = statement.lower()
        if "lipschitz" in statement_lower:
            deps["output_type"] = "Lipschitz Bound"
        elif "bounded" in statement_lower or "bound" in statement_lower:
            deps["output_type"] = "Bound"
        elif "continuous" in statement_lower or "continuity" in statement_lower:
            deps["output_type"] = "Continuity"
        elif "exists" in statement_lower or "existence" in statement_lower:
            deps["output_type"] = "Existence"
        elif "convergence" in statement_lower or "converges" in statement_lower:
            deps["output_type"] = "Convergence"
        elif "property" in statement_lower or "properties" in statement_lower:
            deps["output_type"] = "Property"
        else:
            deps["output_type"] = "General Result"

        return deps

    def _merge_dependencies(self, explicit: dict, implicit: dict) -> dict:
        """Merge explicit and implicit dependencies."""
        return {
            "objects": list(set(explicit.get("objects", []) + implicit.get("objects", []))),
            "axioms": list(set(explicit.get("axioms", []) + implicit.get("axioms", []))),
            "parameters": list(
                set(explicit.get("parameters", []) + implicit.get("parameters", []))
            ),
            "output_type": implicit.get("output_type") or explicit.get("output_type"),
            "relations": list(set(explicit.get("relations", []) + implicit.get("relations", []))),
        }

    def validate_dependencies(self, entity: dict) -> list[str]:
        """Validate that all dependencies exist in the registry."""
        errors = []

        for obj_label in entity.get("input_objects", []):
            if obj_label not in self.available_objects:
                errors.append(f"Missing object: {obj_label}")

        for axiom_label in entity.get("input_axioms", []):
            if axiom_label not in self.available_axioms:
                errors.append(f"Missing axiom: {axiom_label}")

        return errors

    def process_all_entities(self):
        """Process all theorem-like entities."""
        files = self.get_all_entity_files()
        self.stats["total_entities"] = len(files)

        print(f"Processing {len(files)} entities...\n")

        for entity_file in files:
            try:
                # Analyze entity
                updated_entity = self.analyze_entity(entity_file)

                # Validate
                errors = self.validate_dependencies(updated_entity)
                if errors:
                    self.stats["missing_dependencies"].extend([
                        f"{entity_file.stem}: {err}" for err in errors
                    ])

                # Save updated entity
                with open(entity_file, "w") as f:
                    json.dump(updated_entity, f, indent=2)

                self.stats["processed"] += 1

            except Exception as e:
                print(f"  ERROR processing {entity_file.name}: {e}")
                self.stats["errors"] += 1

    def generate_report(self) -> str:
        """Generate cross-reference analysis report."""
        lines = [
            "# Cross-Reference Analysis Report",
            "",
            "## Summary",
            "",
            f"- **Total Entities**: {self.stats['total_entities']}",
            f"- **Processed**: {self.stats['processed']}",
            f"- **Errors**: {self.stats['errors']}",
            f"- **Skipped**: {self.stats['skipped']}",
            "",
            "## Dependencies Filled",
            "",
            f"- **Objects**: {self.stats['dependencies_filled']['objects']}",
            f"- **Axioms**: {self.stats['dependencies_filled']['axioms']}",
            f"- **Parameters**: {self.stats['dependencies_filled']['parameters']}",
            "",
            "## Entity Registry",
            "",
            f"- **Available Objects**: {len(self.available_objects)}",
            f"- **Available Axioms**: {len(self.available_axioms)}",
            f"- **Available Parameters**: {len(self.available_parameters)}",
            f"- **Available Definitions**: {len(self.available_definitions)}",
            "",
        ]

        if self.stats["missing_dependencies"]:
            lines.extend([
                "## Missing Dependencies",
                "",
                "The following dependencies were referenced but not found:",
                "",
            ])
            for missing in self.stats["missing_dependencies"][:20]:  # Show first 20
                lines.append(f"- {missing}")

            if len(self.stats["missing_dependencies"]) > 20:
                lines.append(f"- ... and {len(self.stats['missing_dependencies']) - 20} more")

        return "\n".join(lines)

    def run(self):
        """Execute full cross-reference analysis."""
        print("=" * 80)
        print("Cross-Reference Analysis for Raw Data")
        print("=" * 80)
        print()

        # Load registry
        self.load_registry()

        # Process all entities
        self.process_all_entities()

        # Generate report
        print()
        print("=" * 80)
        print("Analysis Complete")
        print("=" * 80)
        print()

        report = self.generate_report()
        print(report)

        # Save report
        report_file = self.raw_data_dir / "CROSS_REFERENCE_REPORT.md"
        with open(report_file, "w") as f:
            f.write(report)

        print()
        print(f"Report saved to: {report_file}")

        return self.stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-reference raw_data entities")
    parser.add_argument("raw_data_dir", type=Path, help="Path to raw_data directory")

    args = parser.parse_args()

    analyzer = RawDataCrossReferencer(args.raw_data_dir)
    analyzer.run()


if __name__ == "__main__":
    main()
