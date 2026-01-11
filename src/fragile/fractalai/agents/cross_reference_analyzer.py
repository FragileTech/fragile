"""
Cross-Reference Analyzer Agent - Fill Relationships Between Mathematical Objects.

This agent processes document-parser output and fills in all relationships
between mathematical entities, including both explicit cross-references and
implicit dependencies discovered through LLM analysis.

Stage: Stage 1.5 (between document-parser and document-refiner)
Input: JSON files from document-parser output
Output: Enhanced JSON files with filled relationships + relationships/ directory

Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
import json
import operator
from pathlib import Path

from fragile.agents.dependency_detector import DependencyDetector, DependencyReport
from fragile.agents.relationship_builder import RelationshipBuilder, RelationshipValidator
from fragile.proofs import (
    MathematicalRegistry,
    Relationship,
)


class CrossReferenceAnalyzer:
    """
    Autonomous agent for discovering relationships between mathematical entities.

    Processing Pipeline:
    1. Load document-parser output (extraction_inventory.json + individual JSONs)
    2. Build registry of all objects/theorems/axioms
    3. For each theorem/lemma/proposition:
       a. Extract explicit cross-refs (already available)
       b. Detect implicit dependencies via LLM analysis
       c. Classify relationship types
       d. Fill input_objects, input_axioms, input_parameters
       e. Create Relationship objects
    4. Export enhanced JSONs + relationships directory
    """

    def __init__(
        self,
        source_dir: Path | str,
        use_llm: bool = True,
        dual_ai: bool = False,
        glossary_path: Path | None = None,
    ):
        """
        Initialize cross-reference analyzer.

        Args:
            source_dir: Directory containing document-parser output
            use_llm: Whether to use LLM for implicit dependency detection
            dual_ai: Whether to use dual AI analysis (Gemini + Codex)
            glossary_path: Path to docs/glossary.md
        """
        self.source_dir = Path(source_dir)
        self.use_llm = use_llm
        self.dual_ai = dual_ai
        self.glossary_path = glossary_path

        # Determine base directory (remove /data suffix if present)
        if self.source_dir.name == "data":
            self.base_dir = self.source_dir.parent
        else:
            self.base_dir = self.source_dir

        # Set up output directories
        self.data_dir = self.base_dir / "data"
        self.objects_dir = self.base_dir / "objects"
        self.theorems_dir = self.base_dir / "theorems"
        self.axioms_dir = self.base_dir / "axioms"
        self.relationships_dir = self.base_dir / "relationships"

        # Initialize registry and components
        self.registry = MathematicalRegistry()
        self.relationship_builder = RelationshipBuilder()
        self.dependency_detector: DependencyDetector | None = None

        # Statistics
        self.stats = {
            "theorems_processed": 0,
            "explicit_refs_processed": 0,
            "implicit_deps_discovered": 0,
            "relationships_created": 0,
            "input_objects_filled": 0,
            "input_axioms_filled": 0,
            "input_parameters_filled": 0,
            "validation_errors": 0,
        }

    def run(self) -> dict:
        """
        Execute full cross-reference analysis pipeline.

        Returns:
            Summary report dictionary
        """
        print("🔗 CrossReferenceAnalyzer Starting")
        print(f"   Source: {self.source_dir}")
        print(f"   Use LLM: {self.use_llm}")
        print(f"   Dual AI: {self.dual_ai}")
        print()

        # Phase 1: Setup and Registry Building
        print("  Phase 1: Loading registry...")
        self._load_registry()
        print(f"    ✓ Loaded {len(self.registry.get_all_objects())} objects")
        print(f"    ✓ Loaded {len(self.registry.get_all_theorems())} theorems")
        print(f"    ✓ Loaded {len(self.registry.get_all_axioms())} axioms")

        # Initialize dependency detector if using LLM
        if self.use_llm:
            print("  Phase 2: Initializing LLM dependency detector...")
            self.dependency_detector = DependencyDetector(
                registry=self.registry, glossary_path=self.glossary_path, use_dual_ai=self.dual_ai
            )
            print("    ✓ Detector initialized")

        # Phase 3: Process explicit cross-references
        print("  Phase 3: Processing explicit cross-references...")
        self._process_explicit_refs()
        print(f"    ✓ Processed {self.stats['explicit_refs_processed']} explicit refs")

        # Phase 4: Detect implicit dependencies (if LLM enabled)
        if self.use_llm and self.dependency_detector:
            print("  Phase 4: Detecting implicit dependencies (LLM)...")
            self._process_implicit_deps()
            print(f"    ✓ Discovered {self.stats['implicit_deps_discovered']} implicit deps")

        # Phase 5: Construct relationships
        print("  Phase 5: Constructing relationships...")
        relationships = self._construct_relationships()
        print(f"    ✓ Created {len(relationships)} relationships")

        # Phase 6: Validate relationships
        print("  Phase 6: Validating relationships...")
        self._validate_relationships(relationships)
        print(f"    ✓ Validation errors: {self.stats['validation_errors']}")

        # Phase 7: Export enhanced JSONs and relationships
        print("  Phase 7: Exporting results...")
        self._export_enhanced_jsons()
        self._export_relationships(relationships)
        print(f"    ✓ Exported to {self.base_dir}")

        # Generate summary
        summary = self._generate_summary()
        print()
        print("✅ Cross-reference analysis complete!")
        print(f"   Theorems processed: {self.stats['theorems_processed']}")
        print(f"   Relationships created: {self.stats['relationships_created']}")
        print(f"   Input objects filled: {self.stats['input_objects_filled']}")
        print(f"   Validation errors: {self.stats['validation_errors']}")

        return summary

    def _load_registry(self) -> None:
        """Load all mathematical objects into registry."""
        # Load objects
        if self.objects_dir.exists():
            for obj_file in self.objects_dir.glob("*.json"):
                with open(obj_file, encoding="utf-8") as f:
                    obj_data = json.load(f)
                    # Reconstruct object (simplified - would use proper Pydantic deserialization)
                    # For now, just track labels
                    self.registry._objects[obj_data["label"]] = obj_data

        # Load theorems
        if self.theorems_dir.exists():
            for thm_file in self.theorems_dir.glob("*.json"):
                with open(thm_file, encoding="utf-8") as f:
                    thm_data = json.load(f)
                    self.registry._theorems[thm_data["label"]] = thm_data

        # Load axioms
        if self.axioms_dir.exists():
            for axiom_file in self.axioms_dir.glob("*.json"):
                with open(axiom_file, encoding="utf-8") as f:
                    axiom_data = json.load(f)
                    self.registry._axioms[axiom_data["label"]] = axiom_data

    def _process_explicit_refs(self) -> None:
        """Process explicit cross-references from extraction_inventory.json."""
        inventory_file = self.data_dir / "extraction_inventory.json"
        if not inventory_file.exists():
            print("    ⚠️  No extraction_inventory.json found")
            return

        with open(inventory_file, encoding="utf-8") as f:
            inventory = json.load(f)

        # Process each directive with cross-refs
        for directive in inventory.get("directives", []):
            cross_refs = directive.get("cross_refs", [])
            if not cross_refs:
                continue

            source_label = directive["label"]
            for target_label in cross_refs:
                self._record_explicit_ref(source_label, target_label)
                self.stats["explicit_refs_processed"] += 1

    def _record_explicit_ref(self, source_label: str, target_label: str) -> None:
        """Record an explicit cross-reference."""
        # Update theorem JSON to include referenced object/theorem
        if source_label.startswith(("thm-", "lem-", "prop-")):
            thm_file = self.theorems_dir / f"{source_label}.json"
            if thm_file.exists():
                with open(thm_file, encoding="utf-8") as f:
                    thm_data = json.load(f)

                # Categorize the reference
                if target_label.startswith("obj-"):
                    if target_label not in thm_data["input_objects"]:
                        thm_data["input_objects"].append(target_label)
                        self.stats["input_objects_filled"] += 1
                elif target_label.startswith("axiom-"):
                    if target_label not in thm_data["input_axioms"]:
                        thm_data["input_axioms"].append(target_label)
                        self.stats["input_axioms_filled"] += 1
                elif target_label.startswith(("thm-", "lem-", "prop-")):
                    if target_label not in thm_data["internal_lemmas"]:
                        thm_data["internal_lemmas"].append(target_label)

                # Save updated theorem
                with open(thm_file, "w", encoding="utf-8") as f:
                    json.dump(thm_data, f, indent=2)

    def _process_implicit_deps(self) -> None:
        """Detect implicit dependencies using LLM analysis."""
        if not self.dependency_detector:
            return

        # Process each theorem
        for thm_file in self.theorems_dir.glob("*.json"):
            with open(thm_file, encoding="utf-8") as f:
                thm_data = json.load(f)

            # Get theorem content from source document
            theorem_content = self._get_theorem_content(thm_data)
            if not theorem_content:
                continue

            # Analyze with LLM
            try:
                report = self.dependency_detector.analyze_theorem(
                    theorem_label=thm_data["label"], theorem_content=theorem_content
                )

                # Fill in discovered dependencies
                self._apply_dependency_report(thm_data, report)

                # Save updated theorem
                with open(thm_file, "w", encoding="utf-8") as f:
                    json.dump(thm_data, f, indent=2)

                self.stats["theorems_processed"] += 1
                self.stats["implicit_deps_discovered"] += (
                    len(report.input_objects)
                    + len(report.input_axioms)
                    + len(report.input_parameters)
                )

            except Exception as e:
                print(f"    ⚠️  Error analyzing {thm_data['label']}: {e}")

    def _get_theorem_content(self, thm_data: dict) -> str | None:
        """Get theorem content from source document."""
        # This would read from the source markdown document using line_range
        # For now, return placeholder
        return f"Theorem: {thm_data.get('name', '')}"

    def _apply_dependency_report(self, thm_data: dict, report: DependencyReport) -> None:
        """Apply discovered dependencies to theorem data."""
        # Add input objects
        for obj_dep in report.input_objects:
            if obj_dep.label not in thm_data["input_objects"]:
                thm_data["input_objects"].append(obj_dep.label)
                self.stats["input_objects_filled"] += 1

        # Add input axioms
        for axiom_dep in report.input_axioms:
            if axiom_dep.label not in thm_data["input_axioms"]:
                thm_data["input_axioms"].append(axiom_dep.label)
                self.stats["input_axioms_filled"] += 1

        # Add input parameters
        for param_dep in report.input_parameters:
            if param_dep.label not in thm_data["input_parameters"]:
                thm_data["input_parameters"].append(param_dep.label)
                self.stats["input_parameters_filled"] += 1

        # Add attributes_required
        for obj_label, props in report.attributes_required.items():
            if obj_label not in thm_data["attributes_required"]:
                thm_data["attributes_required"][obj_label] = []
            for prop in props:
                if prop not in thm_data["attributes_required"][obj_label]:
                    thm_data["attributes_required"][obj_label].append(prop)

    def _construct_relationships(self) -> list[Relationship]:
        """Construct Relationship objects from filled data."""
        relationships = []

        # For each theorem, create relationships from input_objects
        for thm_file in self.theorems_dir.glob("*.json"):
            with open(thm_file, encoding="utf-8") as f:
                thm_data = json.load(f)

            theorem_label = thm_data["label"]

            # Create relationships for each input object
            for obj_label in thm_data.get("input_objects", []):
                try:
                    rel = self.relationship_builder.build_simple_uses_relationship(
                        source_label=theorem_label,
                        target_label=obj_label,
                        established_by=theorem_label,
                        chapter=thm_data.get("chapter"),
                        document=thm_data.get("document"),
                    )
                    relationships.append(rel)
                    self.stats["relationships_created"] += 1
                except Exception as e:
                    print(f"    ⚠️  Error creating relationship {theorem_label} → {obj_label}: {e}")

        return relationships

    def _validate_relationships(self, relationships: list[Relationship]) -> None:
        """Validate relationships against registry."""
        validator = RelationshipValidator(self.registry)

        for rel in relationships:
            is_valid, errors = validator.validate_relationship(rel)
            if not is_valid:
                self.stats["validation_errors"] += len(errors)
                for error in errors:
                    print(f"    ⚠️  {error}")

    def _export_enhanced_jsons(self) -> None:
        """Export enhanced theorem JSONs with filled relationships."""
        # Already saved during processing

    def _export_relationships(self, relationships: list[Relationship]) -> None:
        """Export relationships to relationships/ directory."""
        # Create relationships directory
        self.relationships_dir.mkdir(parents=True, exist_ok=True)

        # Export each relationship
        for rel in relationships:
            rel_file = self.relationships_dir / f"{rel.label}.json"
            with open(rel_file, "w", encoding="utf-8") as f:
                json.dump(rel.model_dump(), f, indent=2)

        # Create relationship index
        index = {
            "total_relationships": len(relationships),
            "by_type": {},
            "timestamp": datetime.now().isoformat(),
        }

        for rel in relationships:
            rel_type = rel.relationship_type.value
            if rel_type not in index["by_type"]:
                index["by_type"][rel_type] = 0
            index["by_type"][rel_type] += 1

        with open(self.relationships_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        # Generate report
        self._generate_relationship_report(relationships)

    def _generate_relationship_report(self, relationships: list[Relationship]) -> None:
        """Generate human-readable relationship report."""
        report_lines = [
            "# Cross-Reference Analysis Report",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Source**: {self.source_dir}",
            "",
            "## Statistics",
            "",
            f"- **Theorems Processed**: {self.stats['theorems_processed']}",
            f"- **Explicit Refs**: {self.stats['explicit_refs_processed']}",
            f"- **Implicit Deps Discovered**: {self.stats['implicit_deps_discovered']}",
            f"- **Relationships Created**: {self.stats['relationships_created']}",
            f"- **Input Objects Filled**: {self.stats['input_objects_filled']}",
            f"- **Input Axioms Filled**: {self.stats['input_axioms_filled']}",
            f"- **Validation Errors**: {self.stats['validation_errors']}",
            "",
            "## Relationships by Type",
            "",
        ]

        # Count by type
        by_type: dict[str, int] = {}
        for rel in relationships:
            rel_type = rel.relationship_type.value
            by_type[rel_type] = by_type.get(rel_type, 0) + 1

        for rel_type, count in sorted(by_type.items(), key=operator.itemgetter(1), reverse=True):
            report_lines.append(f"- **{rel_type}**: {count}")

        report_content = "\n".join(report_lines)

        with open(self.relationships_dir / "REPORT.md", "w", encoding="utf-8") as f:
            f.write(report_content)

    def _generate_summary(self) -> dict:
        """Generate summary report."""
        return {
            "source": str(self.source_dir),
            "timestamp": datetime.now().isoformat(),
            "use_llm": self.use_llm,
            "dual_ai": self.dual_ai,
            "statistics": self.stats,
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Reference Analyzer Agent")
    parser.add_argument("source", type=Path, help="Source directory (document-parser output)")
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM implicit dependency detection"
    )
    parser.add_argument(
        "--dual-ai", action="store_true", help="Use dual AI analysis (Gemini + Codex)"
    )
    parser.add_argument("--glossary", type=Path, help="Path to docs/glossary.md")

    args = parser.parse_args()

    agent = CrossReferenceAnalyzer(
        source_dir=args.source,
        use_llm=not args.no_llm,
        dual_ai=args.dual_ai,
        glossary_path=args.glossary,
    )

    result = agent.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
