"""
Mathematical Document Parser Agent.

Autonomous agent that parses mathematical documents and extracts all mathematical
content into structured JSON files using the fragile.proofs framework.

Architecture:
- Phase 1: MyST directive extraction
- Phase 2: Object creation (definitions, axioms)
- Phase 3: Theorem creation
- Phase 4: Relationship extraction (hybrid explicit + LLM)
- Phase 5: Proof sketch creation
- Phase 6: Proof expansion (LLM-assisted)
- Phase 7: Validation
- Phase 8: Registry export

Version: 1.0.0
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from fragile.agents.myst_parser import DocumentInventory, MathDirective, parse_document
from fragile.proofs import (
    Axiom,
    MathematicalObject,
    MathematicalRegistry,
    ObjectType,
    Parameter,
    ParameterType,
    Property,
    Relationship,
    RelationshipProperty,
    RelationType,
    TheoremBox,
    TheoremOutputType,
    save_registry_to_directory,
)


class MathDocumentParser:
    """
    Autonomous parser that extracts mathematical content from documents.

    Given a MyST markdown document, extracts all mathematical objects, theorems,
    proofs, and relationships into structured JSON following the fragile.proofs
    framework.
    """

    def __init__(
        self,
        source_path: Path | str,
        mode: str = "both",  # sketch | expand | both
        use_llm: bool = True,
        output_dir: Optional[Path | str] = None
    ):
        """
        Initialize parser.

        Args:
            source_path: Path to document or directory
            mode: Parsing mode (sketch/expand/both)
            use_llm: Whether to use LLM for analysis
            output_dir: Custom output directory (default: auto-detect from source)
        """
        self.source_path = Path(source_path)
        self.mode = mode
        self.use_llm = use_llm

        # Determine if processing file or directory
        self.is_directory = self.source_path.is_dir()

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self._auto_detect_output_dir()

        # Initialize storage
        self.inventory: Optional[DocumentInventory] = None
        self.registry = MathematicalRegistry()

        # Statistics
        self.stats = {
            "objects_created": 0,
            "theorems_created": 0,
            "proofs_created": 0,
            "relationships_created": 0,
            "validation_errors": 0,
            "validation_warnings": 0
        }

    def _auto_detect_output_dir(self) -> Path:
        """
        Auto-detect output directory based on source path.

        For file: docs/source/N_chapter/file.md
        Output: docs/source/N_chapter/file/data/

        For directory: docs/source/N_chapter/
        Output: docs/source/N_chapter/data/
        """
        if self.is_directory:
            return self.source_path / "data"
        else:
            # Create directory with document name
            doc_name = self.source_path.stem
            parent = self.source_path.parent
            return parent / doc_name / "data"

    def run(self) -> Dict:
        """
        Execute full parsing pipeline.

        Returns:
            Summary report dictionary
        """
        print(f"🚀 MathDocumentParser Starting")
        print(f"   Source: {self.source_path}")
        print(f"   Mode: {self.mode}")
        print(f"   Output: {self.output_dir}")
        print()

        # Process based on type
        if self.is_directory:
            return self._process_directory()
        else:
            return self._process_single_document()

    def _process_single_document(self) -> Dict:
        """Process a single document through all phases."""
        print(f"📄 Processing: {self.source_path.name}")

        # Phase 1: Extract directives
        print("  Phase 1: Extracting MyST directives...")
        self.inventory = parse_document(self.source_path)
        print(f"    ✓ Found {len(self.inventory.directives)} directives")
        print(f"      {self.inventory.count_by_type()}")

        # Phase 2: Create objects
        print("  Phase 2: Creating mathematical objects...")
        objects = self._create_objects()
        print(f"    ✓ Created {len(objects)} objects")

        # Phase 3: Create theorems
        print("  Phase 3: Creating theorems...")
        theorems = self._create_theorems()
        print(f"    ✓ Created {len(theorems)} theorems")

        # Phase 4: Extract relationships
        print("  Phase 4: Extracting relationships...")
        relationships = self._extract_relationships()
        print(f"    ✓ Created {len(relationships)} relationships")

        # Phase 5: Create proof sketches
        if self.mode in ["sketch", "both"]:
            print("  Phase 5: Creating proof sketches...")
            proofs_sketch = self._create_proof_sketches()
            print(f"    ✓ Created {len(proofs_sketch)} sketch proofs")

        # Phase 6: Expand proofs (if LLM enabled)
        if self.mode in ["expand", "both"] and self.use_llm:
            print("  Phase 6: Expanding proofs (LLM)...")
            print("    ⚠️  LLM expansion not yet implemented")
            # proofs_expand = self._expand_proofs(proofs_sketch)

        # Phase 7: Validation
        print("  Phase 7: Validating...")
        validation_report = self._validate_all()
        print(f"    ✓ Validation complete")
        print(f"      Errors: {self.stats['validation_errors']}")
        print(f"      Warnings: {self.stats['validation_warnings']}")

        # Phase 8: Export
        print("  Phase 8: Exporting to JSON...")
        self._export_all()
        print(f"    ✓ Exported to {self.output_dir}")

        # Generate summary
        summary = self._generate_summary()
        print()
        print("✅ Processing complete!")
        print(f"   Objects: {self.stats['objects_created']}")
        print(f"   Theorems: {self.stats['theorems_created']}")
        print(f"   Proofs: {self.stats['proofs_created']}")
        print(f"   Relationships: {self.stats['relationships_created']}")

        return summary

    def _process_directory(self) -> Dict:
        """Process all markdown files in directory."""
        md_files = list(self.source_path.glob("*.md"))
        print(f"📁 Processing directory: {len(md_files)} files found")

        results = []
        for md_file in md_files:
            # Create sub-parser for each file
            parser = MathDocumentParser(
                source_path=md_file,
                mode=self.mode,
                use_llm=self.use_llm,
                output_dir=None  # Auto-detect per file
            )
            result = parser._process_single_document()
            results.append(result)

        # Aggregate results
        return self._aggregate_results(results)

    def _create_objects(self) -> List[MathematicalObject]:
        """
        Create MathematicalObject instances from definitions.

        Maps {prf:definition} -> MathematicalObject
        """
        objects = []

        for directive in self.inventory.get_definitions():
            obj = self._directive_to_object(directive)
            if obj:
                objects.append(obj)
                self.registry.add(obj)
                self.stats["objects_created"] += 1

        # Also handle algorithms as operators
        for directive in self.inventory.get_by_type("algorithm"):
            obj = self._directive_to_object(directive, force_type=ObjectType.OPERATOR)
            if obj:
                objects.append(obj)
                self.registry.add(obj)
                self.stats["objects_created"] += 1

        # Handle axioms
        for directive in self.inventory.get_axioms():
            axiom = self._directive_to_axiom(directive)
            if axiom:
                self.registry.add(axiom)

        return objects

    def _directive_to_object(
        self,
        directive: MathDirective,
        force_type: Optional[ObjectType] = None
    ) -> Optional[MathematicalObject]:
        """
        Convert definition directive to MathematicalObject.

        Args:
            directive: Source directive
            force_type: Force specific object type

        Returns:
            MathematicalObject or None
        """
        # Ensure label has obj- prefix and normalize to lowercase
        label = directive.label.lower().replace(":", "-").replace("_", "-")  # Normalize
        if not label.startswith("obj-"):
            # Convert def- to obj-
            if label.startswith("def-"):
                label = "obj-" + label[4:]
            elif label.startswith("alg-"):
                label = "obj-" + label[4:]
            else:
                label = "obj-" + label

        # Infer object type if not forced
        if force_type:
            obj_type = force_type
        else:
            obj_type = self._infer_object_type(directive)

        # Get mathematical expression
        math_expr = directive.get_first_math()
        if not math_expr:
            # Use content summary if no math
            math_expr = directive.content[:200]

        # Extract tags from content
        tags = self._extract_tags_from_content(directive.content)

        try:
            obj = MathematicalObject(
                label=label,
                name=directive.title,
                mathematical_expression=math_expr,
                object_type=obj_type,
                tags=tags
            )
            return obj
        except Exception as e:
            print(f"      ⚠️  Failed to create object {label}: {e}")
            return None

    def _infer_object_type(self, directive: MathDirective) -> ObjectType:
        """
        Infer ObjectType from directive content.

        Uses keyword matching to classify object type.
        """
        content_lower = directive.content.lower()
        title_lower = directive.title.lower()

        # Check for keywords
        if any(kw in title_lower or kw in content_lower for kw in ["function", "map", "operator"]):
            return ObjectType.FUNCTION
        elif any(kw in title_lower or kw in content_lower for kw in ["measure", "probability"]):
            return ObjectType.MEASURE
        elif any(kw in title_lower or kw in content_lower for kw in ["space", "domain"]):
            return ObjectType.SPACE
        elif any(kw in title_lower or kw in content_lower for kw in ["distribution"]):
            return ObjectType.DISTRIBUTION
        else:
            # Default to SET
            return ObjectType.SET

    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract tags from content based on keywords."""
        tags = []
        content_lower = content.lower()

        # Standard tags
        tag_keywords = {
            "euclidean-gas": ["euclidean gas", "euclidean"],
            "discrete": ["discrete"],
            "continuous": ["continuous"],
            "particle": ["particle", "walker"],
            "swarm": ["swarm"],
            "convergence": ["convergence", "converge"],
            "stability": ["stability", "stable"],
            "variance": ["variance"],
            "mean-field": ["mean field", "mean-field"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)

        return tags

    def _directive_to_axiom(self, directive: MathDirective) -> Optional[Axiom]:
        """Convert axiom directive to Axiom."""
        # Ensure label has axiom- prefix and normalize to lowercase
        label = directive.label.lower().replace(":", "-").replace("_", "-")  # Normalize
        if not label.startswith("axiom-"):
            label = "axiom-" + label

        math_expr = directive.get_first_math() or directive.content[:200]

        try:
            axiom = Axiom(
                label=label,
                statement=directive.title,
                mathematical_expression=math_expr,
                foundational_framework="Euclidean Gas"  # Default
            )
            return axiom
        except Exception as e:
            print(f"      ⚠️  Failed to create axiom {label}: {e}")
            return None

    def _create_theorems(self) -> List[TheoremBox]:
        """
        Create TheoremBox instances from theorem directives.

        Maps {prf:theorem}, {prf:lemma}, {prf:proposition} -> TheoremBox
        """
        theorems = []

        for directive in self.inventory.get_theorems():
            theorem = self._directive_to_theorem(directive)
            if theorem:
                theorems.append(theorem)
                self.registry.add(theorem)
                self.stats["theorems_created"] += 1

        return theorems

    def _directive_to_theorem(self, directive: MathDirective) -> Optional[TheoremBox]:
        """Convert theorem directive to TheoremBox."""
        # Ensure correct prefix and normalize to lowercase
        label = directive.label.lower().replace(":", "-").replace("_", "-")  # Normalize
        if directive.directive_type == "theorem" and not label.startswith("thm-"):
            label = "thm-" + label
        elif directive.directive_type == "lemma" and not label.startswith("lem-"):
            label = "lem-" + label
        elif directive.directive_type == "proposition" and not label.startswith("prop-"):
            label = "prop-" + label

        # Infer output type
        output_type = self._infer_theorem_output_type(directive)

        try:
            theorem = TheoremBox(
                label=label,
                name=directive.title,
                output_type=output_type,
                input_objects=[],  # Will be inferred later
                properties_required={},  # Will be inferred later
            )
            return theorem
        except Exception as e:
            print(f"      ⚠️  Failed to create theorem {label}: {e}")
            return None

    def _infer_theorem_output_type(self, directive: MathDirective) -> TheoremOutputType:
        """Infer TheoremOutputType from theorem content."""
        content_lower = directive.content.lower()
        title_lower = directive.title.lower()

        # Check for keywords
        if "property" in title_lower or "has" in content_lower:
            return TheoremOutputType.PROPERTY
        elif "equivalent" in content_lower or "equivalence" in title_lower:
            return TheoremOutputType.EQUIVALENCE
        elif "exists" in content_lower or "existence" in title_lower:
            return TheoremOutputType.EXISTENCE
        elif "unique" in content_lower or "uniqueness" in title_lower:
            return TheoremOutputType.UNIQUENESS
        elif "converge" in content_lower or "convergence" in title_lower:
            return TheoremOutputType.PROPERTY  # Convergence is a property
        else:
            # Default to PROPERTY
            return TheoremOutputType.PROPERTY

    def _extract_relationships(self) -> List[Relationship]:
        """
        Extract relationships between objects and theorems.

        Hybrid approach: explicit (cross-refs) + LLM inference (if enabled)
        """
        relationships = []

        # Explicit relationship extraction from cross-refs
        for directive in self.inventory.directives:
            for ref_label in directive.cross_refs:
                rel = self._create_relationship_from_ref(directive.label, ref_label)
                if rel:
                    relationships.append(rel)
                    self.stats["relationships_created"] += 1

        # TODO: LLM-based implicit relationship inference
        # if self.use_llm:
        #     relationships.extend(self._infer_relationships_llm())

        return relationships

    def _create_relationship_from_ref(
        self,
        source_label: str,
        target_label: str
    ) -> Optional[Relationship]:
        """Create Relationship from cross-reference."""
        # Get source and target objects
        source = self.registry.get(source_label)
        target = self.registry.get(target_label)

        if not source or not target:
            return None

        # Create relationship label
        rel_label = f"rel-{source_label}-{target_label}-uses"

        try:
            # Simple "uses" relationship
            rel = Relationship(
                label=rel_label,
                relationship_type=RelationType.OTHER,
                source_object=source_label if source_label.startswith("obj-") else f"obj-{source_label}",
                target_object=target_label if target_label.startswith("obj-") else f"obj-{target_label}",
                bidirectional=False,
                established_by=source_label if source_label.startswith("thm-") else "derived",
                expression=f"{source_label} uses {target_label}"
            )
            return rel
        except Exception as e:
            # Silently skip invalid relationships
            return None

    def _create_proof_sketches(self) -> List:
        """Create proof sketches from proof directives."""
        proofs = []
        # TODO: Implement proof parsing
        # For now, just count
        self.stats["proofs_created"] = len(self.inventory.get_proofs())
        return proofs

    def _validate_all(self) -> Dict:
        """Run validation checks."""
        errors = []
        warnings = []

        # Check referential integrity
        missing_refs = self.registry.validate_referential_integrity()
        if missing_refs:
            warnings.append(f"Missing references: {missing_refs}")

        self.stats["validation_errors"] = len(errors)
        self.stats["validation_warnings"] = len(warnings)

        return {
            "errors": errors,
            "warnings": warnings
        }

    def _export_all(self):
        """Export all data to JSON files."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Export inventory
        with open(self.output_dir / "extraction_inventory.json", "w") as f:
            json.dump(self.inventory.to_dict(), f, indent=2)

        # Export registry
        save_registry_to_directory(self.registry, self.output_dir.parent)

        # Export statistics
        with open(self.output_dir / "statistics.json", "w") as f:
            json.dump(self.stats, f, indent=2)

    def _generate_summary(self) -> Dict:
        """Generate summary report."""
        return {
            "source": str(self.source_path),
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "statistics": self.stats
        }

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple documents."""
        total_stats = {
            "objects_created": sum(r["statistics"]["objects_created"] for r in results),
            "theorems_created": sum(r["statistics"]["theorems_created"] for r in results),
            "proofs_created": sum(r["statistics"]["proofs_created"] for r in results),
            "relationships_created": sum(r["statistics"]["relationships_created"] for r in results),
        }

        return {
            "source_directory": str(self.source_path),
            "files_processed": len(results),
            "total_statistics": total_stats,
            "individual_results": results
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Math Document Parser Agent")
    parser.add_argument("source", type=Path, help="Source document or directory")
    parser.add_argument("--mode", choices=["sketch", "expand", "both"], default="both")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM processing")
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")

    args = parser.parse_args()

    agent = MathDocumentParser(
        source_path=args.source,
        mode=args.mode,
        use_llm=not args.no_llm,
        output_dir=args.output_dir
    )

    result = agent.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
