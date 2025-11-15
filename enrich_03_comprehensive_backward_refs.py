#!/usr/bin/env python3
"""
Comprehensive backward cross-reference enrichment for 03_cloning.md

This script systematically adds {prf:ref} directives to mathematical concepts
defined in earlier documents (01_fragile_gas_framework.md, 02_euclidean_gas.md).

Strategy:
- Parse document structure (sections, directives)
- For each mathematical entity, identify dependencies
- Add backward references maintaining natural flow
- Track referenced concepts to avoid over-referencing
- NEVER add forward references
"""

import re
from pathlib import Path
from typing import Dict, Set, List, Tuple

# Comprehensive mapping of concepts to labels from earlier documents
BACKWARD_REFERENCES = {
    # === Document 01: Fragile Gas Framework ===

    # Core state space definitions (Section 1)
    "walker": "def-walker",
    "swarm configuration": "def-swarm-and-state-space",
    "alive set": "def-alive-dead-sets",
    "dead set": "def-alive-dead-sets",
    "$\\mathcal{A}$": "def-alive-dead-sets",
    "$\\mathcal{D}$": "def-alive-dead-sets",
    "valid state space": "def-valid-state-space",
    "$\\mathcal{X}_{\\text{valid}}$": "def-valid-state-space",

    # Metrics and distances
    "N-particle displacement": "def-n-particle-displacement-metric",
    "displacement pseudometric": "def-n-particle-displacement-metric",
    "$d_{\\text{Disp},\\mathcal{Y}}$": "def-n-particle-displacement-metric",
    "metric quotient": "def-metric-quotient",
    "Polishness": "lem-polishness-and-w2",
    "W_2": "lem-polishness-and-w2",

    # Reference measures
    "reference noise": "def-reference-measures",
    "heat kernel": "def-reference-measures",
    "$P_\\sigma$": "def-reference-measures",
    "$Q_\\delta$": "def-reference-measures",

    # Axioms - Environmental
    "Axiom of Guaranteed Revival": "def-axiom-guaranteed-revival",
    "guaranteed revival": "def-axiom-guaranteed-revival",
    "Boundary Regularity": "def-axiom-boundary-regularity",
    "Hölder continuous": "def-axiom-boundary-regularity",
    "Boundary Smoothness": "def-axiom-boundary-smoothness",
    "finite perimeter": "def-axiom-boundary-smoothness",
    "Environmental Richness": "def-axiom-environmental-richness",

    # Axioms - Reward and measurement
    "Reward Regularity": "def-axiom-reward-regularity",
    "Lipschitz continuous": "def-axiom-reward-regularity",
    "projection compatibility": "def-axiom-projection-compatibility",
    "Bounded Algorithmic Diameter": "def-axiom-bounded-algorithmic-diameter",
    "$D_Y$": "def-axiom-bounded-algorithmic-diameter",
    "Range-Respecting Mean": "def-axiom-range-respecting-mean",

    # QSD and convergence
    "Quasi-Stationary Distribution": "def-qsd",
    "QSD": "def-qsd",
    "Foster-Lyapunov": "def-foster-lyapunov",
    "geometric ergodicity": "def-geometric-ergodicity",

    # === Document 02: Euclidean Gas ===

    # Euclidean Gas definition
    "Euclidean Gas": "def-euclidean-gas",
    "euclidean gas": "def-euclidean-gas",

    # Kinetic operator
    "Langevin dynamics": "def-langevin-operator",
    "Langevin operator": "def-langevin-operator",
    "$\\Psi_{\\text{kin}}$": "def-langevin-operator",
    "kinetic operator": "def-kinetic-operator",
    "BAOAB integrator": "def-baoab-integrator",
    "BAOAB": "def-baoab-integrator",

    # Hypocoercivity
    "hypocoercive": "def-hypocoercive-distance",
    "hypocoercive distance": "def-hypocoercive-distance",
    "$W_h^2$": "def-hypocoercive-distance",
}

# Specific sections where references should be added
SECTION_SPECIFIC_REFS = {
    "## 0. TLDR": [
        ("Safe Harbor", "axiom-safe-harbor"),
        ("QSD", "def-qsd"),
    ],

    "### 1.2. The Cloning Operator in the Fragile Gas Framework": [
        ("Fragile Gas framework", "def-fragile-gas-framework"),
        ("walker", "def-walker"),
    ],

    "### 2.1. The Single-Swarm State Space": [
        ("walker", "def-walker"),
        ("survival status", "def-walker"),
    ],

    "### 2.2. The Coupled Process and Synchronous Coupling": [
        ("Markov process", "def-markov-kernel"),
    ],

    "## 4. Foundational Assumptions and System Properties": [
        ("axioms", "def-fragile-gas-axioms"),
    ],
}


class BackwardReferenceEnricher:
    """Enriches markdown with backward cross-references."""

    def __init__(self, doc_path: Path):
        self.doc_path = doc_path
        self.content = doc_path.read_text()
        self.lines = self.content.split('\n')
        self.referenced: Set[str] = set()
        self.current_section = ""
        self.in_code_block = False
        self.in_math_block = False
        self.in_directive = False

    def should_skip_line(self, line: str) -> bool:
        """Determine if we should skip adding references to this line."""

        # Update state tracking
        if line.strip().startswith('```'):
            self.in_code_block = not self.in_code_block
            return True

        if line.strip() == '$$':
            self.in_math_block = not self.in_math_block
            return True

        if line.strip().startswith('::::'):
            self.in_directive = not self.in_directive
            return True

        # Skip if in special blocks
        if self.in_code_block or self.in_math_block or self.in_directive:
            return True

        # Skip lines that already have references
        if '{prf:ref}' in line:
            return True

        # Skip directive metadata lines
        if ':label:' in line or ':class:' in line:
            return True

        # Skip empty lines
        if not line.strip():
            return True

        return False

    def add_reference_to_line(self, line: str, concept: str, label: str) -> Tuple[str, bool]:
        """
        Add a backward reference to a concept in a line.

        Returns:
            (modified_line, added_ref_success)
        """

        # Create regex pattern for the concept
        # Handle both text and LaTeX math mode
        pattern = re.compile(
            r'(?<![a-zA-Z_])' + re.escape(concept) + r'(?![a-zA-Z_])',
            re.IGNORECASE
        )

        match = pattern.search(line)
        if not match:
            return line, False

        # Add reference after the concept
        pos = match.end()

        # Check if already followed by a reference
        if line[pos:pos+20].strip().startswith('({prf:ref}'):
            return line, False

        # Determine reference style based on context
        if '$' in line[match.start():match.end()]:
            # Math mode - add reference outside math
            ref = f" ({{prf:ref}}`{label}`)"
        else:
            # Regular text
            ref = f" ({{prf:ref}}`{label}`)"

        modified = line[:pos] + ref + line[pos:]
        return modified, True

    def enrich_document(self) -> str:
        """Main enrichment function."""

        result_lines = []
        reference_count = 0

        for i, line in enumerate(self.lines):
            # Track current section
            if line.startswith('#'):
                self.current_section = line.strip()

            # Check if we should skip this line
            if self.should_skip_line(line):
                result_lines.append(line)
                continue

            modified_line = line
            added_ref = False

            # Try to add references to concepts
            for concept, label in BACKWARD_REFERENCES.items():
                # Skip if already referenced this label
                if label in self.referenced:
                    continue

                # Skip if concept not in line
                if concept.lower() not in line.lower():
                    continue

                # Try to add reference
                new_line, success = self.add_reference_to_line(modified_line, concept, label)

                if success:
                    modified_line = new_line
                    self.referenced.add(label)
                    reference_count += 1
                    added_ref = True
                    break  # Only one reference per line

            result_lines.append(modified_line)

        print(f"Added {reference_count} backward references")
        print(f"Referenced {len(self.referenced)} unique concepts")

        return '\n'.join(result_lines)

    def add_manual_references(self, content: str) -> str:
        """Add manually curated references at specific locations."""

        # These are high-value additions that require precise placement
        manual_additions = [
            # TLDR section - reference Safe Harbor axiom (already present, verify)
            # Line 7 already has the reference

            # Section 1 Introduction - Add QSD reference
            (
                re.compile(r'(theory of \*\*Quasi-Stationary Distributions \(QSDs\)\*\*)'),
                r'\1 ({prf:ref}`def-qsd`)'
            ),

            # Section 2.1 - Reference walker definition
            (
                re.compile(r'(fundamental unit of the system is the walker)'),
                r'\1 ({prf:ref}`def-walker`)'
            ),

            # Section 2.4.1 - Reference domain axiom
            (
                re.compile(r'(valid domain for a single walker\'s position, \$\\mathcal\{X\}_\{\\text\{valid\}\}\$)'),
                r'\1 ({prf:ref}`def-valid-state-space`)'
            ),

            # Section 3 - Reference Wasserstein distance
            (
                re.compile(r'(squared hypocoercive Wasserstein distance)'),
                r'\1 ({prf:ref}`def-hypocoercive-distance`)'
            ),

            # Section 4 - Reference Fragile Gas axioms
            (
                re.compile(r'(complete set of foundational axioms that any valid Fragile Gas must satisfy)'),
                r'\1 ({prf:ref}`def-fragile-gas-axioms`)'
            ),
        ]

        result = content
        added_count = 0

        for pattern, replacement in manual_additions:
            if pattern.search(result):
                result = pattern.sub(replacement, result, count=1)
                added_count += 1

        print(f"Added {added_count} manual references")
        return result


def main():
    """Main execution."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md")

    print("="*70)
    print("Comprehensive Backward Reference Enrichment for 03_cloning.md")
    print("="*70)
    print()

    # Create backup
    print("Creating backup...")
    backup_path = doc_path.with_suffix('.md.backup_comprehensive_backward_refs')
    backup_path.write_text(doc_path.read_text())
    print(f"Backup saved: {backup_path}")
    print()

    # Enrich document
    print("Analyzing document and adding backward references...")
    enricher = BackwardReferenceEnricher(doc_path)
    enriched = enricher.enrich_document()
    print()

    # Add manual references
    print("Adding manually curated references...")
    enriched = enricher.add_manual_references(enriched)
    print()

    # Write result
    print("Writing enriched document...")
    doc_path.write_text(enriched)
    print()

    print("="*70)
    print("ENRICHMENT COMPLETE")
    print("="*70)
    print(f"Document: {doc_path}")
    print(f"Backup: {backup_path}")
    print(f"Total unique backward references: {len(enricher.referenced)}")
    print()


if __name__ == "__main__":
    main()
