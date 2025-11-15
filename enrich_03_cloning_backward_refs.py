#!/usr/bin/env python3
"""
Add backward cross-references to 03_cloning.md

This script enriches the cloning document by adding {prf:ref} directives
to concepts defined earlier in documents 01 and 02.

Strategy:
1. Read the document
2. Identify key concepts that should reference earlier definitions
3. Add backward references maintaining natural text flow
4. NEVER add forward references
"""

import re
from pathlib import Path

# Key concepts from 01_fragile_gas_framework.md and their labels
FRAMEWORK_CONCEPTS = {
    # Core definitions
    "walker": "def-walker",
    "swarm": "def-swarm-and-state-space",
    "alive set": "def-alive-dead-sets",
    "dead set": "def-alive-dead-sets",
    "valid state space": "def-valid-state-space",
    "Polish metric space": "def-valid-state-space",

    # Axioms
    "Axiom of Guaranteed Revival": "def-axiom-guaranteed-revival",
    "Boundary Regularity": "def-axiom-boundary-regularity",
    "Boundary Smoothness": "def-axiom-boundary-smoothness",
    "Environmental Richness": "def-axiom-environmental-richness",
    "Reward Regularity": "def-axiom-reward-regularity",
    "Projection compatibility": "def-axiom-projection-compatibility",
    "Bounded Algorithmic Diameter": "def-axiom-bounded-algorithmic-diameter",

    # Metrics
    "N-Particle Displacement Pseudometric": "def-n-particle-displacement-metric",
    "Wasserstein": "lem-polishness-and-w2",
    "displacement components": "def-displacement-components",

    # Reference measures
    "Reference Noise": "def-reference-measures",
    "heat kernel": "def-reference-measures",

    # QSD concepts
    "Quasi-Stationary Distribution": "def-qsd",
    "QSD": "def-qsd",
}

# Key concepts from 02_euclidean_gas.md
EUCLIDEAN_CONCEPTS = {
    "Euclidean Gas": "def-euclidean-gas",
    "BAOAB integrator": "def-baoab-integrator",
    "Langevin dynamics": "def-langevin-operator",
    "kinetic operator": "def-kinetic-operator",
}


def add_backward_references(content: str) -> str:
    """Add backward cross-references to the document."""

    # Strategy: Add references at first mention of key concepts
    # Track which concepts we've already referenced to avoid over-referencing
    referenced = set()

    lines = content.split('\n')
    result_lines = []

    for i, line in enumerate(lines):
        modified_line = line

        # Skip lines that are already references or inside code blocks
        if '{prf:ref}' in line or line.strip().startswith('```') or line.strip().startswith('$$'):
            result_lines.append(line)
            continue

        # Skip lines inside directives (labels, etc.)
        if ':label:' in line or line.strip().startswith('::::'):
            result_lines.append(line)
            continue

        # Add references to key framework concepts
        for concept, label in FRAMEWORK_CONCEPTS.items():
            # Only add if not already referenced and concept appears
            if label not in referenced and concept.lower() in line.lower():
                # Add reference after first occurrence
                # Be careful to maintain natural flow
                pattern = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
                match = pattern.search(modified_line)
                if match:
                    # Add reference parenthetically after the concept
                    pos = match.end()
                    ref = f" ({{prf:ref}}`{label}`)"
                    modified_line = modified_line[:pos] + ref + modified_line[pos:]
                    referenced.add(label)
                    break  # Only one reference per line

        # Add references to Euclidean Gas concepts
        for concept, label in EUCLIDEAN_CONCEPTS.items():
            if label not in referenced and concept.lower() in line.lower():
                pattern = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
                match = pattern.search(modified_line)
                if match:
                    pos = match.end()
                    ref = f" ({{prf:ref}}`{label}`)"
                    modified_line = modified_line[:pos] + ref + modified_line[pos:]
                    referenced.add(label)
                    break

        result_lines.append(modified_line)

    return '\n'.join(result_lines)


def add_strategic_references(content: str) -> str:
    """
    Add strategic backward references at key locations.

    This function targets specific mathematical entities and adds
    references to foundational concepts from docs 01 and 02.
    """

    # Specific targeted additions for better connectivity
    replacements = [
        # In the TLDR - reference Safe Harbor axiom
        (
            r'(see \{prf:ref\}`axiom-safe-harbor`\))',
            r'\1'  # Already has reference, keep it
        ),

        # In Section 1.2 - reference QSD
        (
            r'(Quasi-Stationary Distributions \(QSDs\))',
            r'\1 ({prf:ref}`def-qsd`)'
        ),

        # In Section 2.1 - reference walker definition from framework
        (
            r'(The fundamental unit of the system is the walker)',
            r'\1 ({prf:ref}`def-walker`)'
        ),

        # In Section 4.1 - reference ambient euclidean structure
        (
            r'(\*\*\(Axiom EG-1\): Lipschitz Regularity of Environmental Fields\*\*)',
            r'\1'  # Will add ref in axiom content
        ),

        # Reference Wasserstein distance when first introduced
        (
            r'(Wasserstein distance)',
            r'\1 ({prf:ref}`lem-polishness-and-w2`)'
        ),
    ]

    result = content
    for pattern, replacement in replacements:
        # Only replace first occurrence
        result = re.sub(pattern, replacement, result, count=1)

    return result


def main():
    """Main execution function."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md")

    print("Reading 03_cloning.md...")
    content = doc_path.read_text()

    print("Creating backup...")
    backup_path = doc_path.with_suffix('.md.backup_backward_refs')
    backup_path.write_text(content)
    print(f"Backup saved to {backup_path}")

    print("Adding backward cross-references...")
    enriched = add_backward_references(content)

    print("Adding strategic references...")
    enriched = add_strategic_references(enriched)

    print("Writing enriched document...")
    doc_path.write_text(enriched)

    print("Done! Backward cross-references added to 03_cloning.md")
    print(f"Original saved to: {backup_path}")


if __name__ == "__main__":
    main()
