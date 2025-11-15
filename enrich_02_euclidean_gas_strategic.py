#!/usr/bin/env python3
"""
Strategic Backward Cross-Reference Enrichment for 02_euclidean_gas.md

This script adds high-value backward cross-references to concepts from 01_fragile_gas_framework.md
based on explicit mentions and mathematical dependencies.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Set

# Load framework definitions from glossary analysis
FRAMEWORK_REFS = {
    # Core structural definitions
    "walker": "def-walker",
    "swarm": "def-swarm-and-state-space",
    "alive set": "def-alive-dead-sets",
    "dead set": "def-alive-dead-sets",
    "valid state space": "def-valid-state-space",
    "cemetery state": "def-cemetery-state-measure",

    # Axioms - Viability
    "Guaranteed Revival": "axiom-guaranteed-revival",
    "Axiom of Guaranteed Revival": "axiom-guaranteed-revival",
    "Almost-sure revival": "thm-revival-guarantee",
    "Boundary Regularity": "axiom-boundary-regularity",
    "Axiom of Boundary Regularity": "axiom-boundary-regularity",
    "Boundary Smoothness": "axiom-boundary-smoothness",
    "Axiom of Boundary Smoothness": "axiom-boundary-smoothness",

    # Axioms - Environmental
    "Environmental Richness": "axiom-environmental-richness",
    "Axiom of Environmental Richness": "axiom-environmental-richness",
    "Reward Regularity": "axiom-reward-regularity",
    "Axiom of Reward Regularity": "axiom-reward-regularity",
    "Projection Compatibility": "axiom-projection-compatibility",
    "Axiom of Projection Compatibility": "axiom-projection-compatibility",
    "Bounded Algorithmic Diameter": "axiom-bounded-algorithmic-diameter",
    "Axiom of Bounded Algorithmic Diameter": "axiom-bounded-algorithmic-diameter",
    "Range-Respecting Mean": "axiom-range-respecting-mean",
    "Axiom of Range-Respecting Mean": "axiom-range-respecting-mean",

    # Axioms - Algorithmic
    "In-Step Independence": "axiom-instep-independence",
    "Assumption A": "axiom-instep-independence",
    "Sufficient Amplification": "axiom-sufficient-amplification",
    "Axiom of Sufficient Amplification": "axiom-sufficient-amplification",
    "Non-Degenerate Noise": "axiom-non-degenerate-noise",
    "Axiom of Non-Degenerate Noise": "axiom-non-degenerate-noise",
    "Geometric Consistency": "axiom-geometric-consistency",
    "Axiom of Geometric Consistency": "axiom-geometric-consistency",

    # Operators and measures
    "algorithmic space": "def-algorithmic-space-generic",
    "algorithmic distance": "def-alg-distance",
    "swarm aggregation": "def-swarm-aggregation-operator-axiomatic",
    "perturbation measure": "def-perturbation-measure",
    "cloning measure": "def-cloning-measure",
    "status update operator": "def-status-update-operator",
    "Fragile Gas Algorithm": "def-fragile-gas-algorithm",
    "Fragile Swarm instantiation": "def-fragile-swarm-instantiation",

    # Measurement pipeline
    "statistical properties": "def-statistical-properties-measurement",
    "patched standardisation": "def-standardization-operator-n-dimensional",
    "Canonical Logistic Rescale": "def-canonical-logistic-rescale-function-example",
    "rescale function": "def-axiom-rescale-function",

    # Distance and metrics
    "displacement metric": "def-n-particle-displacement-metric",
    "metric quotient": "def-metric-quotient",
    "companion selection": "def-companion-selection-measure",
}

def find_existing_refs(content: str) -> Set[str]:
    """Extract all existing {prf:ref} labels."""
    pattern = r'\{prf:ref\}`([^`]+)`'
    return set(re.findall(pattern, content))

def find_enrichment_opportunities(content: str) -> List[Tuple[int, str, str, str]]:
    """
    Find lines where framework concepts are mentioned without cross-references.

    Returns: List of (line_number, concept, label, context)
    """
    opportunities = []
    existing_refs = find_existing_refs(content)
    lines = content.split('\n')

    for i, line in enumerate(lines, start=1):
        # Skip lines that already have references
        if '{prf:ref}' in line:
            continue

        # Skip code blocks
        if line.strip().startswith('```') or line.strip().startswith('$'):
            continue

        # Check each framework concept
        for concept, label in FRAMEWORK_REFS.items():
            # Case-insensitive search for concept mentions
            if re.search(r'\b' + re.escape(concept) + r'\b', line, re.IGNORECASE):
                opportunities.append((i, concept, label, line.strip()))

    return opportunities

def should_add_reference(line: str, concept: str) -> bool:
    """
    Determine if a reference should be added to this line.

    Criteria:
    - Not in a directive title (:::{prf:...})
    - Not already referenced
    - In meaningful context (not just passing mention)
    """
    # Skip directive titles
    if line.strip().startswith(':::'):
        return False

    # Skip if already has a reference to this concept
    if '{prf:ref}' in line:
        return False

    # Skip comments and metadata
    if line.strip().startswith('#') or line.strip().startswith(':'):
        return False

    return True

def add_reference_to_line(line: str, concept: str, label: str) -> str:
    """Add a {prf:ref} to the first occurrence of the concept in the line."""
    # Find the concept (case-insensitive)
    pattern = r'\b(' + re.escape(concept) + r')\b'
    match = re.search(pattern, line, re.IGNORECASE)

    if not match:
        return line

    # Extract the matched text (preserves case)
    matched_text = match.group(1)
    start, end = match.span(1)

    # Construct the replacement with reference
    # Check if we need parenthetical or inline style
    if end < len(line) and line[end] in ',.;:)':
        # Parenthetical style: concept ({prf:ref}`label`)
        replacement = f"{matched_text} ({{prf:ref}}`{label}`)"
    else:
        # Inline style: incorporate smoothly
        replacement = f"{matched_text} ({{prf:ref}}`{label}`)"

    # Replace only the first occurrence
    return line[:start] + replacement + line[end:]

def enrich_document(input_path: Path, output_path: Path, dry_run: bool = False):
    """Main enrichment function."""

    print(f"Reading document: {input_path}")
    content = input_path.read_text()

    print("Analyzing enrichment opportunities...")
    opportunities = find_enrichment_opportunities(content)

    print(f"\nFound {len(opportunities)} potential enrichment locations")

    # Filter to high-value references only
    high_value_opportunities = []
    for line_num, concept, label, context in opportunities:
        lines = content.split('\n')
        line = lines[line_num - 1]

        if should_add_reference(line, concept):
            high_value_opportunities.append((line_num, concept, label, context))

    print(f"Filtered to {len(high_value_opportunities)} high-value references")

    if dry_run:
        print("\nDRY RUN - Would add references at:")
        for line_num, concept, label, context in high_value_opportunities[:20]:
            print(f"  Line {line_num}: {concept} -> {label}")
            print(f"    Context: {context[:80]}...")
        return

    # Apply enrichments
    lines = content.split('\n')
    modifications = 0

    for line_num, concept, label, context in high_value_opportunities:
        line_idx = line_num - 1
        original_line = lines[line_idx]
        enriched_line = add_reference_to_line(original_line, concept, label)

        if enriched_line != original_line:
            lines[line_idx] = enriched_line
            modifications += 1

    # Write enriched content
    enriched_content = '\n'.join(lines)
    output_path.write_text(enriched_content)

    print(f"\nEnrichment complete:")
    print(f"  - Modifications made: {modifications}")
    print(f"  - Output written to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add backward cross-references to 02_euclidean_gas.md")
    parser.add_argument("--input", type=Path,
                       default=Path("docs/source/1_euclidean_gas/02_euclidean_gas.md"),
                       help="Input markdown file")
    parser.add_argument("--output", type=Path,
                       default=Path("docs/source/1_euclidean_gas/02_euclidean_gas.md"),
                       help="Output markdown file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be added without modifying file")

    args = parser.parse_args()

    enrich_document(args.input, args.output, args.dry_run)
