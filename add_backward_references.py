#!/usr/bin/env python3
"""
Add strategic backward cross-references to improve document connectivity.

Priority targets (145 "source" entities):
- Axioms that are foundational but not referenced
- Key definitions used in later proofs
- Important lemmas and theorems
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Priority source entities to connect (sample from the 145 identified)
PRIORITY_SOURCES = {
    # Axioms
    "axiom-bounded-deviation-variance",
    "axiom-bounded-relative-collapse",
    "axiom-bounded-variance-production",
    "axiom-environmental-richness",
    "axiom-geometric-consistency",
    "axiom-instep-independence",
    "axiom-range-respecting-mean",
    "axiom-raw-value-mean-square-continuity",
    "axiom-sufficient-amplification",
    "axiom-boundary-regularity",
    "axiom-boundary-smoothness",
    "axiom-reward-regularity",
    "axiom-guaranteed-revival",
    "axiom-bounded-algorithmic-diameter",
    "axiom-projection-compatibility",
    "axiom-non-degenerate-noise",
    "axiom-position-only-status-margin",
    "axiom-well-behaved-rescale",
    "axiom-bounded-measurement-variance",
    "axiom-bounded-second-moment-perturbation",

    # Key definitions
    "def-walker",
    "def-swarm-and-state-space",
    "def-alive-dead-sets",
    "def-valid-state-space",
    "def-displacement-components",
    "def-valid-noise-measure",
    "def-reward-measurement",
    "def-perturbation-measure",
    "def-cloning-measure",
    "def-algorithmic-space",
    "def-algorithmic-distance",
    "def-swarm-aggregation",
    "def-smoothed-gaussian-measure",
    "def-cemetery-state-measure",
    "def-companion-selection",
    "def-smooth-piecewise-rescale",
    "def-canonical-logistic-rescale",
    "def-raw-value-operator",
    "def-distance-to-companion",
    "def-standardization-operator",
    "def-statistical-properties",
    "def-expected-squared-value-error",
    "def-expected-squared-structural-error",
    "def-value-error-coefficients",
    "def-structural-error-coefficients",
    "def-rescaled-potential-operator",
    "def-swarm-potential-assembly",
    "def-perturbation-operator",
    "def-perturbation-fluctuation-bounds",
    "def-status-update-operator",
    "def-cloning-score",
    "def-stochastic-threshold-cloning",
    "def-total-expected-cloning-action",
    "def-cloning-probability-function",
    "def-expected-cloning-action",
    "def-swarm-update-procedure",
    "def-final-status-change-coefficients",
    "def-wasserstein-2-output",
    "def-fragile-swarm-instantiation",
    "def-fragile-gas-algorithm",

    # Important lemmas
    "lem-boundary-uniform-ball",
    "lem-cloning-probability-lipschitz",
    "lem-empirical-moments-lipschitz",
    "lem-final-status-change-bound",
    "lem-potential-boundedness",
    "lem-rescale-monotonicity",
    "lem-stats-structural-continuity",
    "lem-stats-value-continuity",

    # Key theorems
    "thm-asymptotic-std-dev-structural-continuity",
    "thm-canonical-logistic-validity",
    "thm-expected-raw-distance-bound",
    "thm-forced-activity",
    "thm-revival-guarantee",
}

# Mapping of concepts to their labels (for semantic matching)
CONCEPT_TO_LABEL = {
    # Core concepts
    "walker": "def-walker",
    "walkers": "def-walker",
    "swarm": "def-swarm-and-state-space",
    "swarm state": "def-swarm-and-state-space",
    "alive set": "def-alive-dead-sets",
    "dead set": "def-alive-dead-sets",
    "alive and dead": "def-alive-dead-sets",
    "valid state space": "def-valid-state-space",

    # Axioms (frequently mentioned)
    "bounded algorithmic diameter": "axiom-bounded-algorithmic-diameter",
    "boundary regularity": "axiom-boundary-regularity",
    "boundary smoothness": "axiom-boundary-smoothness",
    "reward regularity": "axiom-reward-regularity",
    "guaranteed revival": "axiom-guaranteed-revival",
    "environmental richness": "axiom-environmental-richness",
    "sufficient amplification": "axiom-sufficient-amplification",
    "geometric consistency": "axiom-geometric-consistency",
    "bounded variance production": "axiom-bounded-variance-production",
    "bounded deviation": "axiom-bounded-deviation-variance",
    "bounded relative collapse": "axiom-bounded-relative-collapse",
    "range-respecting mean": "axiom-range-respecting-mean",
    "well-behaved rescale": "axiom-well-behaved-rescale",
    "non-degenerate noise": "axiom-non-degenerate-noise",

    # Operators and measures
    "algorithmic distance": "def-algorithmic-distance",
    "algorithmic space": "def-algorithmic-space",
    "swarm aggregation": "def-swarm-aggregation",
    "rescaled potential": "def-rescaled-potential-operator",
    "fitness potential": "def-swarm-potential-assembly",
    "perturbation operator": "def-perturbation-operator",
    "cloning score": "def-cloning-score",
    "cloning probability": "def-cloning-probability-function",
    "stochastic threshold cloning": "def-stochastic-threshold-cloning",
    "standardization operator": "def-standardization-operator",
    "raw value operator": "def-raw-value-operator",

    # Key structures
    "displacement components": "def-displacement-components",
    "companion selection": "def-companion-selection",
    "cemetery state": "def-cemetery-state-measure",
    "smoothed gaussian": "def-smoothed-gaussian-measure",
    "canonical logistic": "def-canonical-logistic-rescale",

    # Error terms
    "value error": "def-expected-squared-value-error",
    "structural error": "def-expected-squared-structural-error",
    "standardization error": "def-standardization-operator",
}


def find_entity_locations(content: str) -> Dict[str, Tuple[int, int]]:
    """Find line ranges for all entities in the document."""
    lines = content.split('\n')
    entities = {}
    current_entity = None
    current_start = None

    for i, line in enumerate(lines, 1):
        # Start of entity
        if match := re.match(r'::::\{prf:\w+\}', line):
            if current_entity and current_start:
                entities[current_entity] = (current_start, i - 1)
            current_entity = None
            current_start = i

        # Label line
        elif current_start and line.startswith(':label:'):
            label = line.split(':label:')[1].strip()
            current_entity = label

        # End of entity
        elif line.strip() == '::::' and current_entity:
            entities[current_entity] = (current_start, i)
            current_entity = None
            current_start = None

    return entities


def already_has_reference(text: str, label: str) -> bool:
    """Check if text already references a label."""
    pattern = rf'\{{prf:ref\}}`{label}`'
    return bool(re.search(pattern, text))


def add_reference_to_text(text: str, concept: str, label: str) -> Tuple[str, bool]:
    """
    Add a reference after the first mention of a concept.
    Returns (modified_text, was_modified).
    """
    # Don't add if already present
    if already_has_reference(text, label):
        return text, False

    # Find first occurrence of concept (case insensitive, word boundary)
    pattern = re.compile(rf'\b{re.escape(concept)}\b', re.IGNORECASE)
    match = pattern.search(text)

    if not match:
        return text, False

    # Insert reference after the matched concept
    insert_pos = match.end()

    # Check if we're in a display math block ($$...$$)
    before_match = text[:insert_pos]
    after_match = text[insert_pos:]

    # Count $$ before and after to determine if we're in math
    dollars_before = before_match.count('$$')
    if dollars_before % 2 == 1:  # Inside math block
        return text, False

    # Add reference with proper spacing
    ref = f" ({{prf:ref}}`{label}`)"
    modified = text[:insert_pos] + ref + after_match

    return modified, True


def process_document(file_path: Path) -> Tuple[str, Dict[str, int]]:
    """
    Process document to add backward references.
    Returns (modified_content, statistics).
    """
    content = file_path.read_text()
    lines = content.split('\n')

    # Find all entity locations
    entities = find_entity_locations(content)

    print(f"Found {len(entities)} entities in document")

    # Track modifications
    stats = {
        'refs_added': 0,
        'entities_modified': 0,
        'by_type': {},
    }

    modified_lines = lines.copy()

    # Process each entity
    for label, (start_line, end_line) in sorted(entities.items(), key=lambda x: x[1][0]):
        # Only process entities after line 500 (to ensure backward-only references)
        if start_line < 500:
            continue

        # Get entity content
        entity_lines = modified_lines[start_line-1:end_line]
        entity_text = '\n'.join(entity_lines)

        # Try to add references for each concept
        modified = False
        for concept, target_label in CONCEPT_TO_LABEL.items():
            # Only add backward references (target must be before current)
            if target_label not in entities:
                continue

            target_start, _ = entities[target_label]
            if target_start >= start_line:
                continue  # Would be forward reference

            # Skip if target is not in priority sources
            if target_label not in PRIORITY_SOURCES:
                continue

            # Try to add reference
            new_text, was_modified = add_reference_to_text(entity_text, concept, target_label)

            if was_modified:
                entity_text = new_text
                modified = True
                stats['refs_added'] += 1

                # Track by entity type
                entity_type = target_label.split('-')[0]
                stats['by_type'][entity_type] = stats['by_type'].get(entity_type, 0) + 1

        # Update lines if modified
        if modified:
            new_entity_lines = entity_text.split('\n')
            modified_lines[start_line-1:end_line] = new_entity_lines
            stats['entities_modified'] += 1

    return '\n'.join(modified_lines), stats


def main():
    """Main execution."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    print("Processing document for backward cross-references...")
    print(f"Document: {doc_path}")
    print(f"Priority sources: {len(PRIORITY_SOURCES)}")
    print(f"Concept mappings: {len(CONCEPT_TO_LABEL)}")
    print()

    # Create backup
    backup_path = doc_path.with_suffix('.md.backup2')
    backup_path.write_text(doc_path.read_text())
    print(f"Backup created: {backup_path}")
    print()

    # Process document
    modified_content, stats = process_document(doc_path)

    # Write modified content
    doc_path.write_text(modified_content)

    # Report statistics
    print("\n" + "="*60)
    print("CROSS-REFERENCE ENRICHMENT COMPLETE")
    print("="*60)
    print(f"Total references added: {stats['refs_added']}")
    print(f"Entities modified: {stats['entities_modified']}")
    print()
    print("References added by target type:")
    for entity_type, count in sorted(stats['by_type'].items()):
        print(f"  {entity_type}: {count}")
    print()
    print(f"Modified document saved: {doc_path}")


if __name__ == "__main__":
    main()
