#!/usr/bin/env python3
"""
Final comprehensive backward cross-reference enrichment.

This script adds strategic references to improve connectivity for the 145 "source"
entities identified in the connectivity analysis. It focuses on:

1. Adding refs in proofs that use earlier results
2. Connecting theorems to the axioms they rely on
3. Linking definitions to their foundational concepts
4. Connecting error bounds to their component definitions

The goal is to add 100-200 high-value backward references.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


# Extended priority mappings: concept → label
BACKWARD_REF_TARGETS = {
    # Core framework (refs should be pervasive)
    "walker": "def-walker",
    "swarm": "def-swarm-and-state-space",
    "swarm state": "def-swarm-and-state-space",
    "alive set": "def-alive-dead-sets",
    "dead set": "def-alive-dead-sets",
    "alive walkers": "def-alive-dead-sets",
    "dead walkers": "def-alive-dead-sets",
    "valid state space": "def-valid-state-space",

    # Foundational axioms (connect to proofs that use them)
    "Axiom of Boundary Regularity": "axiom-boundary-regularity",
    "boundary regularity": "axiom-boundary-regularity",
    "Axiom of Boundary Smoothness": "axiom-boundary-smoothness",
    "boundary smoothness": "axiom-boundary-smoothness",
    "Axiom of Reward Regularity": "axiom-reward-regularity",
    "reward regularity": "axiom-reward-regularity",
    "Axiom of Guaranteed Revival": "axiom-guaranteed-revival",
    "guaranteed revival": "axiom-guaranteed-revival",
    "Axiom of Environmental Richness": "axiom-environmental-richness",
    "environmental richness": "axiom-environmental-richness",
    "Axiom of Sufficient Amplification": "axiom-sufficient-amplification",
    "sufficient amplification": "axiom-sufficient-amplification",
    "Axiom of Geometric Consistency": "axiom-geometric-consistency",
    "geometric consistency": "axiom-geometric-consistency",
    "Axiom of Bounded Variance Production": "axiom-bounded-variance-production",
    "bounded variance production": "axiom-bounded-variance-production",
    "Axiom of Bounded Relative Collapse": "axiom-bounded-relative-collapse",
    "bounded relative collapse": "axiom-bounded-relative-collapse",
    "Axiom of Bounded Deviation": "axiom-bounded-deviation-variance",
    "bounded deviation": "axiom-bounded-deviation-variance",
    "Axiom of Range-Respecting Mean": "axiom-range-respecting-mean",
    "range-respecting mean": "axiom-range-respecting-mean",
    "Axiom of Well-Behaved Rescale": "axiom-well-behaved-rescale",
    "well-behaved rescale": "axiom-well-behaved-rescale",
    "Axiom of Non-Degenerate Noise": "axiom-non-degenerate-noise",
    "non-degenerate noise": "axiom-non-degenerate-noise",
    "Axiom of Bounded Measurement Variance": "axiom-bounded-measurement-variance",
    "bounded measurement variance": "axiom-bounded-measurement-variance",
    "Axiom of Position-Only Status Margin": "axiom-position-only-status-margin",
    "position-only status margin": "axiom-position-only-status-margin",
    "Axiom of Bounded Algorithmic Diameter": "axiom-bounded-algorithmic-diameter",
    "bounded algorithmic diameter": "axiom-bounded-algorithmic-diameter",
    "Axiom of Projection Compatibility": "axiom-projection-compatibility",
    "projection compatibility": "axiom-projection-compatibility",
    "Axiom of Conditional Product Structure": "axiom-conditional-product-structure",
    "conditional product structure": "axiom-conditional-product-structure",

    # Key operators (connect uses to definitions)
    "algorithmic distance": "def-algorithmic-distance",
    "algorithmic space": "def-algorithmic-space",
    "swarm aggregation": "def-swarm-aggregation",
    "aggregation operator": "def-swarm-aggregation",
    "displacement components": "def-displacement-components",
    "companion selection": "def-companion-selection",
    "standardization operator": "def-standardization-operator",
    "raw value operator": "def-raw-value-operator",
    "perturbation operator": "def-perturbation-operator",
    "status update operator": "def-status-update-operator",
    "cloning score": "def-cloning-score",
    "cloning probability function": "def-cloning-probability-function",
    "expected cloning action": "def-expected-cloning-action",
    "total expected cloning action": "def-total-expected-cloning-action",
    "rescaled potential": "def-rescaled-potential-operator",
    "swarm potential": "def-swarm-potential-assembly",
    "fitness potential": "def-swarm-potential-assembly",

    # Key structures
    "smoothed Gaussian": "def-smoothed-gaussian-measure",
    "cemetery state": "def-cemetery-state-measure",
    "valid noise measure": "def-valid-noise-measure",
    "smooth piecewise rescale": "def-smooth-piecewise-rescale",
    "canonical logistic": "def-canonical-logistic-rescale",
    "reward measurement": "def-reward-measurement",
    "perturbation measure": "def-perturbation-measure",
    "cloning measure": "def-cloning-measure",
    "distance-to-companion": "def-distance-to-companion",
    "statistical properties": "def-statistical-properties",

    # Error decomposition terms (connect to definitions)
    "value error": "def-expected-squared-value-error",
    "structural error": "def-expected-squared-structural-error",
    "standardization error": "def-standardization-operator",
    "value error coefficients": "def-value-error-coefficients",
    "structural error coefficients": "def-structural-error-coefficients",
    "final status change": "def-final-status-change-coefficients",
    "perturbation fluctuation": "def-perturbation-fluctuation-bounds",

    # Important lemmas (connect to uses)
    "empirical moments": "lem-empirical-moments-lipschitz",
    "potential boundedness": "lem-potential-boundedness",
    "Lipschitz continuity of the fitness potential": "lem-lipschitz-fitness-potential",
    "cloning probability Lipschitz": "lem-cloning-probability-lipschitz",
    "uniform ball boundary": "lem-boundary-uniform-ball",
    "rescale monotonicity": "lem-rescale-monotonicity",
    "smooth rescale Lipschitz": "lem-smooth-rescale-lipschitz",
    "stats value continuity": "lem-stats-value-continuity",
    "stats structural continuity": "lem-stats-structural-continuity",

    # Key theorems (connect to discussions)
    "Theorem of Forced Activity": "thm-forced-activity",
    "forced activity": "thm-forced-activity",
    "Theorem of Guaranteed Revival": "thm-revival-guarantee",
    "revival guarantee": "thm-revival-guarantee",
    "canonical logistic validity": "thm-canonical-logistic-validity",
    "asymptotic behavior": "thm-mse-asymptotic-behavior",
    "total error bound": "thm-total-error-status-changes",
    "expected raw distance bound": "thm-expected-raw-distance-bound",
    "mean-square continuity": "thm-raw-value-mean-square-continuity",
}


def load_document(path: Path) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """Load document and extract entity boundaries."""
    lines = path.read_text().split('\n')

    entities = {}
    current_start = None
    current_label = None

    for i, line in enumerate(lines):
        if re.match(r'::::\{prf:\w+\}', line):
            current_start = i
            current_label = None
        elif current_start is not None and line.startswith(':label:'):
            current_label = line.split(':label:')[1].strip()
        elif line.strip() == '::::' and current_label:
            entities[current_label] = (current_start, i)
            current_start = None
            current_label = None

    return lines, entities


def should_add_reference(line: str, concept: str, label: str) -> bool:
    """Check if reference should be added to this line."""
    # Already has this reference
    if f"{{prf:ref}}`{label}`" in line:
        return False

    # Too many existing references (avoid clutter)
    if line.count('{prf:ref}') >= 4:
        return False

    # In display math block
    if line.strip().startswith('$$') or line.strip().endswith('$$'):
        return False

    # Line is a label or directive
    if line.strip().startswith(':') or line.strip().startswith('::::'):
        return False

    # Concept appears in line
    if not re.search(rf'\b{re.escape(concept)}\b', line, re.IGNORECASE):
        return False

    return True


def add_reference_to_line(line: str, concept: str, label: str) -> str:
    """Add {prf:ref} after first occurrence of concept."""
    # Find concept
    pattern = re.compile(rf'\b{re.escape(concept)}\b', re.IGNORECASE)
    match = pattern.search(line)

    if not match:
        return line

    # Check if in inline math
    before = line[:match.end()]
    if '$' in before:
        dollar_count = before.count('$')
        if dollar_count % 2 == 1:  # Inside math
            return line

    # Insert reference
    insert_pos = match.end()
    ref = f" ({{prf:ref}}`{label}`)"

    return line[:insert_pos] + ref + line[insert_pos:]


def enrich_document(lines: List[str], entities: Dict[str, Tuple[int, int]]) -> Tuple[List[str], Dict]:
    """Add backward references throughout document."""
    modified_lines = lines.copy()

    stats = {
        'refs_added': 0,
        'lines_modified': 0,
        'entities_modified': set(),
        'refs_by_target': defaultdict(int),
        'refs_by_concept': defaultdict(int),
    }

    # For each target entity
    for concept, target_label in BACKWARD_REF_TARGETS.items():
        if target_label not in entities:
            continue

        target_start, target_end = entities[target_label]

        # Look in all later entities
        for entity_label, (entity_start, entity_end) in entities.items():
            # Only backward references
            if entity_start <= target_end:
                continue

            # Process lines in entity
            refs_added_this_entity = 0

            for i in range(entity_start, entity_end + 1):
                line = modified_lines[i]

                if should_add_reference(line, concept, target_label):
                    new_line = add_reference_to_line(line, concept, target_label)

                    if new_line != line:
                        modified_lines[i] = new_line
                        stats['refs_added'] += 1
                        stats['lines_modified'] += 1
                        stats['entities_modified'].add(entity_label)
                        stats['refs_by_target'][target_label] += 1
                        stats['refs_by_concept'][concept] += 1
                        refs_added_this_entity += 1

                        # Limit refs per entity for this concept
                        if refs_added_this_entity >= 2:
                            break

    stats['entities_modified'] = len(stats['entities_modified'])

    return modified_lines, stats


def main():
    """Main execution."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    print("=" * 80)
    print("FINAL COMPREHENSIVE BACKWARD CROSS-REFERENCE ENRICHMENT")
    print("=" * 80)
    print()
    print(f"Document: {doc_path.name}")
    print(f"Target concepts: {len(BACKWARD_REF_TARGETS)}")
    print()

    # Load
    print("Loading document...")
    lines, entities = load_document(doc_path)
    print(f"  Lines: {len(lines):,}")
    print(f"  Entities: {len(entities)}")
    print()

    # Backup
    backup_path = doc_path.parent / "01_fragile_gas_framework_backup_final.md"
    backup_path.write_text('\n'.join(lines))
    print(f"Backup: {backup_path.name}")
    print()

    # Enrich
    print("Enriching with backward references...")
    modified_lines, stats = enrich_document(lines, entities)

    # Save
    doc_path.write_text('\n'.join(modified_lines))

    # Report
    print()
    print("=" * 80)
    print("ENRICHMENT COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Total backward references added: {stats['refs_added']}")
    print(f"✓ Lines modified: {stats['lines_modified']}")
    print(f"✓ Entities modified: {stats['entities_modified']}")
    print()

    print("Top 20 most referenced labels:")
    for label, count in sorted(stats['refs_by_target'].items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {label}: +{count}")
    print()

    print("Top 15 concepts that added refs:")
    for concept, count in sorted(stats['refs_by_concept'].items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  '{concept}': {count}")
    print()

    print(f"✓ Saved: {doc_path}")


if __name__ == "__main__":
    main()
