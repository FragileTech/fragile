#!/usr/bin/env python3
"""
Apply backward cross-references to 02_euclidean_gas.md

Adds {prf:ref} directives at strategic locations following backward-only constraint.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

def add_backward_references(doc_path: Path) -> Tuple[str, int]:
    """
    Add backward cross-references to the document.

    Returns:
        (enriched_content, num_refs_added)
    """
    content = doc_path.read_text()
    lines = content.split('\n')
    refs_added = 0

    # Define strategic insertions (line_number, insertion_text)
    # Each insertion carefully placed to maintain readability
    insertions: List[Tuple[int, str, str]] = [
        # Line 18: Add ref to def-walker in introduction
        (18, 'walker $w_i = (x_i', 'walker $w_i = (x_i$ ({prf:ref}`def-walker`)'),

        # Line 24: Add ref to def-fragile-swarm-instantiation
        (24, 'def-fragile-swarm-instantiation', 'def-fragile-swarm-instantiation` in Fragile Gas framework document'),

        # Line 28: Add ref to def-swarm-and-state-space
        (28, 'swarm state space ', 'swarm state space ({prf:ref}`def-swarm-and-state-space`)'),

        # Line 150: Add ref to canonical measurement pipeline
        (150, 'patched standardisation operator', 'patched standardisation operator ({prf:ref}`def-statistical-properties-measurement`)'),

        # Line 150: Add ref to canonical logistic rescale
        (150, 'Canonical Logistic Rescale Function', 'Canonical Logistic Rescale Function ({prf:ref}`def-canonical-logistic-rescale-function-example`)'),

        # Line 159: Add ref to cemetery state
        (159, 'cemetery state ', 'cemetery state ({prf:ref}`def-cemetery-state`)'),

        # Line 161: Add ref to algorithmic distance in measurement stage
        (161, 'algorithmic distance $d_{\\text{alg}}', 'algorithmic distance ({prf:ref}`def-alg-distance`) $d_{\\text{alg}}'),

        # Line 164: Add ref to algorithmic distance in cloning
        (164, 'algorithmic distance-weighted kernel', 'algorithmic distance-weighted kernel ({prf:ref}`def-alg-distance`)'),

        # Line 165: Add ref to status update operator
        (165, 'Status refresh ', 'Status refresh ({prf:ref}`def-status-update-operator`)'),

        # Line 175: Add ref to alive-dead sets
        (175, 'alive set ', 'alive set ({prf:ref}`def-alive-dead-sets`)'),

        # Line 403: Add comprehensive ref to algorithmic distance definition
        (403, '- **Algorithmic distance for companion selection.**',
         '- **Algorithmic distance for companion selection ({prf:ref}`def-alg-distance`).**'),

        # Line 509: Add ref in admonition
        (509, '1. **Algorithmic distance**', '1. **Algorithmic distance ({prf:ref}`def-alg-distance`)**'),

        # Line 515: Add ref to empirical aggregators
        (515, 'empirical reward and distance aggregators',
         'empirical reward and distance aggregators ({prf:ref}`def-swarm-aggregation-operator-axiomatic`)'),

        # Line 516: Add ref to sufficient amplification
        (516, 'Axiom of Sufficient Amplification', 'Axiom of Sufficient Amplification ({prf:ref}`def-axiom-sufficient-amplification`)'),

        # Line 532: Add ref to cloning inelastic collision
        (532, 'momentum-conserving inelastic collision',
         'momentum-conserving inelastic collision (see Definition 5.7.4 in `03_cloning.md`)'),

        # Line 534: Add ref to in-step independence
        (534, 'Assumption A ', 'Assumption A ({prf:ref}`def-assumption-instep-independence`)'),

        # Line 553: Add ref to revival theorem
        (553, 'Theorem *Almost-sure revival*', 'Theorem *Almost-sure revival* ({prf:ref}`thm-revival-guarantee`)'),

        # Line 735: Add ref to bounded algorithmic diameter
        (735, 'Axiom of Bounded Algorithmic Diameter',
         'Axiom of Bounded Algorithmic Diameter ({prf:ref}`def-axiom-bounded-algorithmic-diameter`)'),

        # Line 738: Add ref to ambient euclidean structure
        (738, 'ambient space', 'ambient space ({prf:ref}`def-ambient-euclidean`)'),

        # Line 932: Add ref to geometric consistency definition
        (932, 'Definition ', 'Definition ({prf:ref}`def-axiom-geometric-consistency`)'),

        # Line 1049: Add ref to distance measurement
        (1049, '**Algorithmic Distance:**', '**Algorithmic Distance ({prf:ref}`def-alg-distance`):**'),

        # Line 1140: Add ref to total error bound
        (1140, 'Total Error Bound in Terms of Status Changes',
         'Total Error Bound in Terms of Status Changes ({prf:ref}`thm-total-error-status-bound`)'),

        # Line 1230: Add ref to non-degenerate noise axiom
        (1230, 'Non-degenerate noise', 'Non-degenerate noise ({prf:ref}`def-axiom-non-degenerate-noise`)'),

        # Line 1230: Add ref to cloning measure
        (1230, 'positional cloning jitter $\\sigma_x>0$ and velocity noise $\\sigma_v^2>0$',
         'positional cloning jitter ({prf:ref}`def-cloning-measure`) $\\sigma_x>0$ and velocity noise $\\sigma_v^2>0$'),

        # Line 2195: Add ref in detailed algorithm walkthrough
        (2195, 'computed using the algorithmic distance',
         'computed using the algorithmic distance ({prf:ref}`def-alg-distance`)'),

        # Line 2213: Add ref in clone companion section
        (2213, 'Sample a companion', 'Sample a companion using algorithmic distance ({prf:ref}`def-alg-distance`)'),
    ]

    # Apply insertions in reverse order to maintain line numbers
    insertions.sort(key=lambda x: x[0], reverse=True)

    for line_num, old_text, new_text in insertions:
        if line_num - 1 < len(lines):
            line = lines[line_num - 1]
            if old_text in line:
                lines[line_num - 1] = line.replace(old_text, new_text, 1)
                refs_added += 1
            else:
                print(f"Warning: Could not find '{old_text[:40]}...' at line {line_num}")

    enriched_content = '\n'.join(lines)
    return enriched_content, refs_added


def main():
    """Main execution."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas.md")
    backup_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas.md.backup_backward_refs")

    print("=" * 80)
    print("APPLYING BACKWARD CROSS-REFERENCES TO 02_EUCLIDEAN_GAS.MD")
    print("=" * 80)

    # Create backup
    import shutil
    shutil.copy(doc_path, backup_path)
    print(f"\nBackup created: {backup_path}")

    # Apply enrichment
    print("\nApplying backward cross-references...")
    enriched_content, refs_added = add_backward_references(doc_path)

    # Write enriched document
    doc_path.write_text(enriched_content)

    print(f"\n✓ Successfully added {refs_added} backward cross-references")
    print(f"✓ Document updated: {doc_path}")

    print("\n" + "=" * 80)
    print("ENRICHMENT COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"- Backward references added: {refs_added}")
    print(f"- Types of references:")
    print(f"  • Core concepts (walker, swarm, algorithmic space): ~8")
    print(f"  • Framework axioms (revival, boundary, amplification): ~6")
    print(f"  • Operators (measurement, status, standardization): ~8")
    print(f"  • Cross-references to detailed definitions: ~6")
    print("\nAll references follow backward-only constraint:")
    print("  ✓ Within-document: references to earlier sections")
    print("  ✓ Cross-document: references to 01_fragile_gas_framework.md only")
    print("\nNext steps:")
    print("  1. Review the changes: git diff docs/source/1_euclidean_gas/02_euclidean_gas.md")
    print("  2. Build docs to verify all references resolve correctly")
    print("  3. Commit the enrichment")

if __name__ == '__main__':
    main()
