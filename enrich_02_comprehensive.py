#!/usr/bin/env python3
"""
Comprehensive backward cross-reference enrichment for 02_euclidean_gas.md

This pass adds more strategic references to enhance mathematical connectivity.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Additional reference targets for comprehensive enrichment
# Format: (concept_name, search_pattern, ref_label, start_line, end_line, description)
ADDITIONAL_TARGETS = [
    # Core framework structures - first mentions
    ("walker structure", r"walker.*\$w_i\s*=\s*\(x_i", "def-walker", 15, 20, "Walker definition in intro"),
    ("swarm configuration", r"swarm configuration", "def-swarm-and-state-space", 55, 65, "Swarm in mermaid diagram"),
    ("dead walkers", r"dead walkers", "def-alive-dead-sets", 550, 560, "Guaranteed revival context"),

    # Metrics and distances
    ("displacement dispersion", r"[Dd]ispersion distance", "def-n-particle-displacement-metric", 494, 505, "Dispersion distance definition"),
    ("Wasserstein distance", r"Wasserstein", "lem-polishness-and-w2", 715, 725, "Wasserstein continuity bound"),

    # Measurement pipeline
    ("raw reward", r"raw reward", "def-reward-measurement", 2188, 2192, "Raw scores in Stage 2"),
    ("companion selection", r"companion.*for the diversity measurement", "def-distance-to-companion", 2189, 2200, "Companion kernel"),
    ("empirical aggregator", r"empirical.*aggregator", "lem-empirical-aggregator-properties", 515, 520, "Aggregator properties"),

    # Noise and perturbation
    ("Gaussian perturbation", r"Gaussian.*jitter", "def-perturbation-measure", 2236, 2245, "Clone jitter"),
    ("kinetic noise", r"kinetic.*kernel", "def-perturbation-measure", 802, 810, "Valid noise measure"),

    # Axioms - explicit mentions
    ("bounded diameter", r"bounded algorithmic diameter", "def-axiom-bounded-algorithmic-diameter", 732, 736, "Finite algorithmic diameter"),
    ("projection Lipschitz", r"projection.*is.*Lipschitz", "def-axiom-projection-compatibility", 400, 405, "Projection compatibility"),
    ("range-respecting", r"range.*respecting", "def-axiom-range-respecting-mean", 190, 200, "Range-respecting property"),

    # Additional framework concepts
    ("cemetery state", r"cemetery state", "def-cemetery-state", 159, 165, "Cemetery check"),
    ("status refresh", r"status refresh", "def-status-update-operator", 165, 170, "Status update"),
    ("cloning gate", r"Clone/Persist gate", "def-stochastic-threshold-cloning", 164, 168, "Cloning decision"),
]


def find_and_add_reference(lines: List[str], pattern: str, ref_label: str,
                          start: int, end: int, context: str) -> Tuple[bool, int]:
    """
    Find pattern in range and add reference if not already present.

    Returns:
        (success, line_number)
    """
    for i in range(max(0, start), min(end, len(lines))):
        line = lines[i]

        # Skip if already has this specific reference
        if f"{{prf:ref}}`{ref_label}`" in line:
            return (False, -1)

        # Skip if already has a different reference to same text
        if '{prf:ref}' in line and re.search(pattern, line, re.IGNORECASE):
            return (False, -1)

        # Skip code blocks and directives
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith(':::'):
            continue

        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            matched_text = match.group(0)

            # Add reference parenthetically
            replacement = f"{matched_text} ({{prf:ref}}`{ref_label}`)"
            new_line = line.replace(matched_text, replacement, 1)

            if new_line != line:
                lines[i] = new_line
                return (True, i)

    return (False, -1)


def add_comprehensive_references(doc_path: Path) -> dict:
    """
    Add comprehensive backward cross-references.

    Returns:
        Statistics dictionary
    """
    with open(doc_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    stats = {
        'refs_added': 0,
        'failed_targets': 0,
        'successes': [],
        'failures': [],
    }

    print("Processing additional reference targets...")
    print()

    for concept, pattern, ref_label, start, end, description in ADDITIONAL_TARGETS:
        success, line_num = find_and_add_reference(lines, pattern, ref_label, start, end, description)

        if success:
            stats['refs_added'] += 1
            stats['successes'].append((concept, line_num + 1, description))
            print(f"✓ Line {line_num + 1}: {concept} → {ref_label}")
            print(f"  Context: {description}")
        else:
            stats['failed_targets'] += 1
            stats['failures'].append((concept, description))

    # Write enriched document
    enriched_content = '\n'.join(lines)
    with open(doc_path, 'w') as f:
        f.write(enriched_content)

    return stats


def main():
    doc_path = Path('docs/source/1_euclidean_gas/02_euclidean_gas.md')

    if not doc_path.exists():
        print(f"Error: Document not found at {doc_path}")
        return 1

    print("=" * 70)
    print("COMPREHENSIVE BACKWARD REFERENCE ENRICHMENT - Pass 2")
    print("=" * 70)
    print()

    # Create backup of current version
    import shutil
    backup_path = doc_path.with_suffix('.md.backup_comprehensive')
    shutil.copy(doc_path, backup_path)
    print(f"✓ Backup created: {backup_path}")
    print()

    # Add references
    stats = add_comprehensive_references(doc_path)

    # Report
    print()
    print("=" * 70)
    print("ENRICHMENT REPORT - Pass 2")
    print("=" * 70)
    print()
    print(f"New backward references added: {stats['refs_added']}")
    print(f"Targets not matched: {stats['failed_targets']}")
    print()

    if stats['successes']:
        print("Successfully added references:")
        for concept, line_num, desc in stats['successes']:
            print(f"  • {concept} (line {line_num})")
            print(f"    → {desc}")

    if stats['failures']:
        print()
        print("Targets not found (may already be referenced):")
        for concept, desc in stats['failures']:
            print(f"  • {concept}: {desc}")

    print()
    print(f"✓ Document updated: {doc_path}")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
