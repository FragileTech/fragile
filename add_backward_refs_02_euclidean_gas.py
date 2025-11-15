#!/usr/bin/env python3
"""
Add backward cross-references to 02_euclidean_gas.md

This script follows the backward-only temporal ordering principle:
- Only adds references to concepts defined EARLIER
- References 01_fragile_gas_framework.md (document 01 < document 02)
- Never adds forward references to later sections or documents

Strategy:
1. Add references at FIRST significant mention of framework concepts
2. Focus on high-value locations (definitions, theorems, algorithm steps)
3. Avoid over-referencing (max ~1 ref per concept per major section)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Mapping of concepts → framework labels (01_fragile_gas_framework.md)
FRAMEWORK_REFS = {
    # Core structures
    r'\bwalker\b(?! state)': 'def-walker',  # Avoid "walker state" which is local
    r'\bswarm state space\b': 'def-swarm-and-state-space',
    r'\bswarm configuration\b': 'def-swarm-and-state-space',
    r'\balive and dead sets\b': 'def-alive-dead-sets',
    r'\balive set\b': 'def-alive-dead-sets',
    r'\bdead set\b': 'def-alive-dead-sets',

    # Metrics and distances
    r'\bN-particle displacement\b': 'def-n-particle-displacement-metric',
    r'\bdisplacement pseudometric\b': 'def-n-particle-displacement-metric',
    r'\bmetric quotient\b': 'def-metric-quotient',
    r'\bWasserstein-2 metric\b': 'lem-polishness-and-w2',

    # Measurement operators
    r'\breward measurement\b': 'def-reward-measurement',
    r'\braw value operator\b': 'def-raw-value-operator',
    r'\bdistance-to-companion\b': 'def-distance-to-companion',
    r'\bswarm aggregation operator\b': 'def-swarm-aggregation-operator-axiomatic',
    r'\bempirical measure aggregator\b': 'lem-empirical-aggregator-properties',

    # Noise measures
    r'\bperturbation measure\b': 'def-perturbation-measure',
    r'\bcloning measure\b': 'def-cloning-measure',
    r'\bheat kernel\b': 'lem-validation-heat-kernel',
    r'\buniform ball measure\b': 'lem-validation-uniform-ball',

    # Algorithmic space
    r'\balgorithmic space\b(?! uses)': 'def-algorithmic-space-generic',  # Avoid "algorithmic space uses"
    r'\balgorithmic distance\b(?! d_alg)': 'def-alg-distance',  # Avoid definition line

    # Axioms (only if not already referenced)
    r'\bAxiom of Range-Respecting Mean\b': 'def-axiom-range-respecting-mean',
    r'\bAxiom of Bounded Relative Collapse\b': 'def-axiom-bounded-relative-collapse',
    r'\bAxiom of Bounded Deviation from Aggregated Variance\b': 'def-axiom-bounded-deviation-variance',
    r'\bAxiom of Bounded Variance Production\b': 'def-axiom-bounded-variance-production',
    r'\bAxiom of Non-Degenerate Noise\b': 'def-axiom-non-degenerate-noise',
    r'\bAxiom of Projection Compatibility\b': 'def-axiom-projection-compatibility',
    r'\bAxiom of Position-Only Status Margin\b': 'def-axiom-margin-stability',

    # Operators
    r'\brescale function\b': 'def-canonical-logistic-rescale-function',
    r'\bperturbation operator\b': 'def-perturbation-operator',
    r'\bstatus update operator\b': 'def-status-update-operator',
    r'\bcloning score function\b': 'def-cloning-score-function',
}

# Sections where we should prioritize adding references
HIGH_VALUE_SECTIONS = [
    (0, 200),      # Introduction and TLDR
    (131, 541),    # Section 2-3: Framework alignment and definition
    (541, 900),    # Section 4: Axiom validation
    (2175, 2480),  # Section 6: Kernel representation
]


def find_first_occurrence(content: str, pattern: str, after_line: int = 0) -> Tuple[int, str]:
    """
    Find the first occurrence of a pattern after a given line number.

    Returns:
        (line_number, matched_text) or (-1, "") if not found
    """
    lines = content.split('\n')
    for i in range(after_line, len(lines)):
        line = lines[i]
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            # Check if already has a reference
            if '{prf:ref}' in line:
                continue  # Skip lines that already have references
            return (i, match.group(0))
    return (-1, "")


def add_reference_at_line(content: str, line_num: int, matched_text: str, ref_label: str) -> str:
    """
    Add a {prf:ref} link to matched text at specified line number.

    Strategy:
    - For axiom names: Add reference inline after the full name
    - For technical terms: Add parenthetical reference after first mention
    """
    lines = content.split('\n')
    line = lines[line_num]

    # Determine reference style
    if matched_text.startswith('Axiom of') or matched_text.startswith('Theorem of'):
        # Inline style for axiom/theorem names
        replacement = f"{matched_text} ({{prf:ref}}`{ref_label}`)"
    else:
        # Parenthetical style for technical terms
        replacement = f"{matched_text} ({{prf:ref}}`{ref_label}`)"

    # Replace first occurrence only
    new_line = line.replace(matched_text, replacement, 1)
    lines[line_num] = new_line

    return '\n'.join(lines)


def enrich_document(doc_path: Path) -> Tuple[str, Dict[str, int]]:
    """
    Add backward cross-references to the document.

    Returns:
        (enriched_content, statistics_dict)
    """
    with open(doc_path, 'r') as f:
        content = f.read()

    stats = {
        'refs_added': 0,
        'concepts_linked': 0,
        'sections_enhanced': set(),
    }

    # Track which concepts we've already referenced (one per major section)
    referenced_concepts = {}

    # Process each concept pattern
    for pattern, ref_label in FRAMEWORK_REFS.items():
        # Find first significant occurrence
        line_num, matched_text = find_first_occurrence(content, pattern)

        if line_num >= 0:
            # Check if in high-value section
            in_high_value = any(start <= line_num <= end for start, end in HIGH_VALUE_SECTIONS)

            if in_high_value:
                # Add reference
                content = add_reference_at_line(content, line_num, matched_text, ref_label)
                stats['refs_added'] += 1
                stats['concepts_linked'] += 1
                referenced_concepts[ref_label] = line_num

                # Determine section
                if line_num < 200:
                    stats['sections_enhanced'].add('Introduction')
                elif line_num < 541:
                    stats['sections_enhanced'].add('Definition')
                elif line_num < 900:
                    stats['sections_enhanced'].add('Axiom Validation')
                else:
                    stats['sections_enhanced'].add('Kernel Representation')

    return content, stats


def main():
    doc_path = Path('docs/source/1_euclidean_gas/02_euclidean_gas.md')

    if not doc_path.exists():
        print(f"Error: Document not found at {doc_path}")
        return 1

    print("=" * 60)
    print("BACKWARD CROSS-REFERENCE ENRICHMENT")
    print("Document: 02_euclidean_gas.md")
    print("=" * 60)
    print()

    # Create backup
    backup_path = doc_path.with_suffix('.md.backup_refs')
    import shutil
    shutil.copy(doc_path, backup_path)
    print(f"✓ Backup created: {backup_path}")
    print()

    # Enrich document
    print("Processing document...")
    enriched_content, stats = enrich_document(doc_path)

    # Write enriched version
    with open(doc_path, 'w') as f:
        f.write(enriched_content)

    # Print statistics
    print()
    print("=" * 60)
    print("ENRICHMENT REPORT")
    print("=" * 60)
    print()
    print(f"Backward references added: {stats['refs_added']}")
    print(f"Framework concepts linked: {stats['concepts_linked']}")
    print(f"Sections enhanced: {len(stats['sections_enhanced'])}")
    for section in sorted(stats['sections_enhanced']):
        print(f"  - {section}")
    print()
    print("✓ Document enriched successfully")
    print(f"✓ Output written to: {doc_path}")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
