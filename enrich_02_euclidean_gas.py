#!/usr/bin/env python3
"""
Add strategic backward cross-references to 02_euclidean_gas.md

Follows backward-only temporal ordering:
- References only to 01_fragile_gas_framework.md (doc 01 < doc 02)
- Adds references at first significant mention
- Avoids over-referencing
"""

import re
from pathlib import Path
from typing import List, Tuple

# High-value locations for adding references (concept, search_pattern, ref_label, section_constraint)
# section_constraint: 'intro' | 'definition' | 'axioms' | 'kernel' | None (any)
REFERENCE_TARGETS = [
    # Introduction § section - establish framework context
    ("walker", r"walker\s+\$w_i", "def-walker", "intro", 30, 45),
    ("swarm state space", r"swarm state space", "def-swarm-and-state-space", "intro", 0, 100),
    ("alive set", r"alive set", "def-alive-dead-sets", "definition", 100, 200),

    # Section 3: Algorithm definition - link operators
    ("displacement metric", r"displacement.*metric", "def-n-particle-displacement-metric", "definition", 490, 520),
    ("measurement companions", r"measurement companions", "def-distance-to-companion", "kernel", 2180, 2210),
    ("perturbation kernel", r"perturbation kernel", "def-perturbation-measure", "definition", 520, 545),
    ("cloning measure", r"[Cc]loning measure", "def-cloning-measure", "definition", 295, 370),

    # Section 4: Axiom validation - explicit axiom linkage
    ("Wasserstein", r"Wasserstein-2 metric", "lem-polishness-and-w2", "axioms", 715, 740),
    ("heat kernel", r"heat kernel", "lem-validation-heat-kernel", "definition", 80, 90),
    ("uniform ball", r"uniform ball", "lem-validation-uniform-ball", "definition", 295, 310),

    # Additional high-value concepts
    ("algorithmic space", r"algorithmic space.*is the", "def-algorithmic-space-generic", "definition", 370, 390),
    ("rescale function", r"[Rr]escale.*function", "def-canonical-logistic-rescale-function", "definition", 160, 170),
    ("swarm aggregation", r"swarm aggregation operator", "def-swarm-aggregation-operator-axiomatic", "definition", 350, 370),
]


def find_target_in_range(lines: List[str], pattern: str, start: int, end: int) -> Tuple[int, str]:
    """
    Find pattern in specified line range, skipping lines with existing refs.

    Returns:
        (line_number, matched_text) or (-1, "")
    """
    for i in range(max(0, start), min(end, len(lines))):
        line = lines[i]

        # Skip if line already has a reference
        if '{prf:ref}' in line:
            continue

        # Skip if in code block or directive
        if line.strip().startswith('```') or line.strip().startswith(':::'):
            continue

        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return (i, match.group(0))

    return (-1, "")


def add_inline_reference(line: str, matched_text: str, ref_label: str) -> str:
    """
    Add {prf:ref} as parenthetical after matched text.

    Handles various text contexts gracefully.
    """
    # Find the matched text and add reference
    # Use parenthetical style for clean integration
    replacement = f"{matched_text} ({{prf:ref}}`{ref_label}`)"

    # Replace only first occurrence
    new_line = line.replace(matched_text, replacement, 1)
    return new_line


def enrich_document(doc_path: Path) -> dict:
    """
    Add backward cross-references to 02_euclidean_gas.md.

    Returns:
        Statistics dictionary
    """
    with open(doc_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    stats = {
        'refs_added': 0,
        'concepts_linked': [],
        'line_numbers': [],
    }

    # Process each reference target
    for concept, pattern, ref_label, section, start, end in REFERENCE_TARGETS:
        line_num, matched_text = find_target_in_range(lines, pattern, start, end)

        if line_num >= 0:
            # Add reference
            old_line = lines[line_num]
            new_line = add_inline_reference(old_line, matched_text, ref_label)

            if new_line != old_line:  # Reference was actually added
                lines[line_num] = new_line
                stats['refs_added'] += 1
                stats['concepts_linked'].append(concept)
                stats['line_numbers'].append(line_num + 1)  # 1-indexed for reporting

                print(f"✓ Line {line_num + 1}: Added ref to '{concept}' → {ref_label}")

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

    print("="  * 70)
    print("BACKWARD CROSS-REFERENCE ENRICHMENT: 02_euclidean_gas.md")
    print("=" * 70)
    print()
    print("Strategy: Add references to framework concepts from 01_fragile_gas_framework.md")
    print("Constraint: Backward-only (doc 01 → doc 02)")
    print()

    # Create backup
    import shutil
    backup_path = doc_path.with_suffix('.md.backup_enrichment')
    shutil.copy(doc_path, backup_path)
    print(f"✓ Backup created: {backup_path}")
    print()

    # Enrich document
    print("Processing...")
    print()
    stats = enrich_document(doc_path)

    # Report
    print()
    print("=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)
    print()
    print(f"Backward references added: {stats['refs_added']}")
    print(f"Framework concepts linked:")
    for i, concept in enumerate(stats['concepts_linked'], 1):
        line_num = stats['line_numbers'][i-1]
        print(f"  {i}. {concept} (line {line_num})")
    print()
    print(f"✓ Document written to: {doc_path}")
    print(f"✓ Backup available at: {backup_path}")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
