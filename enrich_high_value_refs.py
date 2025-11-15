#!/usr/bin/env python3
"""
Add high-value backward references to theorems, lemmas, and axioms.
This focuses on improving connectivity for important results.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

DOC_PATH = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

def extract_labels_with_positions(content: str) -> Dict[str, Tuple[str, int, int]]:
    """Extract all labels with their type and line positions."""
    labels = {}
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]
        directive_match = re.match(r'^::::\{prf:(definition|theorem|lemma|axiom|proposition|corollary|assumption|remark|proof)\}', line)
        if not directive_match:
            directive_match = re.match(r'^:::\{prf:(definition|theorem|lemma|axiom|proposition|corollary|assumption|remark|proof)\}', line)

        if directive_match:
            entity_type = directive_match.group(1)
            start_line = i

            if i + 1 < len(lines) and lines[i+1].startswith(':label:'):
                label_match = re.match(r':label:\s*([a-z0-9\-\_]+)', lines[i+1])
                if label_match:
                    label = label_match.group(1)
                    end_line = i + 2
                    colons_count = 4 if line.startswith('::::') else 3
                    closing = ':' * colons_count

                    while end_line < len(lines):
                        if lines[end_line].startswith(closing) and not lines[end_line].startswith(closing + '{'):
                            break
                        end_line += 1

                    labels[label] = (entity_type, start_line, end_line)
        i += 1

    return labels

def add_high_value_backward_refs(content: str) -> Tuple[str, List[str]]:
    """Add backward references to high-value entities."""
    lines = content.split('\n')
    labels = extract_labels_with_positions(content)
    changes = []

    # High-value entities to reference (ordered by importance)
    high_value_refs = {
        # Core axioms
        "axiom-guaranteed-revival": [
            (r'\bguaranteed revival\b', "guaranteed revival"),
            (r'\brevival guarantee\b', "revival guarantee"),
            (r'\$\\kappa_\{\\text\{revival\}\}\$', r"$\kappa_{\text{revival}}$"),
        ],
        "axiom-boundary-regularity": [
            (r'\bBoundary Regularity\b', "Boundary Regularity"),
            (r'\bboundary regularity\b', "boundary regularity"),
        ],
        "axiom-boundary-smoothness": [
            (r'\bBoundary Smoothness\b', "Boundary Smoothness"),
            (r'\bboundary smoothness\b', "boundary smoothness"),
        ],
        "axiom-reward-regularity": [
            (r'\bReward Regularity\b', "Reward Regularity"),
            (r'\$L_R\$(?!\s*\()', r"$L_R$"),
        ],
        "axiom-non-degenerate-noise": [
            (r'\bnon-degenerate noise\b', "non-degenerate noise"),
        ],
        "axiom-bounded-algorithmic-diameter": [
            (r'\bBounded Algorithmic Diameter\b', "Bounded Algorithmic Diameter"),
            (r'\$D_\{\\mathcal\{Y\}\}\$(?!\s*\()', r"$D_{\mathcal{Y}}$"),
        ],

        # Core definitions
        "def-walker": [
            (r'\bwalker\b(?!\s*\()', "walker"),
        ],
        "def-swarm-and-state-space": [
            (r'\bswarm state\b', "swarm state"),
            (r'\bSwarm State Space\b', "Swarm State Space"),
        ],
        "def-alive-dead-sets": [
            (r'\balive set\b(?!\s*\()', "alive set"),
            (r'\$\\mathcal\{A\}\(\\mathcal\{S\}\)', r"$\mathcal{A}(\mathcal{S})"),
        ],
        "def-perturbation-operator": [
            (r'\bperturbation operator\b(?!\s*\()', "perturbation operator"),
        ],
        "def-standardization-operator-n-dimensional": [
            (r'\bstandardization operator\b(?!\s*\()', "standardization operator"),
        ],
        "def-status-update-operator": [
            (r'\bstatus update\b(?!\s*\()', "status update"),
        ],
        "def-cloning-measure": [
            (r'\bcloning measure\b(?!\s*\()', "cloning measure"),
        ],
        "def-raw-value-operator": [
            (r'\braw value\b(?!\s*\()', "raw value"),
        ],

        # Important theorems
        "thm-revival-guarantee": [
            (r'\bTheorem of Almost-Sure Revival\b', "Theorem of Almost-Sure Revival"),
            (r'\balmost-sure revival\b', "almost-sure revival"),
        ],
        "thm-mcdiarmids-inequality": [
            (r'\bMcDiarmid\'?s inequality\b', "McDiarmid's inequality"),
        ],

        # Important lemmas
        "lem-empirical-aggregator-properties": [
            (r'\bempirical aggregator\b', "empirical aggregator"),
        ],
    }

    # Process each entity
    for current_label, (entity_type, start_line, end_line) in sorted(labels.items(), key=lambda x: x[1][1]):
        entity_lines = lines[start_line:end_line+1]

        for ref_label, patterns in high_value_refs.items():
            if current_label == ref_label:
                continue

            if ref_label in labels:
                ref_start = labels[ref_label][1]
                if ref_start >= start_line:
                    continue
            else:
                continue

            entity_text = '\n'.join(entity_lines)
            if f'{{prf:ref}}`{ref_label}`' in entity_text:
                continue

            # Try each pattern
            for pattern_regex, display_text in patterns:
                added = False
                for i, line in enumerate(entity_lines):
                    if ':label:' in line or line.startswith(':::'):
                        continue

                    match = re.search(pattern_regex, line, re.IGNORECASE)
                    if match:
                        pos = match.end()
                        if pos < len(line) and line[pos].isalnum():
                            continue

                        # Smart insertion
                        insert_text = f' ({{prf:ref}}`{ref_label}`)'

                        # Check context
                        after = line[pos:pos+5]
                        if after.startswith(' ('):
                            insert_text = f', {{prf:ref}}`{ref_label}`'
                        elif after.startswith(')'):
                            insert_text = f' ({{prf:ref}}`{ref_label}`),'

                        new_line = line[:pos] + insert_text + line[pos:]
                        entity_lines[i] = new_line
                        changes.append(f"{current_label} → {ref_label}")
                        added = True
                        break

                if added:
                    break

            if added:
                lines[start_line:end_line+1] = entity_lines
                break  # Only one ref per entity per pass

    return '\n'.join(lines), changes

def main():
    print("=" * 80)
    print("HIGH-VALUE BACKWARD REFERENCE ENRICHMENT")
    print("=" * 80)

    content = DOC_PATH.read_text(encoding='utf-8')
    labels = extract_labels_with_positions(content)
    print(f"\nFound {len(labels)} entities")

    print("\nAdding high-value backward references...")
    enriched, changes = add_high_value_backward_refs(content)

    print(f"\n✓ Added {len(changes)} new references\n")

    if changes:
        for i, change in enumerate(changes[:25], 1):
            print(f"  {i:2}. {change}")
        if len(changes) > 25:
            print(f"  ... and {len(changes) - 25} more")

    # Backup
    backup_path = DOC_PATH.with_suffix('.md.backup_high_value')
    backup_path.write_text(content, encoding='utf-8')

    # Write
    DOC_PATH.write_text(enriched, encoding='utf-8')

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
