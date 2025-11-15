#!/usr/bin/env python3
"""
Final comprehensive pass to add backward references.
Focus on mathematical relationships and concept usage.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys

DOC_PATH = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

def extract_labels(content: str) -> Dict[str, Tuple[str, int, int]]:
    """Extract all labeled entities."""
    labels = {}
    lines = content.split('\n')

    for i, line in enumerate(lines):
        match = re.match(r'^::::\{prf:(definition|theorem|lemma|axiom|proposition|corollary|assumption|remark|proof)\}', line)
        if not match:
            match = re.match(r'^:::\{prf:(definition|theorem|lemma|axiom|proposition|corollary|assumption|remark|proof)\}', line)

        if match and i + 1 < len(lines):
            label_match = re.match(r':label:\s*([a-z0-9\-\_]+)', lines[i+1])
            if label_match:
                entity_type = match.group(1)
                label = label_match.group(1)
                start = i

                colons = 4 if line.startswith('::::') else 3
                closing = ':' * colons
                end = i + 2

                while end < len(lines):
                    if lines[end].startswith(closing) and not lines[end].startswith(closing + '{'):
                        break
                    end += 1

                labels[label] = (entity_type, start, end)

    return labels

def add_comprehensive_refs(content: str) -> Tuple[str, List[str]]:
    """Add backward references comprehensively."""
    lines = content.split('\n')
    labels = extract_labels(content)
    changes = []

    # Extended reference patterns - more comprehensive
    ref_patterns = {
        # Operators
        "def-perturbation-operator": [
            (r'perturbation step\b', "perturbation step"),
            (r'noise perturbation\b', "noise perturbation"),
        ],
        "def-status-update-operator": [
            (r'status check\b', "status check"),
            (r'validity check\b', "validity check"),
        ],
        "def-cloning-measure": [
            (r'cloning step\b', "cloning step"),
            (r'cloning transition\b', "cloning transition"),
        ],
        "def-companion-selection-measure": [
            (r'companion selection\b', "companion selection"),
        ],

        # Metrics and distances
        "def-alg-distance": [
            (r'\$d_\{\\text\{alg\}\}\$', r"$d_{\text{alg}}$"),
            (r'algorithmic distance\b', "algorithmic distance"),
        ],
        "def-distance-to-cemetery-state": [
            (r'cemetery state\b', "cemetery state"),
            (r'\$\\mathcal\{S\}_\\emptyset\$', r"$\mathcal{S}_\emptyset$"),
        ],
        "def-w2-output-metric": [
            (r'\$W_2\$ metric', r"$W_2$ metric"),
            (r'Wasserstein-2\b', "Wasserstein-2"),
        ],

        # Key axioms
        "axiom-margin-stability": [
            (r'margin stability\b', "margin stability"),
            (r'Margin Stability\b', "Margin Stability"),
        ],
        "axiom-bounded-measurement-variance": [
            (r'Bounded Measurement Variance\b', "Bounded Measurement Variance"),
        ],

        # Important definitions
        "def-valid-noise-measure": [
            (r'valid noise\b', "valid noise"),
        ],
        "def-reward-measurement": [
            (r'reward measurement\b', "reward measurement"),
        ],
        "def-cemetery-state-measure": [
            (r'cemetery measure\b', "cemetery measure"),
        ],
    }

    # Process entities
    modified_count = 0
    for current_label, (entity_type, start_line, end_line) in sorted(labels.items(), key=lambda x: x[1][1]):
        # Skip proofs (they already have their own references)
        if entity_type == 'proof':
            continue

        entity_lines = lines[start_line:end_line+1]
        entity_modified = False

        for ref_label, patterns in ref_patterns.items():
            if current_label == ref_label:
                continue

            if ref_label not in labels:
                continue

            ref_start = labels[ref_label][1]
            if ref_start >= start_line:
                continue  # Would be forward ref

            entity_text = '\n'.join(entity_lines)
            if f'{{prf:ref}}`{ref_label}`' in entity_text:
                continue  # Already referenced

            # Try to add reference
            for pattern_regex, display_text in patterns:
                for i, line in enumerate(entity_lines):
                    if ':label:' in line or line.startswith(':::'):
                        continue

                    match = re.search(pattern_regex, line)
                    if match:
                        pos = match.end()
                        if pos < len(line) and line[pos].isalnum():
                            continue

                        new_line = line[:pos] + f' ({{prf:ref}}`{ref_label}`)' + line[pos:]
                        entity_lines[i] = new_line
                        changes.append(f"{current_label} ({entity_type}) → {ref_label}")
                        entity_modified = True
                        break

                if entity_modified:
                    break

            if entity_modified:
                lines[start_line:end_line+1] = entity_lines
                modified_count += 1
                break  # One ref per entity per pass

    return '\n'.join(lines), changes

def main():
    print("=" * 80)
    print("FINAL COMPREHENSIVE BACKWARD REFERENCE PASS")
    print("=" * 80)

    content = DOC_PATH.read_text(encoding='utf-8')
    labels = extract_labels(content)
    print(f"\nTotal entities: {len(labels)}")
    print(f"  - {sum(1 for l, (t, _, _) in labels.items() if t == 'definition')} definitions")
    print(f"  - {sum(1 for l, (t, _, _) in labels.items() if t == 'theorem')} theorems")
    print(f"  - {sum(1 for l, (t, _, _) in labels.items() if t == 'lemma')} lemmas")
    print(f"  - {sum(1 for l, (t, _, _) in labels.items() if t == 'axiom')} axioms")
    print(f"  - {sum(1 for l, (t, _, _) in labels.items() if t == 'proof')} proofs")

    print("\nAdding backward references...")
    enriched, changes = add_comprehensive_refs(content)

    print(f"\n✓ Added {len(changes)} new references\n")

    if changes:
        for i, change in enumerate(changes, 1):
            print(f"  {i:2}. {change}")

    # Backup
    backup = DOC_PATH.with_suffix('.md.backup_final')
    backup.write_text(content, encoding='utf-8')

    # Write
    DOC_PATH.write_text(enriched, encoding='utf-8')

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
