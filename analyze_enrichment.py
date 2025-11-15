#!/usr/bin/env python3
"""Analyze the backward reference enrichment that was performed."""

import re
from pathlib import Path
from collections import defaultdict
import subprocess

def extract_references_from_diff():
    """Extract all new references from git diff."""
    result = subprocess.run(
        ['git', 'diff', 'docs/source/1_euclidean_gas/01_fragile_gas_framework.md'],
        capture_output=True,
        text=True,
        cwd='/home/guillem/fragile'
    )

    diff_lines = result.stdout.split('\n')

    new_refs = []
    for line in diff_lines:
        if line.startswith('+') and '{prf:ref}' in line:
            # Extract all references from this line
            refs = re.findall(r'\{prf:ref\}`([^`]+)`', line)
            new_refs.extend(refs)

    return new_refs


def count_references_in_file(filepath):
    """Count all references in current file."""
    content = Path(filepath).read_text()
    all_refs = re.findall(r'\{prf:ref\}`([^`]+)`', content)
    return all_refs


def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    print("="*80)
    print("BACKWARD CROSS-REFERENCE ENRICHMENT ANALYSIS")
    print("="*80)
    print()

    # Analyze new references
    print("Analyzing new references added...")
    new_refs = extract_references_from_diff()
    print(f"  Total new reference instances: {len(new_refs)}")
    print()

    # Count by target label
    ref_counts = defaultdict(int)
    for ref in new_refs:
        ref_counts[ref] += 1

    print(f"Unique labels referenced: {len(ref_counts)}")
    print()

    # Categorize by entity type
    by_type = defaultdict(int)
    for label in ref_counts.keys():
        entity_type = label.split('-')[0]
        by_type[entity_type] += ref_counts[label]

    print("New references by entity type:")
    for entity_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")
    print()

    print("Top 30 most referenced labels (new refs only):")
    for label, count in sorted(ref_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
        print(f"  {label}: {count} new refs")
    print()

    # Analyze current state
    print("Current document reference statistics:")
    all_refs = count_references_in_file(doc_path)
    print(f"  Total reference instances: {len(all_refs)}")

    all_ref_counts = defaultdict(int)
    for ref in all_refs:
        all_ref_counts[ref] += 1

    print(f"  Unique labels referenced: {len(all_ref_counts)}")
    print()

    # Check priority sources
    priority_sources = [
        "def-walker",
        "def-swarm-and-state-space",
        "def-alive-dead-sets",
        "axiom-boundary-smoothness",
        "axiom-reward-regularity",
        "axiom-guaranteed-revival",
        "def-perturbation-operator",
        "def-status-update-operator",
        "def-standardization-operator",
        "def-cloning-probability-function",
        "def-raw-value-operator",
    ]

    print("Priority source entities - incoming references:")
    for label in priority_sources:
        count = all_ref_counts.get(label, 0)
        new_count = ref_counts.get(label, 0)
        print(f"  {label}: {count} total ({new_count} new)")


if __name__ == "__main__":
    main()
