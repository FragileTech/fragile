#!/usr/bin/env python3
"""
Comprehensive backward cross-reference enrichment for 01_fragile_gas_framework.md

This script adds backward references to improve connectivity by:
1. Identifying entities that are referenced but don't reference back
2. Adding {prf:ref} links where concepts are used but not cited
3. Focusing on high-value connections to core framework concepts
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

# Document path
DOC_PATH = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

# Core framework concepts that should be referenced when used
CORE_CONCEPTS = {
    "def-walker": ["walker", "walkers"],
    "def-swarm-and-state-space": ["swarm", "swarms", r"\$\\mathcal\{S\}\$", r"\$\\Sigma_N\$"],
    "def-alive-dead-sets": ["alive set", "dead set", r"\$\\mathcal\{A\}\$", r"\$\\mathcal\{D\}\$"],
    "def-valid-state-space": ["valid state space", "state space"],
    "def-n-particle-displacement-metric": ["displacement metric", r"\$d_\{\\text\{Disp\}\}\$"],
    "def-perturbation-operator": ["perturbation operator", r"\$\\Psi_\{\\text\{pert\}\}\$"],
    "def-status-update-operator": ["status update operator"],
    "def-standardization-operator-n-dimensional": ["standardization operator"],
    "def-raw-value-operator": ["raw value operator"],
    "def-cloning-measure": ["cloning measure", "cloning transition"],
    "axiom-boundary-regularity": ["boundary regularity", "Axiom of Boundary Regularity"],
    "axiom-boundary-smoothness": ["boundary smoothness", "Axiom of Boundary Smoothness"],
    "axiom-guaranteed-revival": ["guaranteed revival", "Axiom of Guaranteed Revival"],
    "axiom-reward-regularity": ["reward regularity", "Axiom of Reward Regularity"],
    "axiom-non-degenerate-noise": ["non-degenerate noise"],
    "axiom-bounded-algorithmic-diameter": ["bounded algorithmic diameter"],
}

# Patterns for different directive types
DIRECTIVE_PATTERN = re.compile(
    r'::::\{prf:(definition|theorem|lemma|axiom|proposition|corollary|assumption|remark)\}[^\n]*\n'
    r':label:\s*([a-z0-9\-\_]+)',
    re.MULTILINE
)

def load_document(path: Path) -> str:
    """Load the markdown document."""
    return path.read_text(encoding='utf-8')

def extract_all_labels(content: str) -> Dict[str, Tuple[str, int, int]]:
    """
    Extract all labeled entities with their type and position.
    Returns dict: {label: (entity_type, start_pos, end_pos)}
    """
    labels = {}

    for match in DIRECTIVE_PATTERN.finditer(content):
        entity_type = match.group(1)
        label = match.group(2)
        start = match.start()

        # Find the end of this directive (next :::: or end of file)
        end_pattern = re.compile(r'::::', re.MULTILINE)
        end_matches = list(end_pattern.finditer(content, match.end()))
        if end_matches:
            end = end_matches[0].start()
        else:
            end = len(content)

        labels[label] = (entity_type, start, end)

    return labels

def find_entity_content(content: str, label: str, labels_map: Dict) -> str:
    """Extract the content of a specific entity."""
    if label not in labels_map:
        return ""

    _, start, end = labels_map[label]
    return content[start:end]

def has_reference_to(text: str, target_label: str) -> bool:
    """Check if text already contains a reference to the target label."""
    ref_pattern = rf'\{{prf:ref\}}`{re.escape(target_label)}`'
    return bool(re.search(ref_pattern, text))

def find_unreferenced_usage(entity_content: str, concept_label: str, patterns: List[str]) -> List[Tuple[int, str]]:
    """
    Find occurrences of concept patterns that are NOT already referenced.
    Returns list of (position, matched_text) tuples.
    """
    # Skip if already referenced
    if has_reference_to(entity_content, concept_label):
        return []

    occurrences = []
    for pattern in patterns:
        # Find all matches
        for match in re.finditer(pattern, entity_content, re.IGNORECASE):
            # Skip if inside a {prf:ref} directive
            before_text = entity_content[max(0, match.start()-20):match.start()]
            if "{prf:ref}" in before_text:
                continue

            # Skip if inside a label definition
            if ":label:" in before_text:
                continue

            occurrences.append((match.start(), match.group(0)))

    return occurrences

def add_reference_after_first_occurrence(entity_content: str, concept_label: str, patterns: List[str]) -> str:
    """
    Add a reference to concept_label after the first occurrence of any pattern.
    """
    occurrences = find_unreferenced_usage(entity_content, concept_label, patterns)

    if not occurrences:
        return entity_content

    # Sort by position and take the first one
    first_pos, first_match = min(occurrences, key=lambda x: x[0])

    # Find the end of the first occurrence
    insert_pos = first_pos + len(first_match)

    # Check if we're inside a word (add space before reference)
    next_char = entity_content[insert_pos] if insert_pos < len(entity_content) else ''
    if next_char.isalnum():
        return entity_content  # Don't insert mid-word

    # Insert the reference
    ref = f" ({{prf:ref}}`{concept_label}`)"

    # Avoid double parentheses
    if entity_content[insert_pos:insert_pos+2] == " (":
        ref = f", {{prf:ref}}`{concept_label}`"

    new_content = entity_content[:insert_pos] + ref + entity_content[insert_pos:]

    return new_content

def enrich_with_backward_references(content: str) -> Tuple[str, List[str]]:
    """
    Add backward references to core concepts throughout the document.
    Returns (enriched_content, list_of_changes)
    """
    labels_map = extract_all_labels(content)
    changes = []
    enriched = content

    # For each entity in the document
    for label, (entity_type, start, end) in labels_map.items():
        original_entity = content[start:end]
        modified_entity = original_entity

        # Check each core concept
        for concept_label, patterns in CORE_CONCEPTS.items():
            # Skip self-references
            if label == concept_label:
                continue

            # Skip if concept is defined after this entity (only backward refs)
            if concept_label in labels_map:
                concept_start = labels_map[concept_label][1]
                if concept_start >= start:
                    continue  # This would be a forward reference

            # Try to add reference
            new_entity = add_reference_after_first_occurrence(modified_entity, concept_label, patterns)

            if new_entity != modified_entity:
                changes.append(f"Added {concept_label} reference to {label} ({entity_type})")
                modified_entity = new_entity

        # Replace in enriched content if changed
        if modified_entity != original_entity:
            enriched = enriched.replace(original_entity, modified_entity, 1)

    return enriched, changes

def main():
    print("Loading document...")
    content = load_document(DOC_PATH)

    print("Extracting entities...")
    labels_map = extract_all_labels(content)
    print(f"Found {len(labels_map)} labeled entities")

    print("\nAdding backward references...")
    enriched_content, changes = enrich_with_backward_references(content)

    print(f"\n✓ Added {len(changes)} backward references:")
    for change in changes[:20]:  # Show first 20
        print(f"  - {change}")

    if len(changes) > 20:
        print(f"  ... and {len(changes) - 20} more")

    # Create backup
    backup_path = DOC_PATH.with_suffix('.md.backup_enrichment')
    print(f"\nCreating backup at {backup_path}")
    backup_path.write_text(content, encoding='utf-8')

    # Write enriched version
    print(f"Writing enriched document to {DOC_PATH}")
    DOC_PATH.write_text(enriched_content, encoding='utf-8')

    print("\n✓ Done! Re-run connectivity analysis to see improvements.")

if __name__ == "__main__":
    main()
