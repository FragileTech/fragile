#!/usr/bin/env python3
"""
Strategic backward cross-reference enrichment.

This script adds high-value backward references to improve connectivity
for the 145 "source" entities identified in the connectivity analysis.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json


def read_document_section(path: Path, start_line: int, num_lines: int = 1000) -> str:
    """Read a section of the document."""
    with open(path) as f:
        lines = f.readlines()
    return ''.join(lines[start_line:start_line + num_lines])


def find_all_entities(content: str) -> List[Dict]:
    """Extract all entities with their positions."""
    pattern = r'::::\{prf:(\w+)\}[^\n]*\n(?:[^\n]*\n)*?:label:\s*([^\s\n]+)'
    entities = []

    for match in re.finditer(pattern, content):
        entity_type = match.group(1)
        label = match.group(2)
        start_pos = match.start()

        # Find the line number
        line_num = content[:start_pos].count('\n') + 1

        entities.append({
            'type': entity_type,
            'label': label,
            'line': line_num,
            'start_pos': start_pos,
        })

    return entities


def get_entity_content(content: str, start_pos: int) -> Tuple[str, int]:
    """Extract full entity content from start position to closing ::::."""
    # Find the closing ::::
    closing_pattern = r'\n::::\s*\n'
    match = re.search(closing_pattern, content[start_pos:])

    if match:
        end_pos = start_pos + match.end()
        return content[start_pos:end_pos], end_pos
    else:
        # Fallback: take next 500 characters
        return content[start_pos:start_pos + 500], start_pos + 500


def add_reference_after_phrase(text: str, phrase: str, label: str) -> Tuple[str, bool]:
    """
    Add {prf:ref} after first occurrence of phrase.
    Returns (modified_text, was_added).
    """
    # Check if reference already exists
    if f"{{prf:ref}}`{label}`" in text:
        return text, False

    # Find phrase (case-insensitive, flexible whitespace)
    normalized_phrase = re.sub(r'\s+', r'\\s+', phrase)
    pattern = re.compile(rf'\b{normalized_phrase}\b', re.IGNORECASE)

    match = pattern.search(text)
    if not match:
        return text, False

    # Don't add inside math blocks
    before = text[:match.end()]
    if before.count('$$') % 2 == 1 or before.count('$') % 2 == 1:
        return text, False

    # Insert reference
    insert_pos = match.end()
    ref = f" ({{prf:ref}}`{label}`)"

    # Check if there's already punctuation/parenthesis
    next_char = text[insert_pos:insert_pos+1]
    if next_char in '.,;:)':
        # Insert before punctuation
        ref = f" ({{prf:ref}}`{label}`)"
    elif next_char == '(':
        # Already has parenthesis, insert inside
        ref = f" {{prf:ref}}`{label}`,"

    modified = text[:insert_pos] + ref + text[insert_pos:]
    return modified, True


# High-value reference mappings (concept → label)
# Focus on the 145 priority "source" entities
REFERENCE_MAPPINGS = {
    # Core framework concepts (always reference these)
    "walker": "def-walker",
    "swarm state": "def-swarm-and-state-space",
    "alive set": "def-alive-dead-sets",
    "dead set": "def-alive-dead-sets",
    "valid state space": "def-valid-state-space",

    # Foundational axioms (reference when mentioned)
    "bounded algorithmic diameter": "axiom-bounded-algorithmic-diameter",
    "boundary regularity": "axiom-boundary-regularity",
    "boundary smoothness": "axiom-boundary-smoothness",
    "reward regularity": "axiom-reward-regularity",
    "guaranteed revival": "axiom-guaranteed-revival",
    "environmental richness": "axiom-environmental-richness",
    "sufficient amplification": "axiom-sufficient-amplification",
    "geometric consistency": "axiom-geometric-consistency",
    "bounded variance production": "axiom-bounded-variance-production",
    "bounded relative collapse": "axiom-bounded-relative-collapse",
    "range-respecting mean": "axiom-range-respecting-mean",
    "well-behaved rescale": "axiom-well-behaved-rescale",
    "non-degenerate noise": "axiom-non-degenerate-noise",
    "bounded measurement variance": "axiom-bounded-measurement-variance",
    "position-only status margin": "axiom-position-only-status-margin",
    "instep independence": "axiom-instep-independence",
    "projection compatibility": "axiom-projection-compatibility",

    # Key operators and measures
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
    "distance-to-companion": "def-distance-to-companion",
    "companion selection": "def-companion-selection",

    # Key structural components
    "displacement components": "def-displacement-components",
    "cemetery state": "def-cemetery-state-measure",
    "smoothed gaussian": "def-smoothed-gaussian-measure",
    "canonical logistic": "def-canonical-logistic-rescale",
    "smooth piecewise rescale": "def-smooth-piecewise-rescale",

    # Error decompositions
    "value error": "def-expected-squared-value-error",
    "structural error": "def-expected-squared-structural-error",
    "standardization error": "thm-general-bound-standardized-vector",
    "value error coefficients": "def-value-error-coefficients",
    "structural error coefficients": "def-structural-error-coefficients",

    # Important lemmas
    "empirical moments": "lem-empirical-moments-lipschitz",
    "potential boundedness": "lem-potential-boundedness",
    "rescale monotonicity": "lem-rescale-monotonicity",

    # Key theorems
    "forced activity": "thm-forced-activity",
    "revival guarantee": "thm-revival-guarantee",
    "canonical logistic validity": "thm-canonical-logistic-validity",
}


def process_entity_for_references(entity: Dict, content: str, all_entities: List[Dict]) -> Tuple[str, int]:
    """
    Process a single entity to add backward references.
    Returns (modified_content, num_refs_added).
    """
    # Extract entity content
    entity_text, end_pos = get_entity_content(content, entity['start_pos'])

    # Only add references to entities that appear BEFORE this one
    earlier_entities = [e for e in all_entities if e['line'] < entity['line']]
    earlier_labels = {e['label'] for e in earlier_entities}

    refs_added = 0
    modified_text = entity_text

    # Try to add references for each concept
    for phrase, target_label in REFERENCE_MAPPINGS.items():
        # Only add backward references
        if target_label not in earlier_labels:
            continue

        # Try to add reference
        new_text, was_added = add_reference_after_phrase(modified_text, phrase, target_label)

        if was_added:
            modified_text = new_text
            refs_added += 1

    # Replace in original content
    modified_content = content[:entity['start_pos']] + modified_text + content[end_pos:]

    return modified_content, refs_added


def main():
    """Main execution."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    print("="*70)
    print("STRATEGIC BACKWARD CROSS-REFERENCE ENRICHMENT")
    print("="*70)
    print()
    print(f"Document: {doc_path.name}")
    print(f"Target: Add references for {len(REFERENCE_MAPPINGS)} priority concepts")
    print()

    # Read full document
    print("Reading document...")
    content = doc_path.read_text()
    original_length = len(content)
    print(f"Document size: {original_length:,} bytes ({len(content.splitlines()):,} lines)")
    print()

    # Find all entities
    print("Extracting entity structure...")
    entities = find_all_entities(content)
    print(f"Found {len(entities)} entities")
    print()

    # Create backup
    backup_path = doc_path.parent / (doc_path.stem + "_backup_strategic.md")
    backup_path.write_text(content)
    print(f"Backup created: {backup_path.name}")
    print()

    # Process entities
    print("Adding backward cross-references...")
    print("-" * 70)

    total_refs_added = 0
    entities_modified = 0
    modified_content = content

    # Process entities in reverse order (to maintain positions)
    for i, entity in enumerate(reversed(entities), 1):
        # Only process entities after line 500 (ensure backward references)
        if entity['line'] < 500:
            continue

        # Process this entity
        new_content, refs_added = process_entity_for_references(
            entity, modified_content, entities
        )

        if refs_added > 0:
            modified_content = new_content
            total_refs_added += refs_added
            entities_modified += 1

            # Print progress every 10 modifications
            if entities_modified % 10 == 0:
                print(f"  Processed {i}/{len(entities)} entities, "
                      f"{total_refs_added} refs added, "
                      f"{entities_modified} entities modified")

    print()
    print("="*70)
    print("ENRICHMENT COMPLETE")
    print("="*70)
    print()
    print(f"📊 Statistics:")
    print(f"  Total backward references added: {total_refs_added}")
    print(f"  Entities modified: {entities_modified}")
    print(f"  Document size change: {len(modified_content) - original_length:+,} bytes")
    print()

    # Write modified document
    doc_path.write_text(modified_content)
    print(f"✅ Modified document saved: {doc_path.name}")
    print()

    # Analyze what was added
    print("📈 Reference distribution (sample):")
    ref_counts = {}
    for phrase, label in REFERENCE_MAPPINGS.items():
        count = modified_content.count(f"{{prf:ref}}`{label}`")
        if count > 0:
            ref_counts[label] = count

    for label, count in sorted(ref_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {label}: {count} references")


if __name__ == "__main__":
    main()
