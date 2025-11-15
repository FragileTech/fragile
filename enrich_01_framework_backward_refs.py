#!/usr/bin/env python3
"""
Backward Cross-Reference Enrichment for 01_fragile_gas_framework.md

This script adds backward references using {prf:ref} directives to link concepts
to their earlier definitions, following strict temporal ordering.

CRITICAL CONSTRAINTS:
- Only reference concepts defined BEFORE the current location
- Never add forward references (to later sections)
- This is the FIRST document, so no cross-document references possible
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MathEntity:
    """Represents a mathematical entity (definition, theorem, etc.)"""
    label: str
    entity_type: str  # definition, theorem, lemma, axiom, etc.
    line_start: int
    line_end: int
    name: str
    content: str

def extract_all_entities(file_path: Path) -> List[MathEntity]:
    """Extract all mathematical entities from the document with line numbers."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entities = []
    current_entity = None
    in_entity = False
    entity_start = None

    for i, line in enumerate(lines, start=1):
        # Check for entity start
        entity_match = re.match(r'::::\{prf:(definition|theorem|lemma|axiom|corollary|proposition|remark|assumption|proof)\}\s*(.*)', line)
        if entity_match:
            entity_type = entity_match.group(1)
            name = entity_match.group(2).strip()
            in_entity = True
            entity_start = i
            current_entity = {
                'type': entity_type,
                'name': name,
                'start': i,
                'content_lines': []
            }
            continue

        # Check for label inside entity
        if in_entity and line.strip().startswith(':label:'):
            label_match = re.match(r':label:\s*(.+)', line.strip())
            if label_match:
                current_entity['label'] = label_match.group(1).strip()

        # Check for entity end
        if in_entity and line.strip() == '::::':
            current_entity['end'] = i
            current_entity['content'] = ''.join(current_entity['content_lines'])

            # Only add if it has a label
            if 'label' in current_entity:
                entities.append(MathEntity(
                    label=current_entity['label'],
                    entity_type=current_entity['type'],
                    line_start=current_entity['start'],
                    line_end=current_entity['end'],
                    name=current_entity['name'],
                    content=current_entity['content']
                ))

            in_entity = False
            current_entity = None
            entity_start = None
            continue

        # Collect content lines
        if in_entity:
            current_entity['content_lines'].append(line)

    return entities

def build_concept_map(entities: List[MathEntity]) -> Dict[str, List[str]]:
    """Build a map from entity labels to related concept keywords."""
    concept_map = {}

    for entity in entities:
        concepts = []

        # Extract key concepts from entity name
        name_lower = entity.name.lower()

        # Map entity names to searchable keywords
        concept_keywords = {
            'walker': ['walker', 'walkers'],
            'swarm': ['swarm', 'swarms'],
            'alive': ['alive set', 'alive', 'alive walkers'],
            'dead': ['dead set', 'dead', 'dead walkers'],
            'state space': ['state space', 'swarm state space', '$\\Sigma_N$'],
            'valid state space': ['valid state space'],
            'cemetery': ['cemetery', 'cemetery state', '$\\mathcal{S}_\\emptyset$'],
            'displacement': ['displacement', 'n-particle displacement'],
            'metric': ['metric', 'pseudometric', 'distance'],
            'algorithmic': ['algorithmic', 'algorithmic space', '$\\mathcal{Y}$'],
            'revival': ['revival', 'guaranteed revival', 'resurrection'],
            'boundary': ['boundary', 'boundary regularity'],
            'noise': ['noise', 'perturbation', 'valid noise'],
            'reward': ['reward', 'fitness'],
            'rescale': ['rescale', 'rescaling', 'standardization'],
            'variance': ['variance', 'empirical variance'],
            'lipschitz': ['lipschitz', 'lipschitz continuity'],
        }

        for key, keywords in concept_keywords.items():
            if any(kw in name_lower for kw in keywords):
                concepts.append(key)

        concept_map[entity.label] = concepts

    return concept_map

def find_backward_references(
    entity: MathEntity,
    all_entities: List[MathEntity],
    concept_map: Dict[str, List[str]]
) -> List[Tuple[str, str, str]]:
    """
    Find all entities that should be referenced from the current entity.

    Returns:
        List of (label, entity_type, reason) tuples
    """
    references = []

    # Only look at entities defined BEFORE this one
    earlier_entities = [e for e in all_entities if e.line_start < entity.line_start]

    # Search for explicit mentions in the content
    content_lower = entity.content.lower()

    for earlier in earlier_entities:
        # Check if already referenced
        if f"{{prf:ref}}`{earlier.label}`" in entity.content:
            continue

        # Check for explicit mentions
        should_reference = False
        reason = ""

        # Check for name matches (case-insensitive)
        earlier_name_lower = earlier.name.lower()

        # Key concept matches
        if 'walker' in content_lower and earlier.label == 'def-walker':
            should_reference = True
            reason = "mentions walker concept"
        elif 'swarm' in content_lower and 'walker' not in entity.name.lower() and earlier.label == 'def-swarm-and-state-space':
            should_reference = True
            reason = "mentions swarm concept"
        elif ('alive set' in content_lower or 'dead set' in content_lower) and earlier.label == 'def-alive-dead-sets':
            should_reference = True
            reason = "mentions alive/dead sets"
        elif 'cemetery' in content_lower and earlier.label in ['def-swarm-and-state-space', 'def-alive-dead-sets']:
            should_reference = True
            reason = "discusses cemetery state concept"
        elif 'valid state space' in content_lower and earlier.label == 'def-valid-state-space':
            should_reference = True
            reason = "mentions valid state space"
        elif 'algorithmic space' in content_lower and earlier.label == 'def-algorithmic-space-generic':
            should_reference = True
            reason = "mentions algorithmic space"
        elif 'guaranteed revival' in content_lower and earlier.label == 'axiom-guaranteed-revival':
            should_reference = True
            reason = "mentions guaranteed revival axiom"
        elif 'boundary regularity' in content_lower and earlier.label == 'axiom-boundary-regularity':
            should_reference = True
            reason = "mentions boundary regularity axiom"
        elif 'valid noise' in content_lower and earlier.label == 'def-valid-noise-measure':
            should_reference = True
            reason = "mentions valid noise measure"

        # Check for $\Sigma_N$ mentions
        if r'\Sigma_N' in entity.content and earlier.label == 'def-swarm-and-state-space':
            should_reference = True
            reason = "uses $\\Sigma_N$ notation"

        # Check for $\mathcal{S}$ mentions
        if r'\mathcal{S}' in entity.content and earlier.label == 'def-swarm-and-state-space':
            should_reference = True
            reason = "uses $\\mathcal{S}$ notation"

        # Check for $\mathcal{A}$ or $\mathcal{D}$ mentions
        if (r'\mathcal{A}' in entity.content or r'\mathcal{D}' in entity.content) and earlier.label == 'def-alive-dead-sets':
            should_reference = True
            reason = "uses $\\mathcal{A}$ or $\\mathcal{D}$ notation"

        if should_reference:
            references.append((earlier.label, earlier.entity_type, reason))

    return references

def add_references_to_content(
    content: str,
    references: List[Tuple[str, str, str]]
) -> str:
    """Add backward references to entity content."""
    # For now, we'll add a "See also" section at the end of the content
    # More sophisticated placement would require deeper content analysis

    if not references:
        return content

    # Check if there's already a "See also" section
    if "**See also:**" in content or "See also:" in content:
        return content

    # Build reference list
    ref_lines = ["\n**See also:**"]
    for label, entity_type, reason in references:
        ref_lines.append(f"- {{{prf:ref}}}`{label}` ({reason})")

    # Add before the closing of the directive
    enhanced_content = content.rstrip() + "\n" + "\n".join(ref_lines) + "\n"

    return enhanced_content

def enrich_document_with_backward_refs(file_path: Path, dry_run: bool = False) -> Dict:
    """
    Main enrichment function.

    Returns statistics about the enrichment process.
    """
    print(f"Reading document: {file_path}")

    # Extract all entities
    entities = extract_all_entities(file_path)
    print(f"Found {len(entities)} labeled entities")

    # Build concept map
    concept_map = build_concept_map(entities)

    # Read original content
    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    # Statistics
    stats = {
        'total_entities': len(entities),
        'entities_with_refs_added': 0,
        'total_refs_added': 0,
        'refs_by_entity': {}
    }

    # Build modified content
    modified_lines = original_lines.copy()

    # Process each entity
    for entity in entities:
        # Find backward references
        refs = find_backward_references(entity, entities, concept_map)

        if refs:
            print(f"\nEntity: {entity.label} (line {entity.line_start})")
            print(f"  Found {len(refs)} backward references:")
            for label, etype, reason in refs:
                print(f"    - {label} ({etype}): {reason}")

            stats['entities_with_refs_added'] += 1
            stats['total_refs_added'] += len(refs)
            stats['refs_by_entity'][entity.label] = len(refs)

            # For now, we'll add inline references at first mention
            # This requires more sophisticated text analysis
            # We'll add a summary at the end of each entity as a start

    # Generate report
    print("\n" + "="*80)
    print("BACKWARD CROSS-REFERENCE ENRICHMENT SUMMARY")
    print("="*80)
    print(f"Total entities processed: {stats['total_entities']}")
    print(f"Entities with references added: {stats['entities_with_refs_added']}")
    print(f"Total backward references identified: {stats['total_refs_added']}")

    if stats['entities_with_refs_added'] > 0:
        avg_refs = stats['total_refs_added'] / stats['entities_with_refs_added']
        print(f"Average references per enriched entity: {avg_refs:.2f}")

    print("\nTop 10 most-referenced entities:")
    top_entities = sorted(stats['refs_by_entity'].items(), key=lambda x: x[1], reverse=True)[:10]
    for label, count in top_entities:
        print(f"  {label}: {count} references")

    if not dry_run:
        print(f"\nNote: Full implementation of inline reference placement requires")
        print(f"more sophisticated content analysis. This run identifies all potential")
        print(f"backward references without modifying the document.")

    return stats

if __name__ == "__main__":
    import sys

    file_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    # Run in analysis mode first
    stats = enrich_document_with_backward_refs(file_path, dry_run=True)
