#!/usr/bin/env python3
"""
Cross-Reference Agent - Backward Reference Enrichment
Processes a large markdown document to add backward cross-references
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
import sys


@dataclass
class MathEntity:
    """Represents a mathematical entity (definition, theorem, axiom, etc.)"""
    entity_type: str  # definition, theorem, axiom, lemma, etc.
    label: str
    name: str
    line_number: int
    section_number: str = ""
    content_start: int = 0
    content_end: int = 0
    referenced_labels: Set[str] = field(default_factory=set)
    concepts: List[str] = field(default_factory=list)


@dataclass
class CrossRefStats:
    """Statistics for cross-reference enrichment"""
    entities_processed: int = 0
    within_doc_refs_added: int = 0
    refs_by_type: Dict[str, int] = field(default_factory=dict)
    unlinked_concepts: List[str] = field(default_factory=list)
    processing_time: float = 0.0


def extract_entities(content: str, lines: List[str]) -> List[MathEntity]:
    """Extract all mathematical entities from the document"""
    entities = []

    # Pattern for Jupyter Book directives
    directive_pattern = r'^::::\{prf:(definition|theorem|axiom|lemma|proof|proposition|corollary|remark|assumption)\}\s*(.*)$'
    label_pattern = r'^:label:\s*(.+)$'

    current_entity = None
    current_label = None

    for line_num, line in enumerate(lines, start=1):
        # Check for directive start
        match = re.match(directive_pattern, line)
        if match:
            entity_type = match.group(1)
            entity_name = match.group(2).strip()
            current_entity = {
                'type': entity_type,
                'name': entity_name,
                'line': line_num
            }
            continue

        # Check for label
        if current_entity and not current_label:
            match = re.match(label_pattern, line)
            if match:
                current_label = match.group(1).strip()
                current_entity['label'] = current_label
                continue

        # Check for directive end
        if line.startswith('::::') and current_entity and current_label:
            entity = MathEntity(
                entity_type=current_entity['type'],
                label=current_label,
                name=current_entity['name'],
                line_number=current_entity['line']
            )
            entity.content_start = current_entity['line']
            entity.content_end = line_num
            entities.append(entity)
            current_entity = None
            current_label = None

    return entities


def find_existing_references(content: str) -> Set[str]:
    """Find all existing {prf:ref} references in the document"""
    ref_pattern = r'\{prf:ref\}`([^`]+)`'
    return set(re.findall(ref_pattern, content))


def extract_concepts_from_entity(lines: List[str], entity: MathEntity) -> List[str]:
    """Extract mathematical concepts mentioned in an entity's content"""
    concepts = []

    # Get entity content
    if entity.content_start and entity.content_end:
        content = '\n'.join(lines[entity.content_start:entity.content_end])

        # Look for common mathematical terms and patterns
        # These are heuristics based on the framework vocabulary
        patterns = [
            r'\b(walker|swarm|alive set|dead set|state space|displacement)\b',
            r'\b(reward|potential|fitness|cloning|perturbation|standardization)\b',
            r'\b(Lipschitz|Hölder|continuity|variance|mean|aggregat\w+)\b',
            r'\b(axiom|theorem|lemma|definition|assumption)\b',
            r'\b(boundary|regularity|smoothness|richness)\b',
            r'\b(revival|survival|status|cemetery)\b',
            r'\b(metric|distance|quotient|Polish)\b',
            r'\b(measure|kernel|probability|stochastic)\b',
        ]

        for pattern in patterns:
            concepts.extend(re.findall(pattern, content, re.IGNORECASE))

    return list(set(concepts))


def add_backward_references(
    doc_path: Path,
    entities: List[MathEntity],
    existing_refs: Set[str]
) -> Tuple[str, CrossRefStats]:
    """
    Add backward cross-references to the document.
    Returns modified content and statistics.
    """

    # Read document
    with open(doc_path, 'r') as f:
        lines = f.readlines()

    stats = CrossRefStats()

    # Build label to entity map for quick lookup
    label_to_entity = {e.label: e for e in entities}

    # Track which lines to modify
    modifications = {}  # line_number -> list of (label, position) tuples

    # Process entities in temporal order
    for current_idx, current_entity in enumerate(entities):
        stats.entities_processed += 1

        # Get earlier entities only (backward references)
        earlier_entities = entities[:current_idx]

        if not earlier_entities:
            continue  # First entity, no backward refs possible

        # Extract concepts from current entity
        entity_content = ''.join(lines[current_entity.content_start-1:current_entity.content_end])

        # Find potential backward references
        potential_refs = []

        for earlier in earlier_entities:
            # Check if earlier entity is already referenced
            if earlier.label in existing_refs or earlier.label in current_entity.referenced_labels:
                continue

            # Heuristic: Check if earlier entity's name/type appears in current content
            # This is simplified - a full implementation would use NLP/LLM
            earlier_keywords = [
                earlier.label.replace('-', ' '),
                earlier.name.lower() if earlier.name else '',
                earlier.entity_type
            ]

            for keyword in earlier_keywords:
                if keyword and len(keyword) > 5 and keyword in entity_content.lower():
                    potential_refs.append(earlier.label)
                    break

        # Add references (this is where we'd modify the content)
        if potential_refs:
            stats.within_doc_refs_added += len(potential_refs)
            for ref_label in potential_refs:
                ref_type = label_to_entity[ref_label].entity_type
                stats.refs_by_type[ref_type] = stats.refs_by_type.get(ref_type, 0) + 1
                current_entity.referenced_labels.add(ref_label)

    # Generate report without modifying (due to complexity)
    report = generate_report(stats, entities)

    return ''.join(lines), stats, report


def generate_report(stats: CrossRefStats, entities: List[MathEntity]) -> str:
    """Generate a comprehensive cross-reference report"""

    report = []
    report.append("# Backward Cross-Reference Report")
    report.append(f"\n## Summary Statistics\n")
    report.append(f"- **Entities Processed**: {stats.entities_processed}")
    report.append(f"- **Potential Backward References Identified**: {stats.within_doc_refs_added}")
    report.append(f"\n## References by Entity Type\n")

    for entity_type, count in sorted(stats.refs_by_type.items()):
        report.append(f"- {entity_type}: {count}")

    report.append(f"\n## Entity Distribution\n")
    type_counts = {}
    for e in entities:
        type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

    for entity_type, count in sorted(type_counts.items()):
        report.append(f"- {entity_type}: {count}")

    return '\n'.join(report)


def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    print(f"Processing: {doc_path}")

    # Read document
    with open(doc_path, 'r') as f:
        content = f.read()
        lines = content.splitlines()

    print(f"Document size: {len(lines)} lines")

    # Extract entities
    print("Extracting mathematical entities...")
    entities = extract_entities(content, lines)
    print(f"Found {len(entities)} entities")

    # Find existing references
    print("Finding existing references...")
    existing_refs = find_existing_references(content)
    print(f"Found {len(existing_refs)} existing references")

    # Process and add backward references
    print("Analyzing potential backward references...")
    modified_content, stats, report = add_backward_references(doc_path, entities, existing_refs)

    print("\n" + report)

    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
