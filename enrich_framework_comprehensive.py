#!/usr/bin/env python3
"""
Comprehensive Backward Cross-Reference Enrichment for 01_fragile_gas_framework.md

Strategy:
1. Extract all entities with labels and line numbers
2. Build dependency graph based on concept mentions
3. Add inline backward references at first mention of concepts
4. Generate detailed enrichment report

CRITICAL: BACKWARD-ONLY references (never forward)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Entity:
    """Mathematical entity with full metadata"""
    label: str
    entity_type: str
    name: str
    line_start: int
    line_end: int
    content: List[str]  # Lines of content

    def __post_init__(self):
        self.content_text = ''.join(self.content)

    def has_reference_to(self, target_label: str) -> bool:
        """Check if this entity already references the target"""
        return f"{{prf:ref}}`{target_label}`" in self.content_text

@dataclass
class BackwardReference:
    """Represents a backward reference opportunity"""
    source_label: str
    target_label: str
    target_type: str
    reason: str
    first_mention_line_offset: int = -1  # Offset within entity
    suggested_placement: str = ""
    priority: int = 1  # 1=high, 2=medium, 3=low

def read_document(file_path: Path) -> List[str]:
    """Read document as list of lines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def extract_entities(lines: List[str]) -> List[Entity]:
    """Extract all labeled mathematical entities"""
    entities = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Match entity start: :::{prf:TYPE} NAME
        match = re.match(r'::::\{prf:(definition|theorem|lemma|axiom|corollary|proposition|remark|assumption|proof)\}\s*(.*)', line)
        if match:
            entity_type = match.group(1)
            name = match.group(2).strip()
            line_start = i + 1  # 1-indexed

            # Collect entity content until closing ::::
            content = []
            label = None
            i += 1

            while i < len(lines) and lines[i].strip() != ':::':
                content.append(lines[i])

                # Extract label
                label_match = re.match(r':label:\s*(.+)', lines[i].strip())
                if label_match:
                    label = label_match.group(1).strip()

                i += 1

            line_end = i + 1  # 1-indexed

            # Only add if labeled
            if label:
                entities.append(Entity(
                    label=label,
                    entity_type=entity_type,
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    content=content
                ))

        i += 1

    return entities

# Core concept mappings for high-priority references
CORE_CONCEPTS = {
    'def-walker': {
        'keywords': ['walker', 'walkers', r'\bw\b', r'w_i', r'w_1'],
        'priority': 1
    },
    'def-swarm-and-state-space': {
        'keywords': ['swarm', 'swarms', r'\$\\mathcal\{S\}\$', r'\\mathcal\{S\}',
                     r'\$\\Sigma_N\$', r'\\Sigma_N', 'swarm state space'],
        'priority': 1
    },
    'def-alive-dead-sets': {
        'keywords': ['alive set', 'dead set', r'\$\\mathcal\{A\}\$', r'\\mathcal\{A\}',
                     r'\$\\mathcal\{D\}\$', r'\\mathcal\{D\}', 'alive walkers', 'dead walkers'],
        'priority': 1
    },
    'def-valid-state-space': {
        'keywords': ['valid state space', 'valid domain'],
        'priority': 1
    },
    'axiom-guaranteed-revival': {
        'keywords': ['guaranteed revival', 'revival mechanism', 'resurrection'],
        'priority': 1
    },
    'axiom-boundary-regularity': {
        'keywords': ['boundary regularity', 'regular boundary'],
        'priority': 1
    },
    'def-valid-noise-measure': {
        'keywords': ['valid noise', 'noise measure'],
        'priority': 1
    },
    'def-algorithmic-space-generic': {
        'keywords': ['algorithmic space', r'\$\\mathcal\{Y\}\$', r'\\mathcal\{Y\}'],
        'priority': 1
    },
    'def-n-particle-displacement-metric': {
        'keywords': ['displacement metric', 'n-particle displacement', r'd_\{\\text\{Disp\}'],
        'priority': 2
    },
}

def find_backward_references(entity: Entity, all_entities: List[Entity]) -> List[BackwardReference]:
    """Find all backward reference opportunities for this entity"""
    refs = []

    # Only consider earlier entities
    earlier_entities = [e for e in all_entities if e.line_start < entity.line_start]

    content_lower = entity.content_text.lower()

    for earlier in earlier_entities:
        # Skip if already referenced
        if entity.has_reference_to(earlier.label):
            continue

        # Check core concepts
        if earlier.label in CORE_CONCEPTS:
            concept = CORE_CONCEPTS[earlier.label]
            for keyword in concept['keywords']:
                if keyword.startswith(r'\b') or keyword.startswith(r'\$'):
                    # Regex pattern
                    if re.search(keyword, entity.content_text):
                        refs.append(BackwardReference(
                            source_label=entity.label,
                            target_label=earlier.label,
                            target_type=earlier.entity_type,
                            reason=f"uses {earlier.name} concept/notation",
                            priority=concept['priority']
                        ))
                        break
                else:
                    # Simple substring match
                    if keyword.lower() in content_lower:
                        refs.append(BackwardReference(
                            source_label=entity.label,
                            target_label=earlier.label,
                            target_type=earlier.entity_type,
                            reason=f"mentions '{keyword}'",
                            priority=concept['priority']
                        ))
                        break

    return refs

def generate_enrichment_report(
    entities: List[Entity],
    all_refs: Dict[str, List[BackwardReference]]
) -> str:
    """Generate comprehensive enrichment report"""

    total_refs = sum(len(refs) for refs in all_refs.values())
    entities_with_refs = len([e for e in all_refs.values() if e])

    report = []
    report.append("=" * 80)
    report.append("BACKWARD CROSS-REFERENCE ENRICHMENT REPORT")
    report.append("Document: 01_fragile_gas_framework.md")
    report.append("=" * 80)
    report.append("")

    # Statistics
    report.append("## Summary Statistics")
    report.append("")
    report.append(f"- Total labeled entities: {len(entities)}")
    report.append(f"- Entities needing enrichment: {entities_with_refs}")
    report.append(f"- Total backward references identified: {total_refs}")

    if entities_with_refs > 0:
        avg = total_refs / entities_with_refs
        report.append(f"- Average references per entity: {avg:.2f}")

    report.append("")

    # Breakdown by priority
    priority_counts = defaultdict(int)
    for refs in all_refs.values():
        for ref in refs:
            priority_counts[ref.priority] += 1

    report.append("### By Priority")
    report.append(f"- High priority (core concepts): {priority_counts[1]}")
    report.append(f"- Medium priority: {priority_counts[2]}")
    report.append(f"- Low priority: {priority_counts[3]}")
    report.append("")

    # Most-referenced entities (targets)
    target_counts = defaultdict(int)
    for refs in all_refs.values():
        for ref in refs:
            target_counts[ref.target_label] += 1

    report.append("### Top 15 Most-Referenced Entities")
    report.append("")
    top_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for i, (label, count) in enumerate(top_targets, 1):
        # Find entity name
        entity = next((e for e in entities if e.label == label), None)
        name = entity.name if entity else "Unknown"
        report.append(f"{i}. `{label}` ({entity.entity_type if entity else '?'}): {count} references")
        report.append(f"   \"{name}\"")
    report.append("")

    # Detailed reference map (first 100)
    report.append("## Detailed Reference Map (First 100 Opportunities)")
    report.append("")

    count = 0
    for entity in entities:
        if entity.label not in all_refs or not all_refs[entity.label]:
            continue

        refs = all_refs[entity.label]
        report.append(f"### Source: `{entity.label}` (line {entity.line_start}-{entity.line_end})")
        report.append(f"**{entity.name}** ({entity.entity_type})")
        report.append("")

        for ref in refs:
            count += 1
            if count > 100:
                break

            priority_str = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}[ref.priority]
            report.append(f"**{count}.** Target: `{ref.target_label}` ({ref.target_type})")
            report.append(f"   - Priority: {priority_str}")
            report.append(f"   - Reason: {ref.reason}")
            report.append(f"   - Suggested: Add {{prf:ref}}`{ref.target_label}` at first mention")
            report.append("")

        if count > 100:
            report.append("... (remaining references omitted for brevity)")
            break

    return "\n".join(report)

def main():
    file_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    print(f"Reading document: {file_path}")
    lines = read_document(file_path)
    print(f"Total lines: {len(lines)}")

    print("\nExtracting entities...")
    entities = extract_entities(lines)
    print(f"Found {len(entities)} labeled entities")

    print("\nFinding backward references...")
    all_refs = {}
    for entity in entities:
        refs = find_backward_references(entity, entities)
        if refs:
            all_refs[entity.label] = refs

    print(f"Found backward references for {len(all_refs)} entities")

    print("\nGenerating report...")
    report = generate_enrichment_report(entities, all_refs)

    # Save report
    report_path = Path("/home/guillem/fragile/BACKWARD_REF_ANALYSIS_01.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print("\n" + "="*80)
    print(report)

if __name__ == "__main__":
    main()
