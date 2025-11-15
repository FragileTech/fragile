#!/usr/bin/env python3
"""
Backward Cross-Reference Enrichment for 01_fragile_gas_framework.md

CRITICAL CONSTRAINTS:
- Only reference concepts defined BEFORE current location (backward-only)
- Never add forward references
- This is document 01 (first), so NO cross-document references
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Entity:
    """Mathematical entity with metadata"""
    label: str
    entity_type: str
    name: str
    line_start: int
    line_end: int
    content: List[str]

    @property
    def content_text(self) -> str:
        return ''.join(self.content)

    def has_reference_to(self, target_label: str) -> bool:
        return f"{{prf:ref}}`{target_label}`" in self.content_text

def extract_entities(file_path: Path) -> List[Entity]:
    """Extract all labeled mathematical entities"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    entities = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Match entity start: :::{prf:TYPE} NAME
        match = re.match(r':::\{prf:(definition|theorem|lemma|axiom|corollary|proposition|remark|assumption|proof)\}\s*(.*)', line)
        if match:
            entity_type = match.group(1)
            name = match.group(2).strip()
            line_start = i + 1  # 1-indexed

            # Collect content until closing :::
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

            line_end = i + 1

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

# Core concepts for high-priority backward references
CORE_CONCEPTS = {
    'def-walker': {
        'keywords': [r'\bwalker\b', r'\bwalkers\b', r'\bw\b', r'\bw_i\b'],
        'priority': 1,
        'name': 'walker'
    },
    'def-swarm-and-state-space': {
        'keywords': [r'\bswarm\b', r'\\mathcal\{S\}', r'\\Sigma_N', 'swarm state space'],
        'priority': 1,
        'name': 'swarm'
    },
    'def-alive-dead-sets': {
        'keywords': ['alive set', 'dead set', r'\\mathcal\{A\}', r'\\mathcal\{D\}'],
        'priority': 1,
        'name': 'alive/dead sets'
    },
    'def-valid-state-space': {
        'keywords': ['valid state space', 'valid domain'],
        'priority': 1,
        'name': 'valid state space'
    },
    'axiom-guaranteed-revival': {
        'keywords': ['guaranteed revival', 'revival mechanism'],
        'priority': 1,
        'name': 'guaranteed revival'
    },
    'axiom-boundary-regularity': {
        'keywords': ['boundary regularity'],
        'priority': 1,
        'name': 'boundary regularity'
    },
    'def-valid-noise-measure': {
        'keywords': ['valid noise', 'noise measure'],
        'priority': 1,
        'name': 'valid noise'
    },
    'def-algorithmic-space-generic': {
        'keywords': ['algorithmic space', r'\\mathcal\{Y\}'],
        'priority': 1,
        'name': 'algorithmic space'
    },
}

def find_backward_refs(entity: Entity, all_entities: List[Entity]) -> List[Tuple[str, str, str, int]]:
    """
    Find backward reference opportunities.

    Returns: List of (target_label, target_type, reason, priority)
    """
    refs = []

    # Only consider earlier entities (BACKWARD-ONLY)
    earlier_entities = [e for e in all_entities if e.line_start < entity.line_start]

    content = entity.content_text
    content_lower = content.lower()

    for earlier in earlier_entities:
        # Skip if already referenced
        if entity.has_reference_to(earlier.label):
            continue

        # Check core concepts
        if earlier.label in CORE_CONCEPTS:
            concept = CORE_CONCEPTS[earlier.label]

            for keyword in concept['keywords']:
                # Check if keyword appears in content
                if keyword.startswith(r'\b') or keyword.startswith(r'\\'):
                    # Regex pattern
                    if re.search(keyword, content, re.IGNORECASE):
                        refs.append((
                            earlier.label,
                            earlier.entity_type,
                            f"uses {concept['name']} concept/notation",
                            concept['priority']
                        ))
                        break
                else:
                    # Simple substring
                    if keyword.lower() in content_lower:
                        refs.append((
                            earlier.label,
                            earlier.entity_type,
                            f"mentions '{keyword}'",
                            concept['priority']
                        ))
                        break

    return refs

def generate_report(entities: List[Entity], all_refs: Dict[str, List]) -> str:
    """Generate enrichment report"""

    total_refs = sum(len(refs) for refs in all_refs.values())
    entities_with_refs = len([r for r in all_refs.values() if r])

    report = []
    report.append("="*80)
    report.append("BACKWARD CROSS-REFERENCE ENRICHMENT REPORT")
    report.append("Document: 01_fragile_gas_framework.md")
    report.append("="*80)
    report.append("")

    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    report.append(f"- Total labeled entities: {len(entities)}")
    report.append(f"- Entities needing enrichment: {entities_with_refs}")
    report.append(f"- Total backward references identified: {total_refs}")

    if entities_with_refs > 0:
        avg = total_refs / entities_with_refs
        report.append(f"- Average references per entity: {avg:.2f}")

    report.append("")

    # Count by priority
    priority_counts = defaultdict(int)
    for refs in all_refs.values():
        for _, _, _, priority in refs:
            priority_counts[priority] += 1

    report.append("### By Priority")
    report.append(f"- High priority (core concepts): {priority_counts[1]}")
    report.append(f"- Medium priority: {priority_counts[2]}")
    report.append(f"- Low priority: {priority_counts[3]}")
    report.append("")

    # Most-referenced entities
    target_counts = defaultdict(int)
    for refs in all_refs.values():
        for target_label, _, _, _ in refs:
            target_counts[target_label] += 1

    report.append("### Top 15 Most-Referenced Entities")
    report.append("")
    top_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    entity_map = {e.label: e for e in entities}

    for i, (label, count) in enumerate(top_targets, 1):
        entity = entity_map.get(label)
        if entity:
            report.append(f"{i}. `{label}` ({entity.entity_type}): {count} references")
            report.append(f"   \"{entity.name}\"")
        else:
            report.append(f"{i}. `{label}`: {count} references")

    report.append("")

    # Detailed reference map
    report.append("## Detailed Reference Map (First 100)")
    report.append("")

    count = 0
    for entity in entities:
        if entity.label not in all_refs or not all_refs[entity.label]:
            continue

        refs = all_refs[entity.label]
        report.append(f"### Source: `{entity.label}` (line {entity.line_start}-{entity.line_end})")
        report.append(f"**{entity.name}** ({entity.entity_type})")
        report.append("")

        for target_label, target_type, reason, priority in refs:
            count += 1
            if count > 100:
                break

            priority_str = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}.get(priority, "UNKNOWN")
            report.append(f"**{count}.** Target: `{target_label}` ({target_type})")
            report.append(f"   - Priority: {priority_str}")
            report.append(f"   - Reason: {reason}")
            report.append(f"   - Suggested: Add {{prf:ref}}`{target_label}` at first mention")
            report.append("")

        if count > 100:
            report.append("... (remaining references omitted)")
            break

    return "\n".join(report)

def main():
    file_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    print(f"Reading: {file_path}")
    entities = extract_entities(file_path)
    print(f"Extracted {len(entities)} labeled entities")

    print("\nFinding backward references...")
    all_refs = {}
    for entity in entities:
        refs = find_backward_refs(entity, entities)
        if refs:
            all_refs[entity.label] = refs

    print(f"Found references for {len(all_refs)} entities")

    print("\nGenerating report...")
    report = generate_report(entities, all_refs)

    # Save report
    report_path = Path("/home/guillem/fragile/BACKWARD_REF_REPORT_01.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    print("\n" + report)

if __name__ == "__main__":
    main()
