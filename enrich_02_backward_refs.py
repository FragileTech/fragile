#!/usr/bin/env python3
"""
Backward Cross-Reference Enrichment for 02_euclidean_gas.md

This script adds strategic backward references from 02_euclidean_gas.md to:
1. Earlier entities within the same document (within-document backward refs)
2. Entities from 01_fragile_gas_framework.md (cross-document backward refs)

Follows strict temporal ordering: only references to EARLIER definitions.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

@dataclass
class Entity:
    """Represents a mathematical entity with label and line number."""
    label: str
    entity_type: str
    line_number: int
    name: str

class BackwardReferenceEnricher:
    """Enriches markdown with backward cross-references."""

    def __init__(self, doc_path: Path, glossary_path: Path):
        self.doc_path = doc_path
        self.glossary_path = glossary_path
        self.doc_content = doc_path.read_text()
        self.lines = self.doc_content.split('\n')

        # Track entities and existing references
        self.doc_entities: List[Entity] = []
        self.doc_01_labels: Set[str] = set()
        self.existing_refs: Set[str] = set()

        self._parse_document()
        self._parse_glossary()
        self._extract_existing_refs()

    def _parse_document(self):
        """Extract all entities from current document with line numbers."""
        pattern = r':::?\{prf:(definition|theorem|lemma|axiom|algorithm|proposition|corollary|remark)'
        label_pattern = r':label:\s+(\S+)'

        current_entity_line = None
        current_entity_type = None

        for i, line in enumerate(self.lines, 1):
            # Find directive start
            match = re.search(pattern, line)
            if match:
                current_entity_line = i
                current_entity_type = match.group(1)

            # Find label within directive
            if current_entity_line and i <= current_entity_line + 10:
                label_match = re.search(label_pattern, line)
                if label_match:
                    label = label_match.group(1)
                    # Try to extract name from next few lines
                    name = self._extract_entity_name(i, current_entity_type)
                    self.doc_entities.append(Entity(
                        label=label,
                        entity_type=current_entity_type,
                        line_number=current_entity_line,
                        name=name
                    ))
                    current_entity_line = None
                    current_entity_type = None

    def _extract_entity_name(self, start_line: int, entity_type: str) -> str:
        """Extract entity name from lines after label."""
        for i in range(start_line, min(start_line + 5, len(self.lines))):
            line = self.lines[i].strip()
            if line and not line.startswith(':') and not line.startswith(':::'):
                # Clean up markdown formatting
                line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
                line = re.sub(r'\$(.+?)\$', '', line)
                return line[:60]  # Truncate long names
        return f"Unnamed {entity_type}"

    def _parse_glossary(self):
        """Extract all labels from document 01 in glossary."""
        glossary_content = self.glossary_path.read_text()

        # Find section for 01_fragile_gas_framework.md
        in_doc_01 = False
        for line in glossary_content.split('\n'):
            if '01_fragile_gas_framework' in line and '###' in line:
                in_doc_01 = True
                continue
            elif in_doc_01 and '##' in line and 'Source:' in line:
                # New source section, stop
                in_doc_01 = False

            if in_doc_01:
                # Extract label
                label_match = re.search(r'`([a-z0-9\-_]+)`', line)
                if label_match and '**Label:**' in line:
                    self.doc_01_labels.add(label_match.group(1))

    def _extract_existing_refs(self):
        """Extract all existing {prf:ref} references."""
        pattern = r'\{prf:ref\}`([a-z0-9\-_]+)`'
        for match in re.finditer(pattern, self.doc_content):
            self.existing_refs.add(match.group(1))

    def identify_missing_references(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """
        Identify strategic missing backward references.

        Returns:
            Dict mapping line_number to list of (label, concept, reason) tuples
        """
        missing_refs = {}

        # Key concepts to link (concept_text -> target_label)
        concept_map = {
            # Core framework concepts
            'walker': 'def-walker',
            'walkers': 'def-walker',
            'swarm state': 'def-swarm-and-state-space',
            'swarm configuration': 'def-swarm-and-state-space',
            'alive set': 'def-alive-dead-sets',
            'dead set': 'def-alive-dead-sets',
            'alive and dead sets': 'def-alive-dead-sets',
            'cemetery state': 'def-cemetery-state',
            'algorithmic space': 'def-algorithmic-space-generic',
            'algorithmic distance': 'def-alg-distance',

            # Axioms (most critical for framework compliance)
            'guaranteed revival': 'def-axiom-guaranteed-revival',
            'axiom of guaranteed revival': 'def-axiom-guaranteed-revival',
            'boundary regularity': 'def-axiom-boundary-regularity',
            'axiom of boundary regularity': 'def-axiom-boundary-regularity',
            'boundary smoothness': 'def-axiom-boundary-smoothness',
            'environmental richness': 'def-axiom-environmental-richness',
            'axiom of environmental richness': 'def-axiom-environmental-richness',
            'reward regularity': 'def-axiom-reward-regularity',
            'axiom of reward regularity': 'def-axiom-reward-regularity',
            'bounded algorithmic diameter': 'def-axiom-bounded-algorithmic-diameter',
            'axiom of bounded algorithmic diameter': 'def-axiom-bounded-algorithmic-diameter',
            'sufficient amplification': 'def-axiom-sufficient-amplification',
            'non-degenerate noise': 'def-axiom-non-degenerate-noise',
            'geometric consistency': 'def-axiom-geometric-consistency',
            'axiom of geometric consistency': 'def-axiom-geometric-consistency',
            'bounded variance production': 'def-axiom-bounded-variance-production',
            'projection compatibility': 'def-axiom-projection-compatibility',

            # Measurement operators
            'perturbation measure': 'def-perturbation-measure',
            'cloning measure': 'def-cloning-measure',
            'distance measurement': 'def-distance-positional-measures',
            'reward measurement': 'def-reward-measurement',
            'standardization operator': 'def-statistical-properties-measurement',
            'patched standardization': 'def-statistical-properties-measurement',
            'statistical properties measurement': 'def-statistical-properties-measurement',

            # Fragile Gas algorithm
            'fragile gas algorithm': 'def-fragile-gas-algorithm',
            'fragile swarm instantiation': 'def-fragile-swarm-instantiation',

            # Status and operators
            'status update operator': 'def-status-update-operator',
            'in-step independence': 'def-assumption-instep-independence',
            'assumption a': 'def-assumption-instep-independence',

            # Canonical rescale
            'canonical logistic rescale': 'def-canonical-logistic-rescale-function-example',
            'logistic rescale function': 'def-canonical-logistic-rescale-function-example',

            # Reference measures
            'ambient euclidean structure': 'def-ambient-euclidean',
            'reference measures': 'def-reference-measures',
        }

        # Scan document for concept mentions WITHOUT existing references
        for i, line in enumerate(self.lines, 1):
            # Skip lines already containing references
            if '{prf:ref}' in line:
                continue

            # Skip directive definitions themselves
            if re.search(r':::?\{prf:', line):
                continue

            line_lower = line.lower()

            # Check each concept
            for concept, target_label in concept_map.items():
                # Skip if already referenced
                if target_label in self.existing_refs:
                    continue

                # Check if concept appears in line
                if concept in line_lower:
                    # Verify this is from doc 01 or earlier in doc 02
                    if target_label in self.doc_01_labels:
                        source = "01_fragile_gas_framework.md"
                    else:
                        # Check if it's earlier in this document
                        matching_entities = [e for e in self.doc_entities if e.label == target_label and e.line_number < i]
                        if not matching_entities:
                            continue  # Not a backward reference
                        source = "02_euclidean_gas.md (earlier)"

                    if i not in missing_refs:
                        missing_refs[i] = []

                    missing_refs[i].append((target_label, concept, source))

        return missing_refs

    def generate_report(self, missing_refs: Dict[str, List[Tuple[int, str, str]]]) -> str:
        """Generate detailed enrichment report."""
        report = ["# Backward Cross-Reference Enrichment Report"]
        report.append(f"\n**Document:** {self.doc_path.name}")
        report.append(f"**Total entities in document:** {len(self.doc_entities)}")
        report.append(f"**Existing references:** {len(self.existing_refs)}")
        report.append(f"**Missing strategic references identified:** {len(missing_refs)}")

        # Categorize by source
        doc_01_refs = [ref for refs in missing_refs.values() for ref in refs if "01_" in ref[2]]
        within_doc_refs = [ref for refs in missing_refs.values() for ref in refs if "02_" in ref[2]]

        report.append(f"\n## Summary")
        report.append(f"- Cross-document refs (to 01): {len(doc_01_refs)}")
        report.append(f"- Within-document refs: {len(within_doc_refs)}")

        # Top missing references
        from collections import Counter
        all_refs = [ref[0] for refs in missing_refs.values() for ref in refs]
        top_missing = Counter(all_refs).most_common(20)

        report.append(f"\n## Top 20 Missing References")
        for label, count in top_missing:
            source = "01" if label in self.doc_01_labels else "02"
            report.append(f"- `{label}` ({count} locations) [from doc {source}]")

        # Sample locations
        report.append(f"\n## Sample Missing References (first 30)")
        for i, (line_num, refs) in enumerate(sorted(missing_refs.items())[:30], 1):
            report.append(f"\n### {i}. Line {line_num}")
            report.append(f"**Context:** `{self.lines[line_num-1][:80]}...`")
            for label, concept, source in refs:
                report.append(f"- Add `{{prf:ref}}\`{label}\`` for concept '{concept}' (from {source})")

        return '\n'.join(report)

def main():
    """Main execution."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas.md")
    glossary_path = Path("/home/guillem/fragile/docs/glossary.md")

    print("=" * 80)
    print("BACKWARD CROSS-REFERENCE ENRICHMENT ANALYSIS")
    print("=" * 80)

    enricher = BackwardReferenceEnricher(doc_path, glossary_path)

    print(f"\nDocument parsed: {len(enricher.doc_entities)} entities found")
    print(f"Document 01 labels loaded: {len(enricher.doc_01_labels)}")
    print(f"Existing references: {len(enricher.existing_refs)}")

    print("\n" + "=" * 80)
    print("IDENTIFYING MISSING BACKWARD REFERENCES...")
    print("=" * 80)

    missing_refs = enricher.identify_missing_references()

    print(f"\nFound {len(missing_refs)} locations with missing references")

    # Generate report
    report = enricher.generate_report(missing_refs)

    report_path = Path("/home/guillem/fragile/BACKWARD_REF_ANALYSIS_02.md")
    report_path.write_text(report)
    print(f"\nDetailed report written to: {report_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the report to identify high-value missing references")
    print("2. Run targeted enrichment to add strategic backward references")
    print("3. Prioritize:")
    print("   - Framework axiom references (compliance verification)")
    print("   - Core concept definitions (walker, swarm, algorithmic space)")
    print("   - Operator specifications (measurement, perturbation, status)")

if __name__ == '__main__':
    main()
