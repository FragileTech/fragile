#!/usr/bin/env python3
"""
Strategic backward reference enrichment for 03_cloning.md

This script identifies high-value locations for adding backward references
and adds them systematically without over-cluttering the document.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json

# Comprehensive mapping of concepts to labels from docs 01 and 02
# Format: "concept phrase": ("label", "priority", "context")
# Priority: 1=high (fundamental concepts), 2=medium, 3=low (only if highly relevant)

BACKWARD_REFS = {
    # === HIGH PRIORITY: Fundamental Framework Concepts (Doc 01) ===
    "walker": ("def-walker", 1, "fundamental"),
    "swarm": ("def-swarm-and-state-space", 1, "fundamental"),
    "alive set": ("def-alive-dead-sets", 1, "status"),
    "dead set": ("def-alive-dead-sets", 1, "status"),
    "valid state space": ("def-valid-state-space", 1, "domain"),
    "Markov kernel": ("def-markov-kernel", 1, "stochastic"),
    "geometric ergodicity": ("def-geometric-ergodicity", 1, "convergence"),
    "Foster-Lyapunov": ("def-foster-lyapunov", 1, "convergence"),
    "quasi-stationary distribution": ("def-qsd", 1, "convergence"),
    "QSD": ("def-qsd", 1, "convergence"),
    "Fragile Gas axioms": ("def-fragile-gas-axioms", 1, "framework"),
    "hypocoercive Lyapunov": ("def-hypocoercive-lyapunov", 1, "analysis"),

    # === HIGH PRIORITY: Euclidean Gas Core (Doc 02) ===
    "Euclidean Gas": ("def-euclidean-gas", 1, "algorithm"),
    "kinetic operator": ("def-kinetic-operator", 1, "operator"),
    "Langevin operator": ("def-langevin-operator", 1, "operator"),
    "Langevin dynamics": ("def-langevin-operator", 1, "dynamics"),
    "cloning operator": ("def-cloning-operator", 1, "operator"),

    # === MEDIUM PRIORITY: Key Axioms (Doc 01) ===
    "Safe Harbor": ("axiom-safe-harbor", 2, "axiom"),
    "bounded domain": ("axiom-boundary-regularity", 2, "axiom"),
    "environmental richness": ("axiom-environmental-richness", 2, "axiom"),
    "reward regularity": ("axiom-reward-regularity", 2, "axiom"),
    "sufficient amplification": ("axiom-sufficient-amplification", 2, "axiom"),
    "geometric consistency": ("axiom-geometric-consistency", 2, "axiom"),

    # === MEDIUM PRIORITY: Measurement Infrastructure (Doc 01) ===
    "algorithmic distance": ("def-alg-distance", 2, "measurement"),
    "raw value operator": ("def-raw-value-operator", 2, "pipeline"),
    "swarm aggregation": ("def-swarm-aggregation-operator-axiomatic", 2, "pipeline"),
    "standardization operator": ("def-standardization-operator-n-dimensional", 2, "pipeline"),
    "rescale function": ("def-canonical-logistic-rescale-function-example", 2, "pipeline"),
    "companion selection": ("def-companion-selection-measure", 2, "pairing"),

    # === LOW PRIORITY: Technical Details ===
    "perturbation measure": ("def-perturbation-measure", 3, "noise"),
    "cloning measure": ("def-cloning-measure", 3, "noise"),
    "Wasserstein distance": ("def-n-particle-displacement-metric", 3, "metric"),
}

class BackwardReferenceEnricher:
    def __init__(self, doc_path: Path):
        self.doc_path = doc_path
        self.lines = []
        self.enriched_lines = []
        self.stats = {
            'original_refs': 0,
            'added_refs': 0,
            'by_priority': {1: 0, 2: 0, 3: 0},
            'by_section': {}
        }

    def load_document(self):
        """Load the document into memory."""
        with open(self.doc_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.enriched_lines = self.lines.copy()
        self.stats['original_refs'] = sum(1 for line in self.lines if '{prf:ref}' in line)
        print(f"Loaded {len(self.lines)} lines with {self.stats['original_refs']} existing references")

    def identify_mathematical_entities(self) -> List[Tuple[int, str, str]]:
        """
        Find all mathematical entities (definitions, theorems, lemmas, etc.)
        Returns list of (line_number, entity_type, label) tuples.
        """
        entities = []
        in_entity = False
        entity_start = 0
        entity_type = ""
        entity_label = ""

        for i, line in enumerate(self.lines):
            # Check for directive start
            match = re.match(r':::+\{prf:(definition|theorem|lemma|proposition|corollary|axiom|algorithm)', line)
            if match:
                in_entity = True
                entity_start = i
                entity_type = match.group(1)

            # Check for label
            if in_entity and ':label:' in line:
                label_match = re.search(r':label:\s*(\S+)', line)
                if label_match:
                    entity_label = label_match.group(1)

            # Check for directive end
            if in_entity and re.match(r':::+\s*$', line):
                if entity_label:
                    entities.append((entity_start, entity_type, entity_label))
                in_entity = False
                entity_label = ""

        print(f"Found {len(entities)} mathematical entities")
        return entities

    def should_add_reference(self, line_num: int, concept: str, label: str, priority: int) -> bool:
        """
        Determine if a reference should be added at this location.

        Rules:
        1. Never add if line already has this reference
        2. Never add if line is a label definition
        3. Never add if line is a directive opening
        4. For priority 3, only add in high-value contexts (entity statements)
        5. Avoid over-referencing (max 2-3 refs per entity)
        """
        line = self.enriched_lines[line_num]

        # Rule 1: Skip if already referenced
        if f"{{prf:ref}}`{label}`" in line:
            return False

        # Rule 2: Skip label definitions
        if ':label:' in line:
            return False

        # Rule 3: Skip directive openings
        if re.match(r':::+\{', line):
            return False

        # Rule 4: For low priority, check context
        if priority == 3:
            # Only add in entity content (not in proofs or remarks)
            # Check if we're in a high-value entity
            context_start = max(0, line_num - 5)
            context = ''.join(self.enriched_lines[context_start:line_num+1])
            if '{prf:proof}' in context or '{prf:remark}' in context:
                return False

        # Rule 5: Check reference density in nearby lines
        refs_nearby = sum(1 for i in range(max(0, line_num-3), min(len(self.enriched_lines), line_num+4))
                         if '{prf:ref}' in self.enriched_lines[i])
        if refs_nearby > 5:  # Too many references nearby
            return False

        return True

    def add_reference_natural(self, line: str, concept: str, label: str) -> Tuple[str, bool]:
        """
        Add a reference naturally to a line.
        Returns (modified_line, was_modified).
        """
        # Skip if already has reference or is in skip context
        if f"{{prf:ref}}`{label}`" in line or ':label:' in line or re.match(r':::+\{', line):
            return line, False

        # Find the concept (case-insensitive, word boundaries)
        pattern = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
        match = pattern.search(line)

        if not match:
            return line, False

        # Check if already in a reference context
        start = match.start()
        context_before = line[max(0, start-15):start]
        context_after = line[match.end():match.end()+15]

        if '{prf:ref}' in context_before + context_after:
            return line, False

        # Add reference
        matched_text = match.group()
        replacement = f"{matched_text} ({{prf:ref}}`{label}`)"

        new_line = line[:start] + replacement + line[match.end():]
        return new_line, True

    def enrich_entity(self, entity_start: int, entity_type: str, entity_label: str) -> int:
        """
        Enrich a single mathematical entity with backward references.
        Returns number of references added.
        """
        # Find entity end
        entity_end = entity_start + 1
        while entity_end < len(self.lines) and not re.match(r':::+\s*$', self.lines[entity_end]):
            entity_end += 1

        refs_added = 0
        max_refs_per_entity = 5 if entity_type in ['theorem', 'definition', 'lemma'] else 3

        # Sort concepts by priority and length (longer first to handle "Euclidean Gas" before "Gas")
        concepts = sorted(
            [(concept, label, priority, ctx) for concept, (label, priority, ctx) in BACKWARD_REFS.items()],
            key=lambda x: (x[2], -len(x[0]))  # Sort by priority, then by length descending
        )

        for line_num in range(entity_start, entity_end + 1):
            if refs_added >= max_refs_per_entity:
                break

            for concept, label, priority, context in concepts:
                if refs_added >= max_refs_per_entity:
                    break

                if self.should_add_reference(line_num, concept, label, priority):
                    new_line, modified = self.add_reference_natural(
                        self.enriched_lines[line_num], concept, label
                    )
                    if modified:
                        self.enriched_lines[line_num] = new_line
                        refs_added += 1
                        self.stats['added_refs'] += 1
                        self.stats['by_priority'][priority] += 1

        return refs_added

    def enrich_document(self):
        """Main enrichment workflow."""
        print("\n=== Starting backward reference enrichment ===\n")

        # Identify all mathematical entities
        entities = self.identify_mathematical_entities()

        # Enrich each entity
        section_counts = {}
        for entity_start, entity_type, entity_label in entities:
            refs_added = self.enrich_entity(entity_start, entity_type, entity_label)
            if refs_added > 0:
                section = entity_label.split('-')[0]  # e.g., "def", "thm", "lem"
                section_counts[section] = section_counts.get(section, 0) + refs_added
                print(f"  {entity_type:12s} {entity_label:40s} → +{refs_added} refs")

        self.stats['by_section'] = section_counts

    def save_enriched_document(self, backup: bool = True):
        """Save the enriched document."""
        if backup:
            backup_path = self.doc_path.with_suffix('.md.backup_strategic_enrichment')
            with open(self.doc_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            print(f"\nBackup created: {backup_path}")

        with open(self.doc_path, 'w', encoding='utf-8') as f:
            f.writelines(self.enriched_lines)

        print(f"Enriched document saved: {self.doc_path}")

    def print_statistics(self):
        """Print enrichment statistics."""
        total_refs = sum(1 for line in self.enriched_lines if '{prf:ref}' in line)

        print("\n" + "="*70)
        print("ENRICHMENT STATISTICS")
        print("="*70)
        print(f"Original references:     {self.stats['original_refs']:4d}")
        print(f"References added:        {self.stats['added_refs']:4d}")
        print(f"Total references:        {total_refs:4d}")
        print(f"\nBy priority:")
        print(f"  High priority (1):     {self.stats['by_priority'][1]:4d}")
        print(f"  Medium priority (2):   {self.stats['by_priority'][2]:4d}")
        print(f"  Low priority (3):      {self.stats['by_priority'][3]:4d}")
        print(f"\nBy entity type:")
        for section, count in sorted(self.stats['by_section'].items()):
            print(f"  {section}:                    {count:4d}")
        print("="*70)

def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md")

    enricher = BackwardReferenceEnricher(doc_path)
    enricher.load_document()
    enricher.enrich_document()
    enricher.save_enriched_document(backup=True)
    enricher.print_statistics()

    print("\n✓ Strategic backward reference enrichment complete!")

if __name__ == "__main__":
    main()
