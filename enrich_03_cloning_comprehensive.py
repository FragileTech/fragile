#!/usr/bin/env python3
"""
Comprehensive backward reference enrichment for 03_cloning.md
Adds {prf:ref} links to concepts defined in 01_fragile_gas_framework.md and 02_euclidean_gas.md
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Key concepts from documents 01 and 02 with their labels
PREVIOUS_CONCEPTS = {
    # Document 01: Core framework concepts
    "walker": "def-walker",
    "walkers": "def-walker",
    "swarm": "def-swarm-and-state-space",
    "swarm configuration": "def-swarm-and-state-space",
    "alive set": "def-alive-dead-sets",
    "dead set": "def-alive-dead-sets",
    "valid state space": "def-valid-state-space",
    "valid domain": "def-valid-state-space",
    "Markov kernel": "def-markov-kernel",
    "geometric ergodicity": "def-geometric-ergodicity",
    "Foster-Lyapunov": "def-foster-lyapunov",
    "quasi-stationary distribution": "def-qsd",
    "QSD": "def-qsd",
    "Fragile Gas axioms": "def-fragile-gas-axioms",
    "hypocoercive Lyapunov": "def-hypocoercive-lyapunov",

    # Document 02: Euclidean Gas concepts
    "Euclidean Gas": "def-euclidean-gas",
    "kinetic operator": "def-kinetic-operator",
    "Langevin operator": "def-langevin-operator",
    "Langevin dynamics": "def-langevin-operator",
    "cloning operator": "def-cloning-operator",
    "BAOAB integrator": "alg-baoab-integrator",

    # Axioms from document 01
    "Safe Harbor": "axiom-safe-harbor",
    "bounded domain": "axiom-boundary-regularity",
    "environmental richness": "axiom-environmental-richness",
    "reward regularity": "axiom-reward-regularity",
}

# Patterns to avoid (already have references or are in special contexts)
SKIP_PATTERNS = [
    r'\{prf:ref\}',  # Already has a reference
    r':label:',      # Label definition line
    r':::\{',        # Directive opening
    r'^\s*#',        # Header lines
]

def should_skip_line(line: str) -> bool:
    """Check if a line should be skipped for enrichment."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, line):
            return True
    return False

def find_concept_in_line(line: str, concept: str, label: str) -> List[Tuple[int, int, str]]:
    """
    Find occurrences of a concept in a line that don't already have references.
    Returns list of (start_pos, end_pos, matched_text) tuples.
    """
    if should_skip_line(line):
        return []

    # Check if line already references this label
    if f"{{prf:ref}}`{label}`" in line:
        return []

    # Case-insensitive search for the concept
    pattern = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
    matches = []

    for match in pattern.finditer(line):
        start, end = match.span()
        # Check context - avoid adding reference if already in a reference
        context_start = max(0, start - 20)
        context = line[context_start:end+10]
        if '{prf:ref}' not in context:
            matches.append((start, end, match.group()))

    return matches

def add_reference_to_line(line: str, concept: str, label: str, max_refs_per_line: int = 1) -> str:
    """
    Add a backward reference to the first occurrence of concept in line.
    Only adds one reference per line to avoid cluttering.
    """
    matches = find_concept_in_line(line, concept, label)

    if not matches or len(matches) == 0:
        return line

    # Only add reference to first occurrence
    start, end, matched_text = matches[0]

    # Determine natural placement style
    # Check if the concept is in a natural phrase
    before = line[:start].rstrip()
    after = line[end:].lstrip()

    # If followed by punctuation or end of sentence, use parenthetical style
    if after and after[0] in ',.;:)':
        # Parenthetical: "the walker ({prf:ref}`def-walker`)"
        replacement = f"{matched_text} ({{prf:ref}}`{label}`)"
    else:
        # Inline: "the walker ({prf:ref}`def-walker`) state"
        replacement = f"{matched_text} ({{prf:ref}}`{label}`)"

    return line[:start] + replacement + line[end:]

def enrich_section(lines: List[str], start_line: int, end_line: int) -> List[str]:
    """
    Enrich a specific section with backward references.
    Returns modified lines.
    """
    enriched_lines = lines.copy()
    concepts_added = set()

    # Sort concepts by length (longest first) to handle "Euclidean Gas" before "Gas"
    sorted_concepts = sorted(PREVIOUS_CONCEPTS.items(), key=lambda x: len(x[0]), reverse=True)

    for i in range(start_line, min(end_line, len(lines))):
        original_line = enriched_lines[i]
        modified = False

        # Try to add references for each concept (max 1-2 per line)
        refs_added_this_line = 0
        max_refs = 2

        for concept, label in sorted_concepts:
            if refs_added_this_line >= max_refs:
                break

            # Skip if we already added this concept recently in this section
            if f"{label}_{i//50}" in concepts_added:  # Track by 50-line chunks
                continue

            new_line = add_reference_to_line(enriched_lines[i], concept, label)
            if new_line != enriched_lines[i]:
                enriched_lines[i] = new_line
                concepts_added.add(f"{label}_{i//50}")
                refs_added_this_line += 1
                modified = True

        if modified:
            print(f"Line {i+1}: Added {refs_added_this_line} reference(s)")

    return enriched_lines

def process_document(doc_path: Path, output_path: Path):
    """Process the entire document section by section."""
    print(f"Reading document: {doc_path}")

    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total lines: {total_lines}")

    # Key sections to enrich (based on grep analysis)
    # Focus on sections with high conceptual density
    sections = [
        (0, 200, "Introduction and foundations"),
        (200, 500, "Coupled state space and barrier function"),
        (500, 1000, "Lyapunov function and coercivity"),
        (1000, 1400, "Foundational axioms"),
        (1400, 2400, "Measurement pipeline"),
        (2400, 3400, "Geometry of error"),
        (3400, 4800, "Corrective nature of fitness"),
        (4800, 5900, "Keystone lemma"),
        (5900, 6400, "Cloning operator definition"),
        (6400, 7100, "Variance drift analysis"),
        (7100, 7900, "Boundary potential analysis"),
        (7900, 8710, "Synergistic drift and conclusion"),
    ]

    print("\nEnriching document sections...")
    enriched_lines = lines.copy()

    for start, end, desc in sections:
        print(f"\n=== Processing: {desc} (lines {start+1}-{end}) ===")
        enriched_lines = enrich_section(enriched_lines, start, end)

    # Write enriched document
    print(f"\nWriting enriched document to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(enriched_lines)

    print("\n✓ Enrichment complete!")

    # Generate statistics
    original_refs = sum(1 for line in lines if '{prf:ref}' in line)
    enriched_refs = sum(1 for line in enriched_lines if '{prf:ref}' in line)
    added_refs = enriched_refs - original_refs

    print(f"\nStatistics:")
    print(f"  Original references: {original_refs}")
    print(f"  Enriched references: {enriched_refs}")
    print(f"  References added: {added_refs}")

def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md")
    output_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md")

    # Create backup
    backup_path = doc_path.with_suffix('.md.backup_comprehensive_enrichment')
    print(f"Creating backup: {backup_path}")

    with open(doc_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())

    # Process document
    process_document(doc_path, output_path)

if __name__ == "__main__":
    main()
