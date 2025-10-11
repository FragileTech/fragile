#!/usr/bin/env python3
"""
Extract all mathematical objects from fractal set documents and compile them
into a structured format for integration into 00_reference.md.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MathObject:
    """Represents a mathematical object (definition, theorem, etc.)"""
    type: str  # Definition, Theorem, Lemma, etc.
    label: str
    title: str
    source_file: str
    section: str
    tags: List[str]
    statement: str
    related_results: List[str]
    line_number: int


def extract_section_from_file(filepath: Path, line_num: int) -> str:
    """Extract the section heading that precedes a given line number."""
    lines = filepath.read_text().split('\n')
    section = "Unknown Section"

    # Walk backwards from line_num to find the most recent section heading
    for i in range(line_num - 1, -1, -1):
        line = lines[i]
        # Match markdown headings: ## or ### or ####
        if match := re.match(r'^(#{2,4})\s+(.+)$', line):
            section = match.group(2).strip()
            break

    return section


def extract_math_object(filepath: Path, start_line: int) -> MathObject:
    """Extract a complete mathematical object starting at the given line."""
    lines = filepath.read_text().split('\n')

    # Parse the opening directive (can be ::: or ::::)
    first_line = lines[start_line].strip()
    match = re.match(r':::+\{prf:(\w+)\}\s*(.*)$', first_line)
    if not match:
        raise ValueError(f"Invalid prf directive at line {start_line}: {first_line}")

    obj_type = match.group(1).title()  # definition -> Definition
    title = match.group(2).strip()

    # Extract label from the next line (typically :label: ...)
    label = ""
    content_start = start_line + 1
    if content_start < len(lines):
        label_line = lines[content_start].strip()
        if label_match := re.match(r':label:\s*(.+)$', label_line):
            label = label_match.group(1).strip()
            content_start += 1

    # Extract content until closing ::: or ::::
    content_lines = []
    i = content_start
    while i < len(lines):
        line = lines[i]
        if re.match(r':::+\s*$', line.strip()):
            break
        content_lines.append(line)
        i += 1

    statement = '\n'.join(content_lines).strip()

    # Extract section
    section = extract_section_from_file(filepath, start_line)

    # Generate tags based on filename and content
    tags = generate_tags(filepath.name, title, statement)

    # Extract related results (cross-references)
    related = extract_cross_references(statement)

    return MathObject(
        type=obj_type,
        label=label,
        title=title,
        source_file=filepath.name,
        section=section,
        tags=tags,
        statement=statement,
        related_results=related,
        line_number=start_line + 1  # 1-indexed for display
    )


def generate_tags(filename: str, title: str, statement: str) -> List[str]:
    """Generate relevant tags based on filename, title, and content."""
    tags = ['fractal-set']

    # File-based tags
    file_tags_map = {
        '01_fractal_set.md': ['data-structure', 'nodes', 'edges', 'cst', 'ig'],
        '02_computational_equivalence.md': ['computational-equivalence', 'episodes'],
        '03_yang_mills_noether.md': ['yang-mills', 'gauge-theory', 'noether', 'symmetry'],
        '04_rigorous_additions.md': ['proofs', 'rigor'],
        '05_qsd_stratonovich.md': ['qsd', 'stratonovich', 'sde'],
        '06_continuum_limit_theory.md': ['continuum-limit', 'laplacian'],
        '07_discrete_symmetries_gauge.md': ['discrete-symmetries', 'gauge-symmetry'],
        '08_lattice_qft_framework.md': ['lattice-qft', 'qft', 'field-theory'],
        '09_geometric_algorithms.md': ['geometric-algorithms', 'algorithms'],
        '10_areas_volumes_integration.md': ['riemannian-geometry', 'integration', 'volumes'],
    }

    for file_pattern, file_tags in file_tags_map.items():
        if file_pattern in filename:
            tags.extend(file_tags)

    # Content-based tags
    content_lower = (title + ' ' + statement).lower()

    keyword_tags = {
        'spinor': 'spinor',
        'causal': 'causal-tree',
        'gauge': 'gauge-theory',
        'u(1)': 'u1-symmetry',
        'su(2)': 'su2-symmetry',
        'su(3)': 'su3-symmetry',
        'so(10)': 'so10-gut',
        'fermion': 'fermionic',
        'qsd': 'qsd',
        'laplacian': 'laplacian',
        'riemannian': 'riemannian',
        'metric': 'metric-tensor',
        'curvature': 'curvature',
        'volume': 'volume',
        'area': 'area',
        'integration': 'integration',
        'convergence': 'convergence',
        'lattice': 'lattice',
        'field theory': 'field-theory',
        'symmetry': 'symmetry',
        'noether': 'noether',
        'conserv': 'conservation',
    }

    for keyword, tag in keyword_tags.items():
        if keyword in content_lower and tag not in tags:
            tags.append(tag)

    return sorted(set(tags))


def extract_cross_references(statement: str) -> List[str]:
    """Extract cross-references from the statement."""
    # Pattern: {prf:ref}`label-name`
    refs = re.findall(r'\{prf:ref\}`([^`]+)`', statement)
    return list(set(refs))


def format_for_reference(obj: MathObject) -> str:
    """Format a MathObject for inclusion in 00_reference.md."""
    output = []

    # Title
    output.append(f"### {obj.title}")
    output.append("")

    # Metadata
    output.append(f"**Type:** {obj.type}")
    output.append(f"**Label:** `{obj.label}`")
    output.append(f"**Source:** [13_fractal_set_new/{obj.source_file} § {obj.section}](13_fractal_set_new/{obj.source_file})")
    output.append(f"**Tags:** {', '.join(f'`{tag}`' for tag in obj.tags)}")
    output.append("")

    # Statement
    output.append("**Statement:**")
    output.append(obj.statement)
    output.append("")

    # Related results
    if obj.related_results:
        refs = ', '.join(f'`{ref}`' for ref in obj.related_results)
        output.append(f"**Related Results:** {refs}")
        output.append("")

    output.append("---")
    output.append("")

    return '\n'.join(output)


def process_file(filepath: Path) -> List[MathObject]:
    """Process a single file and extract all mathematical objects."""
    objects = []
    lines = filepath.read_text().split('\n')

    for i, line in enumerate(lines):
        # Match opening directive (can be ::: or ::::)
        if re.match(r':::+\{prf:(definition|theorem|lemma|proposition|axiom|corollary)\}', line):
            try:
                obj = extract_math_object(filepath, i)
                objects.append(obj)
            except Exception as e:
                print(f"Error extracting from {filepath.name} line {i+1}: {e}")

    return objects


def main():
    """Main extraction workflow."""
    # Define the documents to process in order
    doc_dir = Path('/home/guillem/fragile/docs/source/13_fractal_set_new')

    doc_files = [
        '01_fractal_set.md',
        '02_computational_equivalence.md',
        '03_yang_mills_noether.md',
        '04_rigorous_additions.md',
        '05_qsd_stratonovich_foundations.md',
        '06_continuum_limit_theory.md',
        '07_discrete_symmetries_gauge.md',
        '08_lattice_qft_framework.md',
        '09_geometric_algorithms.md',
        '10_areas_volumes_integration.md',
    ]

    all_objects = []

    for doc_file in doc_files:
        filepath = doc_dir / doc_file
        if not filepath.exists():
            print(f"Warning: {doc_file} not found")
            continue

        print(f"Processing {doc_file}...")
        objects = process_file(filepath)
        all_objects.extend(objects)
        print(f"  Found {len(objects)} mathematical objects")

    print(f"\nTotal: {len(all_objects)} mathematical objects extracted")

    # Group by source file
    by_file = {}
    for obj in all_objects:
        if obj.source_file not in by_file:
            by_file[obj.source_file] = []
        by_file[obj.source_file].append(obj)

    # Generate output
    output_lines = []

    # Order: Definitions, Lemmas, Propositions, Theorems, Corollaries, Axioms
    type_order = ['Definition', 'Axiom', 'Lemma', 'Proposition', 'Theorem', 'Corollary']

    # Header
    output_lines.append("# Fractal Set Theory - Mathematical Reference")
    output_lines.append("")
    output_lines.append("This document contains all mathematical definitions, theorems, lemmas, propositions, axioms, and corollaries from the Fractal Set theory documents (13_fractal_set_new/).")
    output_lines.append("")
    output_lines.append(f"**Total mathematical objects:** {len(all_objects)}")
    output_lines.append("")
    output_lines.append("**Source documents:**")
    for doc_file in doc_files:
        count = len(by_file.get(doc_file, []))
        output_lines.append(f"- {doc_file}: {count} objects")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")

    # Content organized by file
    for doc_file in doc_files:
        if doc_file not in by_file:
            continue

        objects = by_file[doc_file]
        output_lines.append(f"## {doc_file}")
        output_lines.append("")
        output_lines.append(f"**Objects in this document:** {len(objects)}")
        output_lines.append("")

        # Group by type
        by_type = {}
        for obj in objects:
            if obj.type not in by_type:
                by_type[obj.type] = []
            by_type[obj.type].append(obj)

        for obj_type in type_order:
            if obj_type not in by_type:
                continue

            type_objects = by_type[obj_type]
            output_lines.append(f"### {obj_type}s ({len(type_objects)})")
            output_lines.append("")

            for obj in type_objects:
                output_lines.append(format_for_reference(obj))

        output_lines.append("")

    # Write output
    output_file = Path('/home/guillem/fragile/FRACTAL_SET_REFERENCE.md')
    output_file.write_text('\n'.join(output_lines))
    print(f"\nOutput written to: {output_file}")

    # Also create a summary by type
    type_counts = {}
    for obj in all_objects:
        type_counts[obj.type] = type_counts.get(obj.type, 0) + 1

    print("\nSummary by type:")
    for obj_type in type_order:
        if obj_type in type_counts:
            print(f"  {obj_type}: {type_counts[obj_type]}")


if __name__ == '__main__':
    main()
