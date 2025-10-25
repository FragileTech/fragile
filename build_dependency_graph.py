#!/usr/bin/env python3
"""
Build dependency graph for theorems in docs/source/2_geometric_gas/
that need proofs.
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict, deque

# Load theorems from CSV
def load_theorems(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Load glossary to check for external dependencies
def load_glossary(glossary_path):
    """Load all labels from glossary"""
    with open(glossary_path, 'r') as f:
        content = f.read()

    # Extract all labels from glossary entries
    # Pattern: **Label**: `label-name`
    pattern = r'\*\*Label\*\*:\s*`([^`]+)`'
    labels = set(re.findall(pattern, content))
    return labels

# Extract theorem content and find dependencies
def extract_dependencies(doc_path, line_num, label):
    """Read theorem and surrounding context, extract prf:ref dependencies"""
    try:
        with open(doc_path, 'r') as f:
            lines = f.readlines()

        # Convert line_num to 0-indexed
        line_idx = int(line_num) - 1

        # Read context: from line_num until we hit the next theorem/lemma/proposition directive
        # or end of current proof block, with max 300 lines safety limit
        start_idx = max(0, line_idx)
        max_end_idx = min(len(lines), line_idx + 300)

        # Find the end of current block - stop at next {prf:theorem}/{prf:lemma}/{prf:proposition}
        # but only after we've closed the current one (look for :::)
        end_idx = max_end_idx
        in_current_block = True
        lines_since_close = 0

        for i in range(start_idx + 1, max_end_idx):
            line = lines[i]

            # Check if we've closed the current directive block
            if in_current_block and line.strip() == ':::':
                in_current_block = False
                lines_since_close = 0
                continue

            # After closing, if we hit another theorem/lemma/proposition, stop
            if not in_current_block:
                lines_since_close += 1
                # Allow some buffer (e.g., 20 lines) after closing for admonitions
                # but stop at next main theorem directive
                if lines_since_close > 20 and re.match(r':::\{prf:(theorem|lemma|proposition|corollary)', line):
                    end_idx = i
                    break

        context = ''.join(lines[start_idx:end_idx])

        # Extract all {prf:ref} references
        # Patterns: {prf:ref}`label` or {prf:ref}`text <label>`
        ref_pattern = r'\{prf:ref\}`(?:.*?<)?([^>`]+?)(?:>)?`'
        refs = re.findall(ref_pattern, context)

        all_refs = list(set(refs))  # Remove duplicates

        return all_refs

    except Exception as e:
        print(f"Error reading {doc_path}:{line_num} ({label}): {e}")
        return []

def classify_dependency(ref_label, theorems_dict, glossary_labels, current_doc):
    """Classify a dependency as: in_current_pipeline, satisfied_externally, or truly_missing"""

    # Check if it's in current pipeline
    if ref_label in theorems_dict:
        thm = theorems_dict[ref_label]
        return {
            'label': ref_label,
            'status': 'in_current_pipeline',
            'source': thm['Document'],
            'line': thm['Line'],
            'will_be_proven': thm['Status'] in ['needs_proof', 'has_sketch'],
            'current_status': thm['Status']
        }

    # Check if it's in glossary (external framework)
    if ref_label in glossary_labels:
        return {
            'label': ref_label,
            'status': 'satisfied_externally',
            'source': 'framework_glossary'
        }

    # Not found anywhere - truly missing
    return {
        'label': ref_label,
        'status': 'truly_missing',
        'source': None
    }

def build_dependency_graph(csv_path, docs_base_path, glossary_path):
    """Build complete dependency graph"""

    # Load data
    theorems = load_theorems(csv_path)
    glossary_labels = load_glossary(glossary_path)

    # Filter to theorems needing proof
    needs_proof = [t for t in theorems if t['Status'] in ['needs_proof', 'has_sketch']]

    # Create lookup dict for all theorems
    theorems_dict = {t['Label']: t for t in theorems}

    # Build dependency records
    dependency_graph = []

    print(f"Processing {len(needs_proof)} theorems that need proofs...")

    for i, thm in enumerate(needs_proof, 1):
        label = thm['Label']
        doc = thm['Document']
        line = thm['Line']

        if i % 10 == 0:
            print(f"  Processed {i}/{len(needs_proof)} theorems...")

        # Construct full path
        doc_path = docs_base_path / doc

        # Extract dependencies
        raw_deps = extract_dependencies(doc_path, line, label)

        # Filter out self-references
        raw_deps = [dep for dep in raw_deps if dep != label]

        # Classify each dependency
        dependencies = []
        truly_missing = []

        for dep_label in raw_deps:
            dep_info = classify_dependency(dep_label, theorems_dict, glossary_labels, doc)
            dependencies.append(dep_info)

            if dep_info['status'] == 'truly_missing':
                truly_missing.append(dep_label)

        # Create record
        record = {
            'label': label,
            'document': doc,
            'line': int(line),
            'type': thm['Type'],
            'title': thm['Title'],
            'status': thm['Status'],
            'dependencies': dependencies,
            'truly_missing_deps': truly_missing
        }

        dependency_graph.append(record)

    return dependency_graph

def topological_sort(dependency_graph):
    """
    Perform topological sort on theorems based on dependencies.
    Returns: (sorted_order, cycles)
    """

    # Build adjacency list (only for in_current_pipeline dependencies)
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    # All theorems that need proof
    all_labels = {item['label'] for item in dependency_graph}

    # Initialize in_degree
    for label in all_labels:
        in_degree[label] = 0

    # Build graph
    for item in dependency_graph:
        label = item['label']
        for dep in item['dependencies']:
            if dep['status'] == 'in_current_pipeline' and dep['will_be_proven']:
                # dep['label'] must be proven before 'label'
                graph[dep['label']].append(label)
                in_degree[label] += 1

    # Kahn's algorithm for topological sort
    queue = deque([label for label in all_labels if in_degree[label] == 0])
    sorted_order = []

    while queue:
        node = queue.popleft()
        sorted_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    cycles = []
    if len(sorted_order) < len(all_labels):
        # There are cycles - find them
        remaining = all_labels - set(sorted_order)
        cycles = list(remaining)

    return sorted_order, cycles

def main():
    csv_path = Path('/home/guillem/fragile/geometric_gas_theorems.csv')
    docs_base = Path('/home/guillem/fragile/docs/source/2_geometric_gas')
    glossary_path = Path('/home/guillem/fragile/docs/glossary.md')
    output_path = Path('/home/guillem/fragile/docs/source/2_geometric_gas/theorem_dependencies.json')

    # Build dependency graph
    dep_graph = build_dependency_graph(csv_path, docs_base, glossary_path)

    # Topological sort
    sorted_order, cycles = topological_sort(dep_graph)

    # Create output
    output = {
        'metadata': {
            'total_theorems_needing_proof': len(dep_graph),
            'topological_sort_successful': len(cycles) == 0,
            'cycles_detected': cycles
        },
        'execution_order': sorted_order,
        'dependency_graph': dep_graph
    }

    # Write output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Dependency graph written to {output_path}")
    print(f"\nSummary:")
    print(f"  Total theorems needing proof: {len(dep_graph)}")
    print(f"  Topological sort successful: {len(cycles) == 0}")
    if cycles:
        print(f"  ⚠ Circular dependencies detected: {cycles}")

    # Statistics
    total_deps = sum(len(item['dependencies']) for item in dep_graph)
    in_pipeline = sum(1 for item in dep_graph for dep in item['dependencies'] if dep['status'] == 'in_current_pipeline')
    satisfied_ext = sum(1 for item in dep_graph for dep in item['dependencies'] if dep['status'] == 'satisfied_externally')
    truly_missing = sum(1 for item in dep_graph for dep in item['dependencies'] if dep['status'] == 'truly_missing')

    print(f"\nDependency Statistics:")
    print(f"  Total dependencies: {total_deps}")
    print(f"  In current pipeline: {in_pipeline}")
    print(f"  Satisfied externally: {satisfied_ext}")
    print(f"  Truly missing: {truly_missing}")

    if truly_missing > 0:
        print(f"\n⚠ Truly missing dependencies found:")
        for item in dep_graph:
            if item['truly_missing_deps']:
                print(f"  {item['label']}: {item['truly_missing_deps']}")

if __name__ == '__main__':
    main()
