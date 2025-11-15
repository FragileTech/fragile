#!/usr/bin/env python3
"""
Apply backward cross-references to 01_fragile_gas_framework.md

This script modifies the document by adding {prf:ref} directives at the first
mention of key concepts within each entity.

Strategy:
1. For each entity that needs enrichment
2. Find first occurrence of target concept keyword
3. Add inline {prf:ref} at that location
4. Preserve all formatting and mathematical content
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

@dataclass
class Entity:
    label: str
    entity_type: str
    name: str
    line_start: int  # 1-indexed
    line_end: int    # 1-indexed
    content: List[str]

    @property
    def content_text(self) -> str:
        return ''.join(self.content)

# Same extraction logic as before
def extract_entities(file_path: Path) -> List[Entity]:
    """Extract all labeled mathematical entities"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    entities = []
    i = 0

    while i < len(lines):
        line = lines[i]

        match = re.match(r':::\{prf:(definition|theorem|lemma|axiom|corollary|proposition|remark|assumption|proof)\}\s*(.*)', line)
        if match:
            entity_type = match.group(1)
            name = match.group(2).strip()
            line_start = i + 1

            content = []
            label = None
            i += 1

            while i < len(lines) and lines[i].strip() != ':::':
                content.append(lines[i])

                label_match = re.match(r':label:\s*(.+)', lines[i].strip())
                if label_match:
                    label = label_match.group(1).strip()

                i += 1

            line_end = i + 1

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

# Define reference additions (from our analysis)
# Format: {source_label: [(target_label, keyword_pattern, reason)]}
REFERENCES_TO_ADD = {
    # Core walker references
    'axiom-bounded-relative-collapse': [('def-walker', r'\bwalker', 'walker concept')],
    'def-cloning-measure': [('def-walker', r'\bwalker', 'walker concept')],
    'rem-projection-choice': [('def-walker', r'\bwalker', 'walker concept')],

    # Swarm and state space references (most common)
    'def-metric-quotient': [('def-swarm-and-state-space', r'\\Sigma_N', 'swarm state space notation')],
    'proof-lem-borel-image-of-the-projected-swarm-space': [('def-swarm-and-state-space', r'swarm', 'swarm concept')],
    'rem-margin-stability': [('def-swarm-and-state-space', r'swarm', 'swarm concept')],

    # Alive/dead set references
    'thm-mean-square-standardization-error': [('def-alive-dead-sets', r'\\mathcal\{A\}', 'alive set notation')],
    'proof-lem-empirical-aggregator-properties': [('def-alive-dead-sets', r'alive set', 'alive set concept')],

    # Algorithmic space references
    'def-algorithmic-cemetery-extension': [('def-algorithmic-space-generic', r'algorithmic space', 'algorithmic space concept')],
    'def-cemetery-state-measure': [('def-algorithmic-space-generic', r'\\mathcal\{Y\}', 'algorithmic space notation')],

    # Valid noise measure references
    'proof-lem-validation-of-the-heat-kernel': [('def-valid-noise-measure', r'noise measure', 'noise measure concept')],
    'lem-validation-of-the-uniform-ball-measure': [('def-valid-noise-measure', r'valid noise', 'valid noise concept')],

    # Guaranteed revival
    'def-stochastic-threshold-cloning': [('axiom-guaranteed-revival', r'revival mechanism', 'revival mechanism concept')],

    # Valid state space
    'def-valid-noise-measure': [('def-valid-state-space', r'valid state space', 'valid state space concept')],
}

def find_first_match(text: str, pattern: str) -> Tuple[int, int]:
    """
    Find first occurrence of pattern in text.

    Returns: (start_pos, end_pos) or (-1, -1) if not found
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return -1, -1

def add_reference_inline(content_lines: List[str], pattern: str, ref_label: str) -> Tuple[List[str], bool]:
    """
    Add {prf:ref} inline at first occurrence of pattern.

    Returns: (modified_lines, was_modified)
    """
    content_text = ''.join(content_lines)

    # Check if already has reference
    if f"{{prf:ref}}`{ref_label}`" in content_text:
        return content_lines, False

    # Find first match
    start_pos, end_pos = find_first_match(content_text, pattern)
    if start_pos == -1:
        return content_lines, False

    # Insert reference after the matched text
    # Strategy: Find the end of the current word/symbol and add parenthetical reference
    # For simplicity, we'll add it right after the match in parentheses

    # Convert position to line/column
    char_count = 0
    target_line_idx = -1
    target_col = -1

    for idx, line in enumerate(content_lines):
        if char_count + len(line) > end_pos:
            target_line_idx = idx
            target_col = end_pos - char_count
            break
        char_count += len(line)

    if target_line_idx == -1:
        return content_lines, False

    # Insert reference
    line = content_lines[target_line_idx]

    # Add reference in parenthetical form
    ref_text = f" ({{prf:ref}}`{ref_label}`)"

    # Check if we're inside a word or at a boundary
    if target_col < len(line):
        # Insert at position
        new_line = line[:target_col] + ref_text + line[target_col:]
    else:
        new_line = line.rstrip() + ref_text + "\n"

    modified_lines = content_lines.copy()
    modified_lines[target_line_idx] = new_line

    return modified_lines, True

def apply_enrichment(file_path: Path, output_path: Path) -> Dict:
    """
    Apply backward reference enrichment to document.

    Returns: Statistics about the enrichment
    """
    print(f"Reading: {file_path}")
    entities = extract_entities(file_path)
    print(f"Found {len(entities)} entities")

    # Read original lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    modified_lines = lines.copy()
    entity_map = {e.label: e for e in entities}

    stats = {
        'entities_processed': 0,
        'references_added': 0,
        'references_skipped': 0,
        'by_target': {}
    }

    # Apply references
    for source_label, ref_specs in REFERENCES_TO_ADD.items():
        if source_label not in entity_map:
            print(f"Warning: Source entity not found: {source_label}")
            continue

        entity = entity_map[source_label]
        stats['entities_processed'] += 1

        # Get entity content from modified_lines
        entity_start_idx = entity.line_start  # 1-indexed → need line_start-1 for 0-indexed, but +1 for header
        entity_end_idx = entity.line_end - 1  # 1-indexed → -1 for exclusive end

        # Find entity boundaries in modified_lines
        # The entity starts at line_start (1-indexed), which is lines[line_start-1] in 0-indexed
        # But we need to account for the directive line itself
        # Entity content is between the directive and the closing :::

        # Scan to find directive
        directive_idx = -1
        for idx in range(len(modified_lines)):
            if f":label: {source_label}" in modified_lines[idx]:
                # Found label, directive is a few lines before
                for back_idx in range(max(0, idx-5), idx):
                    if modified_lines[back_idx].startswith(":::{prf:"):
                        directive_idx = back_idx
                        break
                break

        if directive_idx == -1:
            print(f"Warning: Could not find directive for {source_label}")
            continue

        # Find closing :::
        closing_idx = -1
        for idx in range(directive_idx + 1, len(modified_lines)):
            if modified_lines[idx].strip() == ':::':
                closing_idx = idx
                break

        if closing_idx == -1:
            print(f"Warning: Could not find closing ::: for {source_label}")
            continue

        # Extract entity content (between directive and closing)
        content_start = directive_idx + 1
        content_end = closing_idx
        entity_content = modified_lines[content_start:content_end]

        # Apply each reference
        for target_label, pattern, reason in ref_specs:
            print(f"Adding ref in {source_label} → {target_label} (pattern: {pattern})")

            modified_content, was_modified = add_reference_inline(entity_content, pattern, target_label)

            if was_modified:
                # Update modified_lines
                modified_lines[content_start:content_end] = modified_content
                stats['references_added'] += 1

                if target_label not in stats['by_target']:
                    stats['by_target'][target_label] = 0
                stats['by_target'][target_label] += 1

                print(f"  ✓ Added")
            else:
                stats['references_skipped'] += 1
                print(f"  ✗ Skipped (already exists or pattern not found)")

    # Write output
    print(f"\nWriting enriched document to: {output_path}")
    with open(output_path, 'w') as f:
        f.writelines(modified_lines)

    return stats

def main():
    input_file = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md")
    output_file = input_file  # Modify in-place (backup already created)

    stats = apply_enrichment(input_file, output_file)

    print("\n" + "="*80)
    print("BACKWARD REFERENCE ENRICHMENT COMPLETE")
    print("="*80)
    print(f"Entities processed: {stats['entities_processed']}")
    print(f"References added: {stats['references_added']}")
    print(f"References skipped: {stats['references_skipped']}")

    print("\nReferences added by target:")
    for target, count in sorted(stats['by_target'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {target}: {count}")

    print(f"\nBackup saved at: {input_file}.backup_cross_ref")
    print(f"Enriched document written to: {output_file}")

if __name__ == "__main__":
    main()
