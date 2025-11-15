#!/usr/bin/env python3
"""
Fix nested or malformed references in 03_cloning.md
"""

import re
from pathlib import Path

def fix_nested_references(line: str) -> str:
    """
    Fix nested references like:
    {prf:ref}`def-single-swarm ({prf:ref}`def-swarm-and-state-space`)-space`

    Should be:
    {prf:ref}`def-single-swarm-space`
    """
    # Pattern for nested references
    nested_pattern = r'\{prf:ref\}`[^`]*\{prf:ref\}`[^`]+`[^`]*`'

    if re.search(nested_pattern, line):
        # This is a specific case: should reference def-single-swarm-space
        line = re.sub(
            r'\{prf:ref\}`def-single-swarm \(\{prf:ref\}`def-swarm-and-state-space`\)-space`',
            r'{prf:ref}`def-single-swarm-space`',
            line
        )

    return line

def fix_document(doc_path: Path):
    """Fix all nested references in the document."""
    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_lines = []
    fixes_count = 0

    for i, line in enumerate(lines):
        original = line
        fixed = fix_nested_references(line)

        if fixed != original:
            print(f"Line {i+1}: Fixed nested reference")
            fixes_count += 1

        fixed_lines.append(fixed)

    with open(doc_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

    print(f"\nFixed {fixes_count} nested references")
    return fixes_count

def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md")
    fixes_count = fix_document(doc_path)

    if fixes_count > 0:
        print(f"\n✓ Fixed {fixes_count} nested reference(s)")
    else:
        print("\n✓ No nested references found")

if __name__ == "__main__":
    main()
