#!/usr/bin/env python3
"""
Fix awkward reference placements where {prf:ref} appears right after "Let".
Move references to after variable definitions for natural reading flow.
"""

import re
from pathlib import Path

DOC_PATH = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

def read_file(filepath: Path) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath: Path, content: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def fix_let_patterns(content: str) -> tuple[str, int]:
    """Fix patterns where references appear immediately after 'Let'."""
    count = 0

    # Pattern 1: Let ({ref}) $var = ... (most common)
    # Move ref to after the definition
    pattern1 = r'Let \({prf:ref}`([^`]+)`\) \$([^$]+)\$'
    def repl1(m):
        nonlocal count
        count += 1
        ref_label = m.group(1)
        math_content = m.group(2)
        return f'Let ${math_content}$ ({{prf:ref}}`{ref_label}`)'

    content = re.sub(pattern1, repl1, content)

    # Pattern 2: Let ({ref1}) ({ref2}) ... (multiple refs at start)
    # Move all refs to end of sentence
    pattern2 = r'Let (\({prf:ref}`[^`]+`\)\s*)+([^.]+)\.'
    def repl2(m):
        nonlocal count
        refs = re.findall(r'\({prf:ref}`[^`]+`\)', m.group(0))
        main_content = m.group(2)
        if refs and main_content:
            count += 1
            refs_str = ' '.join(refs)
            return f'Let {main_content} {refs_str}.'
        return m.group(0)

    content = re.sub(pattern2, repl2, content)

    return content, count

def main():
    print("=" * 80)
    print("FIXING AWKWARD REFERENCE PLACEMENTS")
    print("=" * 80)
    print()

    # Read document
    print(f"Reading: {DOC_PATH}")
    content = read_file(DOC_PATH)

    # Find awkward patterns
    awkward_count = len(re.findall(r'^Let \({prf:ref}', content, re.MULTILINE))
    print(f"Found {awkward_count} awkward 'Let ({{prf:ref}}...)' patterns\n")

    # Fix patterns
    print("Applying fixes...")
    fixed_content, changes = fix_let_patterns(content)

    # Write back
    write_file(DOC_PATH, fixed_content)

    print(f"✓ Fixed {changes} awkward reference placements")
    print("=" * 80)

    # Verify
    remaining = len(re.findall(r'^Let \({prf:ref}', fixed_content, re.MULTILINE))
    print(f"\nRemaining awkward patterns: {remaining}")

    if remaining > 0:
        print("\nNote: Some complex patterns may require manual review")

if __name__ == "__main__":
    main()
