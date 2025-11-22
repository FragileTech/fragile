#!/usr/bin/env python3
"""
Fix remaining issues in the MyST-converted NS draft.
"""

import re
from pathlib import Path


def fix_multi_citations(text: str) -> str:
    """Fix multi-reference citations like [2, 3] → [@constantin1993; @moffatt1992]."""
    citation_map = {
        '1': 'beale1984',
        '2': 'constantin1993',
        '3': 'moffatt1992',
        '4': 'tao2016',
        '5': 'luo2014',
        '6': 'escauriaza2003',
        '7': 'benjamin1962',
        '8': 'caffarelli1982',
        '9': 'lin1998',
        '10': 'naber2017',
        '11': 'seregin2012',
        '12': 'bianchi1991',
        '13': 'dolbeault2024',
    }

    # Pattern: [N, M] or [N, M, O] etc.
    def replace_multi_cite(match):
        numbers = match.group(1).split(',')
        numbers = [n.strip() for n in numbers]
        cite_keys = [f"@{citation_map.get(n, f'ref{n}')}" for n in numbers if n in citation_map]
        if cite_keys:
            return f"[{'; '.join(cite_keys)}]"
        return match.group(0)

    text = re.sub(r'\[(\d+(?:,\s*\d+)+)\]', replace_multi_cite, text)

    return text


def remove_stray_closures(text: str) -> str:
    """Remove stray ::: that appear alone on lines inappropriately."""
    lines = text.split('\n')
    output = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # If this is a standalone :::, check context
        if line.strip() == ':::':
            # Look back to see if we're actually in a directive block
            # Check previous non-empty line
            prev_content_idx = i - 1
            while prev_content_idx >= 0 and not lines[prev_content_idx].strip():
                prev_content_idx -= 1

            if prev_content_idx >= 0:
                prev_line = lines[prev_content_idx]

                # If previous line is an equation block ending or a regular paragraph,
                # and we're not clearly in a directive, this ::: is likely spurious
                if (prev_line.strip().startswith('$$') or
                    (not prev_line.strip().startswith(':::') and
                     not prev_line.strip().startswith(':label:') and
                     len(prev_line.strip()) > 0 and
                     not prev_line.strip().startswith('#'))):

                    # Check if there's a directive opening nearby (within 20 lines back)
                    has_opening = False
                    for j in range(max(0, i - 20), i):
                        if re.match(r':::.*\{prf:', lines[j]):
                            has_opening = True
                            break

                    if not has_opening:
                        # This is likely a stray closure - skip it
                        i += 1
                        continue

        output.append(line)
        i += 1

    return '\n'.join(output)


def fix_theorem_statement(text: str) -> str:
    """Fix the main theorem statement that got split incorrectly."""
    # Find and fix the "Theorem (Structural Dichotomy" block
    pattern = r':::\{prf:theorem\} Structural Dichotomy for Navier-Stokes\n:label: the-structural-dichotomy-for-navier-stokes\n\n\nThe proof proceeds'

    replacement = ''':::{prf:theorem} Structural Dichotomy for Navier-Stokes
:label: thm-structural-dichotomy

Any renormalized blow-up candidate belongs to one of two branches. If it is variationally efficient, it converges (modulo symmetries) to a smooth, coherent profile that is excluded by spectral, geometric, or defocusing rigidity. If it is variationally inefficient, the efficiency deficit forces strictly positive growth of the Gevrey radius, excluding collapse. In either branch, the 3D Navier-Stokes solution with smooth initial data remains smooth for all time.
:::

The proof proceeds'''

    text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    return text


def main():
    input_file = Path("docs/source/navier_stokes/ns_draft_myst.md")

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Fixing multi-citations...")
    text = fix_multi_citations(text)

    print("Fixing main theorem statement...")
    text = fix_theorem_statement(text)

    print("Removing stray closures...")
    text = remove_stray_closures(text)

    print("Writing fixed version...")
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"✓ Fixes applied to {input_file}")


if __name__ == "__main__":
    main()
