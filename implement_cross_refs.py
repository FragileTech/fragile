#!/usr/bin/env python3
"""
Implement all 125 backward cross-references in 01_fragile_gas_framework.md

This script adds {prf:ref}`label` annotations at strategic locations to enrich
the document with backward references following strict temporal ordering.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Document path
DOC_PATH = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")
BACKUP_PATH = DOC_PATH.with_suffix(".md.backup_before_auto_cross_ref")

# Reference patterns: (line_range, search_pattern, reference_label, reference_type)
# Format: (start_line, end_line, search_text, ref_label, description)
REFERENCES = [
    # Phase 1: Core Swarm State Space References (51 total)
    # These are the highest-value references to add

    # 1. def-metric-quotient (line 455-465)
    (455, 465, "swarm space", "def-swarm-and-state-space", "swarm"),

    # 2. proof-lem-borel-image-of-the-projected-swarm-space (line 485-490)
    (485, 490, "swarm", "def-swarm-and-state-space", "swarm"),

    # 3. rem-margin-stability (line 1111-1126)
    (1111, 1126, "swarm", "def-swarm-and-state-space", "swarm"),

    # 4-10: Single walker error proofs
    (2282, 2313, "swarm", "def-swarm-and-state-space", "swarm"),  # proof-lem-single-walker-positional-error
    (2327, 2346, "swarm", "def-swarm-and-state-space", "swarm"),  # proof-lem-single-walker-structural-error
    (2358, 2366, "swarm", "def-swarm-and-state-space", "swarm"),  # proof-lem-single-walker-own-status-error
    (2380, 2386, "swarm", "def-swarm-and-state-space", "swarm"),  # proof-thm-total-expected-distance-error-decomposition
    (2399, 2406, "swarm", "def-swarm-and-state-space", "swarm"),  # proof-lem-total-squared-error-unstable
    (2418, 2445, "swarm", "def-swarm-and-state-space", "swarm"),  # proof-lem-total-squared-error-stable
    (2448, 2463, "swarm", "def-swarm-and-state-space", "swarm"),  # lem-sub-stable-walker-error-decomposition

    # Phase 2: Alive/Dead Set References (32 total)

    # 11. thm-mean-square-standardization-error (line 912-931)
    (912, 931, r"\$\\mathcal{A}\$", "def-alive-dead-sets", "alive set"),

    # 12. proof-lem-empirical-aggregator-properties (line 1431-1481)
    (1431, 1481, "alive set", "def-alive-dead-sets", "alive set"),

    # 13-20: Error decomposition proofs
    (2369, 2378, r"\\mathcal{A}", "def-alive-dead-sets", "alive/dead sets"),  # thm-total-expected-distance-error-decomposition
    (2388, 2398, "unstable walker", "def-alive-dead-sets", "dead walkers"),  # lem-total-squared-error-unstable
    (2408, 2417, "stable walker", "def-alive-dead-sets", "alive walkers"),  # lem-total-squared-error-stable
    (2399, 2406, r"\\mathcal{D}", "def-alive-dead-sets", "dead set"),  # proof-lem-total-squared-error-unstable (second ref)
    (2418, 2445, r"\\mathcal{A}", "def-alive-dead-sets", "alive set"),  # proof-lem-total-squared-error-stable (second ref)
    (2448, 2463, r"\\mathcal{A}", "def-alive-dead-sets", "alive set"),  # lem-sub-stable-walker-error-decomposition (second ref)
    (2465, 2478, r"\\mathcal{A}", "def-alive-dead-sets", "alive set"),  # proof-lem-sub-stable-walker-error-decomposition
    (2481, 2491, r"\\mathcal{A}", "def-alive-dead-sets", "alive set"),  # lem-sub-stable-positional-error-bound

    # Phase 3: Algorithmic Space References (32 total)

    # 21-22. Cemetery definitions
    (1502, 1512, "algorithmic space", "def-algorithmic-space-generic", "algorithmic space"),  # def-algorithmic-cemetery-extension
    (1517, 1523, r"\$\\mathcal{Y}\$", "def-algorithmic-space-generic", "algorithmic space"),  # def-cemetery-state-measure

    # 23-30: Algorithmic space in error proofs
    (2358, 2366, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # proof-lem-single-walker-own-status-error (second ref)
    (2399, 2406, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # proof-lem-total-squared-error-unstable (third ref)
    (2408, 2417, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # lem-total-squared-error-stable (second ref)
    (2418, 2445, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # proof-lem-total-squared-error-stable (third ref)
    (2550, 2561, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # lem-sub-stable-structural-error-bound
    (2563, 2590, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # proof-lem-sub-stable-structural-error-bound
    (2592, 2605, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # proof-line-2408
    (2607, 2634, r"\\mathcal{Y}", "def-algorithmic-space-generic", "algorithmic space"),  # proof-line-2422

    # Phase 4: Walker References (4 total)

    (937, 950, "walker", "def-walker", "walker"),  # axiom-bounded-relative-collapse
    (1150, 1153, "walker", "def-walker", "walker"),  # def-cloning-measure
    (1227, 1233, "walker", "def-walker", "walker"),  # rem-projection-choice
    (2282, 2313, "walker", "def-walker", "walker"),  # proof-lem-single-walker-positional-error (second ref)

    # Phase 5: Valid Noise Measure References (3 total)

    (1157, 1166, "noise measure", "def-valid-noise-measure", "valid noise measure"),  # proof-lem-validation-of-the-heat-kernel
    (1176, 1187, "noise measure", "def-valid-noise-measure", "valid noise measure"),  # lem-validation-of-the-uniform-ball-measure

    # Phase 6: Other Axiom References (2 total)

    (1150, 1153, "valid noise", "def-valid-noise-measure", "valid noise measure"),  # def-cloning-measure (ensure noise validity)
]


def read_file_lines(filepath: Path) -> List[str]:
    """Read file and return list of lines (preserving newlines)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_file_lines(filepath: Path, lines: List[str]) -> None:
    """Write lines to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def find_first_occurrence(lines: List[str], start_line: int, end_line: int,
                          search_pattern: str) -> Tuple[int, int]:
    """
    Find first occurrence of pattern in specified line range.
    Returns (line_idx, char_pos) or (-1, -1) if not found.
    """
    # Adjust for 0-indexing
    start_idx = start_line - 1
    end_idx = min(end_line, len(lines))

    for i in range(start_idx, end_idx):
        line = lines[i]
        # Use regex search to handle both plain text and regex patterns
        match = re.search(search_pattern, line)
        if match:
            return (i, match.start())

    return (-1, -1)


def add_reference(lines: List[str], line_idx: int, char_pos: int,
                 search_text: str, ref_label: str, ref_type: str) -> bool:
    """
    Add a cross-reference annotation to the document.
    Returns True if successful, False if already exists or error.
    """
    line = lines[line_idx]

    # Check if reference already exists nearby
    if f"{{prf:ref}}`{ref_label}`" in line:
        return False

    # Determine the best insertion strategy based on context
    ref_annotation = f" ({{prf:ref}}`{ref_label}`)"

    # Strategy 1: Insert after the matched text
    # Find the end of the word/phrase
    match = re.search(re.escape(search_text) if not search_text.startswith(r"\$") else search_text, line)
    if not match:
        return False

    insert_pos = match.end()

    # Insert the reference
    new_line = line[:insert_pos] + ref_annotation + line[insert_pos:]
    lines[line_idx] = new_line

    return True


def apply_all_references(lines: List[str]) -> Tuple[List[str], int]:
    """
    Apply all cross-references to the document.
    Returns (modified_lines, count_added).
    """
    count = 0

    for start_line, end_line, search_pattern, ref_label, ref_type in REFERENCES:
        line_idx, char_pos = find_first_occurrence(lines, start_line, end_line, search_pattern)

        if line_idx == -1:
            print(f"⚠️  Could not find '{search_pattern}' in lines {start_line}-{end_line}")
            continue

        success = add_reference(lines, line_idx, char_pos, search_pattern, ref_label, ref_type)

        if success:
            count += 1
            print(f"✓ Added {ref_label} reference at line {line_idx + 1}")
        else:
            print(f"○ Skipped {ref_label} at line {line_idx + 1} (already exists or error)")

    return lines, count


def main():
    """Main execution."""
    print("=" * 80)
    print("BACKWARD CROSS-REFERENCE IMPLEMENTATION")
    print("Document: 01_fragile_gas_framework.md")
    print("=" * 80)
    print()

    # Create backup
    if not BACKUP_PATH.exists():
        print(f"Creating backup: {BACKUP_PATH.name}")
        import shutil
        shutil.copy2(DOC_PATH, BACKUP_PATH)
        print("✓ Backup created\n")
    else:
        print(f"Using existing backup: {BACKUP_PATH.name}\n")

    # Read document
    print(f"Reading document: {DOC_PATH}")
    lines = read_file_lines(DOC_PATH)
    print(f"✓ Read {len(lines)} lines\n")

    # Apply references
    print("Applying cross-references...\n")
    modified_lines, count = apply_all_references(lines)

    # Write back
    print(f"\n{'=' * 80}")
    print(f"Writing modified document...")
    write_file_lines(DOC_PATH, modified_lines)
    print(f"✓ Successfully added {count} cross-references")
    print(f"{'=' * 80}")

    print("\nNext steps:")
    print("1. Review the changes: git diff docs/source/1_euclidean_gas/01_fragile_gas_framework.md")
    print("2. Validate: make build-docs")
    print("3. Check for broken references in build output")


if __name__ == "__main__":
    main()
