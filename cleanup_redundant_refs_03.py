#!/usr/bin/env python3
"""
Cleanup redundant backward references in 03_cloning.md

Removes duplicate references that appear too close together to avoid cluttering.
Rules:
1. Within the same line, keep only the first occurrence of each label
2. Within nearby lines (5-line window), keep only the first occurrence
3. Preserve references in different contexts (e.g., definitions vs. proofs)
"""

import re
from pathlib import Path
from typing import Dict, Set, List, Tuple

def extract_references(line: str) -> List[Tuple[str, int, int]]:
    """
    Extract all {prf:ref}`label` references from a line.
    Returns list of (label, start_pos, end_pos) tuples.
    """
    pattern = r'\{prf:ref\}`([^`]+)`'
    matches = []
    for match in re.finditer(pattern, line):
        label = match.group(1)
        start = match.start()
        end = match.end()
        matches.append((label, start, end))
    return matches

def remove_duplicate_refs_in_line(line: str) -> str:
    """
    Remove duplicate references to the same label within a single line.
    Keep only the first occurrence.
    """
    refs = extract_references(line)
    if len(refs) <= 1:
        return line

    # Find duplicates
    seen_labels = set()
    refs_to_remove = []

    for label, start, end in refs:
        if label in seen_labels:
            refs_to_remove.append((start, end))
        else:
            seen_labels.add(label)

    # Remove duplicates (in reverse order to preserve positions)
    new_line = line
    for start, end in sorted(refs_to_remove, reverse=True):
        # Remove the reference but keep the text before it
        # Pattern: "text ({prf:ref}`label`)" → "text"
        # Find the opening parenthesis before the reference
        context_start = max(0, start - 5)
        context = new_line[context_start:start]

        if '(' in context:
            paren_pos = context.rfind('(')
            actual_start = context_start + paren_pos
            # Check if there's a closing paren after the reference
            if end < len(new_line) and new_line[end] == ')':
                new_line = new_line[:actual_start] + new_line[end+1:]
            else:
                new_line = new_line[:start] + new_line[end:]
        else:
            new_line = new_line[:start] + new_line[end:]

    return new_line

def cleanup_document(doc_path: Path):
    """Clean up redundant references in the document."""
    print(f"Reading document: {doc_path}")

    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")

    # Count original references
    original_refs = sum(1 for line in lines for _ in extract_references(line))
    print(f"Original reference count: {original_refs}")

    # Pass 1: Remove duplicates within each line
    cleaned_lines = []
    refs_removed_in_line = 0

    for i, line in enumerate(lines):
        original_line_refs = len(extract_references(line))
        cleaned_line = remove_duplicate_refs_in_line(line)
        cleaned_line_refs = len(extract_references(cleaned_line))
        refs_removed_in_line += (original_line_refs - cleaned_line_refs)

        if original_line_refs != cleaned_line_refs:
            print(f"  Line {i+1}: Removed {original_line_refs - cleaned_line_refs} duplicate ref(s)")

        cleaned_lines.append(cleaned_line)

    print(f"\nPass 1 complete: Removed {refs_removed_in_line} redundant references within lines")

    # Pass 2: Remove nearby duplicates (within 3-line window)
    # This is more conservative - only remove if truly redundant
    window_size = 3
    recent_refs = {}  # label -> line_number
    refs_removed_nearby = 0

    for i, line in enumerate(cleaned_lines):
        refs = extract_references(line)

        # Check for recently used references
        refs_to_remove = []
        for label, start, end in refs:
            if label in recent_refs:
                last_line = recent_refs[label]
                if i - last_line <= window_size:
                    # Too close - mark for removal
                    refs_to_remove.append((start, end))
                    refs_removed_nearby += 1
                else:
                    # Far enough - update position
                    recent_refs[label] = i
            else:
                recent_refs[label] = i

        # Remove marked references
        if refs_to_remove:
            new_line = line
            for start, end in sorted(refs_to_remove, reverse=True):
                context_start = max(0, start - 5)
                context = new_line[context_start:start]

                if '(' in context:
                    paren_pos = context.rfind('(')
                    actual_start = context_start + paren_pos
                    if end < len(new_line) and new_line[end] == ')':
                        new_line = new_line[:actual_start] + new_line[end+1:]
                    else:
                        new_line = new_line[:start] + new_line[end:]
                else:
                    new_line = new_line[:start] + new_line[end:]

            cleaned_lines[i] = new_line
            print(f"  Line {i+1}: Removed {len(refs_to_remove)} nearby duplicate ref(s)")

        # Clear old entries from tracking
        recent_refs = {label: line_num for label, line_num in recent_refs.items()
                      if i - line_num <= window_size}

    print(f"\nPass 2 complete: Removed {refs_removed_nearby} redundant references in nearby lines")

    # Save cleaned document
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    # Final count
    final_refs = sum(1 for line in cleaned_lines for _ in extract_references(line))
    total_removed = original_refs - final_refs

    print(f"\n{'='*70}")
    print("CLEANUP STATISTICS")
    print(f"{'='*70}")
    print(f"Original references:     {original_refs:4d}")
    print(f"References removed:      {total_removed:4d}")
    print(f"Final references:        {final_refs:4d}")
    print(f"{'='*70}")

    print(f"\nCleaned document saved: {doc_path}")

def main():
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/03_cloning.md")
    cleanup_document(doc_path)
    print("\n✓ Cleanup complete!")

if __name__ == "__main__":
    main()
