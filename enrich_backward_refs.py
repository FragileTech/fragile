#!/usr/bin/env python3
"""
Add strategic backward cross-references to improve connectivity.
Focus on high-value connections to improve the document's reference graph.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

DOC_PATH = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

# Extract directive labels
def extract_labels_with_positions(content: str) -> Dict[str, Tuple[str, int, int]]:
    """Extract all labels with their type and line positions."""
    labels = {}
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Match directive opening
        directive_match = re.match(r'^::::\{prf:(definition|theorem|lemma|axiom|proposition|corollary|assumption|remark|proof)\}', line)
        if not directive_match:
            directive_match = re.match(r'^:::\{prf:(definition|theorem|lemma|axiom|proposition|corollary|assumption|remark|proof)\}', line)

        if directive_match:
            entity_type = directive_match.group(1)
            start_line = i

            # Look for label on next line
            if i + 1 < len(lines) and lines[i+1].startswith(':label:'):
                label_match = re.match(r':label:\s*([a-z0-9\-\_]+)', lines[i+1])
                if label_match:
                    label = label_match.group(1)

                    # Find end of directive (matching closing :::)
                    end_line = i + 2
                    colons_count = 4 if line.startswith('::::') else 3
                    closing = ':' * colons_count

                    while end_line < len(lines):
                        if lines[end_line].startswith(closing) and not lines[end_line].startswith(closing + '{'):
                            break
                        end_line += 1

                    labels[label] = (entity_type, start_line, end_line)

        i += 1

    return labels

def get_entity_content(lines: List[str], start: int, end: int) -> str:
    """Get content of entity between start and end lines."""
    return '\n'.join(lines[start:end+1])

def add_backward_refs_to_document(content: str) -> Tuple[str, List[str]]:
    """Add backward references where concepts are mentioned but not cited."""

    lines = content.split('\n')
    labels = extract_labels_with_positions(content)
    changes = []

    # Define strategic connections: label -> patterns to search for
    strategic_refs = {
        "def-walker": [
            (r'\bwalker\b(?!\s*\({prf:ref})', "walker"),
            (r'\bwalkers\b(?!\s*\({prf:ref})', "walkers"),
        ],
        "def-swarm-and-state-space": [
            (r'\bswarm\b(?!\s*\({prf:ref})', "swarm"),
            (r'\$\\Sigma_N\$(?!\s*\({prf:ref})', r"$\Sigma_N$"),
        ],
        "def-alive-dead-sets": [
            (r'\balive set\b(?!\s*\({prf:ref})', "alive set"),
            (r'\bdead set\b(?!\s*\({prf:ref})', "dead set"),
            (r'\$\\mathcal\{A\}\(', r"$\mathcal{A}("),
            (r'\$\\mathcal\{D\}\(', r"$\mathcal{D}("),
        ],
        "def-valid-state-space": [
            (r'\bValid State Space\b(?!\s*\({prf:ref})', "Valid State Space"),
            (r'\bvalid domain\b(?!\s*\({prf:ref})', "valid domain"),
        ],
        "def-perturbation-operator": [
            (r'\bperturbation operator\b(?!\s*\({prf:ref})', "perturbation operator"),
            (r'\$\\Psi_\{\\text\{pert\}\}\$(?!\s*\({prf:ref})', r"$\Psi_{\text{pert}}$"),
        ],
        "def-standardization-operator-n-dimensional": [
            (r'\bstandardization operator\b(?!\s*\({prf:ref})', "standardization operator"),
        ],
        "def-status-update-operator": [
            (r'\bstatus update operator\b(?!\s*\({prf:ref})', "status update operator"),
        ],
        "axiom-guaranteed-revival": [
            (r'\bAxiom of Guaranteed Revival\b(?!\s*\({prf:ref})', "Axiom of Guaranteed Revival"),
            (r'\bguaranteed revival\b(?!\s*\({prf:ref})', "guaranteed revival"),
        ],
        "axiom-boundary-regularity": [
            (r'\bAxiom of Boundary Regularity\b(?!\s*\({prf:ref})', "Axiom of Boundary Regularity"),
        ],
        "axiom-reward-regularity": [
            (r'\bAxiom of Reward Regularity\b(?!\s*\({prf:ref})', "Axiom of Reward Regularity"),
        ],
    }

    # Process each entity
    for current_label, (entity_type, start_line, end_line) in sorted(labels.items(), key=lambda x: x[1][1]):
        entity_lines = lines[start_line:end_line+1]
        modified = False

        # For each strategic reference to add
        for ref_label, patterns in strategic_refs.items():
            # Skip self-references
            if current_label == ref_label:
                continue

            # Only add backward references (ref must be defined before current entity)
            if ref_label in labels:
                ref_start = labels[ref_label][1]
                if ref_start >= start_line:
                    continue  # Would be forward ref
            else:
                continue  # Ref label doesn't exist

            # Check if already referenced
            entity_text = '\n'.join(entity_lines)
            if f'{{prf:ref}}`{ref_label}`' in entity_text:
                continue  # Already referenced

            # Look for first occurrence of any pattern
            for pattern_regex, display_text in patterns:
                for i, line in enumerate(entity_lines):
                    # Skip label lines and directive headers
                    if ':label:' in line or line.startswith(':::'):
                        continue

                    match = re.search(pattern_regex, line)
                    if match:
                        # Add reference after first occurrence
                        pos = match.end()

                        # Avoid inserting mid-word
                        if pos < len(line) and line[pos].isalnum():
                            continue

                        # Insert reference
                        new_line = line[:pos] + f' ({{prf:ref}}`{ref_label}`)' + line[pos:]
                        entity_lines[i] = new_line
                        changes.append(f"{current_label} ({entity_type}) now references {ref_label}")
                        modified = True
                        break

                if modified:
                    break  # Only add one reference per concept per entity

            if modified:
                break  # Move to next entity after one modification

        # Update lines if modified
        if modified:
            lines[start_line:end_line+1] = entity_lines

    return '\n'.join(lines), changes

def main():
    print("=" * 80)
    print("BACKWARD CROSS-REFERENCE ENRICHMENT")
    print("=" * 80)

    print(f"\nTarget: {DOC_PATH}")

    # Load document
    print("\n[1/4] Loading document...")
    content = DOC_PATH.read_text(encoding='utf-8')
    original_lines = len(content.split('\n'))
    print(f"      Loaded {original_lines} lines")

    # Extract labels
    print("\n[2/4] Analyzing entity structure...")
    labels = extract_labels_with_positions(content)
    print(f"      Found {len(labels)} labeled entities")

    # Add references
    print("\n[3/4] Adding backward cross-references...")
    enriched, changes = add_backward_refs_to_document(content)
    print(f"      Added {len(changes)} new references")

    # Show changes
    if changes:
        print("\n      Changes:")
        for i, change in enumerate(changes[:15], 1):
            print(f"      {i:2}. {change}")
        if len(changes) > 15:
            print(f"      ... and {len(changes) - 15} more")

    # Write output
    print("\n[4/4] Writing enriched document...")

    # Backup
    backup_path = DOC_PATH.with_suffix('.md.backup_ref_enrichment')
    backup_path.write_text(content, encoding='utf-8')
    print(f"      Backup: {backup_path}")

    # Write enriched
    DOC_PATH.write_text(enriched, encoding='utf-8')
    print(f"      Output: {DOC_PATH}")

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: uv run mathster connectivity 01_fragile_gas_framework")
    print("  2. Review connectivity improvements")

if __name__ == "__main__":
    main()
