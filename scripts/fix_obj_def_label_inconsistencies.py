#!/usr/bin/env python3
"""
Fix label inconsistencies: Replace def- with obj- for 3 objects.

These objects have both obj- and def- versions in refined_data, but theorems
reference the def- version. This script updates theorems to use obj- consistently.
"""

import json
from pathlib import Path


# Mapping: def- label → obj- label
LABEL_REPLACEMENTS = {
    "def-expected-cloning-action": "obj-expected-cloning-action",
    "def-status-update-operator": "obj-status-update-operator",
    "def-total-expected-cloning-action": "obj-total-expected-cloning-action",
}

# Files and line numbers from grep results
FILES_TO_FIX = {
    "thm-expected-cloning-action-continuity.json": ["def-expected-cloning-action"],
    "thm-total-expected-cloning-action-continuity.json": [
        "def-expected-cloning-action",
        "def-total-expected-cloning-action",
    ],
    "cor-pipeline-continuity-margin-stability.json": ["def-status-update-operator"],
    "prop-psi-markov-kernel.json": ["def-status-update-operator"],
    "thm-pipeline-continuity-margin-stability.json": ["def-status-update-operator"],
    "thm-post-perturbation-status-update-continuity.json": ["def-status-update-operator"],
}


def fix_labels_in_list(data_list, replacements):
    """Replace def- labels with obj- labels in a list."""
    if not isinstance(data_list, list):
        return data_list

    return [replacements.get(item, item) if isinstance(item, str) else item for item in data_list]


def fix_labels_recursive(data, replacements):
    """Recursively fix labels in nested data structures."""
    if isinstance(data, dict):
        return {key: fix_labels_recursive(value, replacements) for key, value in data.items()}
    if isinstance(data, list):
        return [fix_labels_recursive(item, replacements) for item in data]
    if isinstance(data, str):
        return replacements.get(data, data)
    return data


def main():
    theorems_dir = Path(
        "docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/theorems"
    )

    if not theorems_dir.exists():
        print(f"Error: Theorems directory not found: {theorems_dir}")
        return

    print("=" * 70)
    print("Fixing obj-/def- Label Inconsistencies")
    print("=" * 70)
    print()

    total_files = 0
    total_replacements = 0

    for filename, expected_labels in FILES_TO_FIX.items():
        file_path = theorems_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  {filename}: File not found")
            continue

        # Load JSON
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Count replacements before
        json_str_before = json.dumps(data)
        replacements_in_file = sum(1 for label in expected_labels if label in json_str_before)

        if replacements_in_file == 0:
            print(f"  ⚠️  {filename}: No def- labels found (already fixed?)")
            continue

        # Fix labels recursively
        data_fixed = fix_labels_recursive(data, LABEL_REPLACEMENTS)

        # Verify replacements
        json_str_after = json.dumps(data_fixed)
        actual_replacements = replacements_in_file - sum(
            1 for label in expected_labels if label in json_str_after
        )

        if actual_replacements != replacements_in_file:
            print(
                f"  ⚠️  {filename}: Expected {replacements_in_file} replacements, "
                f"got {actual_replacements}"
            )
            continue

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_fixed, f, indent=2, ensure_ascii=False)

        print(f"  ✅ {filename}: Fixed {actual_replacements} label(s)")
        for label in expected_labels:
            if label in json_str_before:
                print(f"      {label} → {LABEL_REPLACEMENTS[label]}")

        total_files += 1
        total_replacements += actual_replacements

    print()
    print("=" * 70)
    print(f"Summary: {total_files} files fixed, {total_replacements} labels replaced")
    print("=" * 70)
    print()

    # Show what was changed
    print("Label mapping applied:")
    for old_label, new_label in LABEL_REPLACEMENTS.items():
        count = sum(1 for labels in FILES_TO_FIX.values() if old_label in labels)
        print(f"  {old_label} → {new_label} ({count} references)")


if __name__ == "__main__":
    main()
