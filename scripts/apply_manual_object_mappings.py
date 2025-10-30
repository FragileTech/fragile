#!/usr/bin/env python3
"""Apply manual source location mappings to remaining objects."""

import json
from pathlib import Path


def apply_manual_mappings():
    """Apply manual mappings from manual_object_mappings.json."""
    repo_root = Path(__file__).parent.parent
    mappings_file = repo_root / "scripts/manual_object_mappings.json"
    objects_dir = (
        repo_root / "docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/objects"
    )

    # Load mappings
    with open(mappings_file, encoding="utf-8") as f:
        mappings = json.load(f)

    # Remove comment fields
    mappings = {k: v for k, v in mappings.items() if not k.startswith("_")}

    print("Applying Manual Object Mappings")
    print("=" * 70)
    print(f"Mappings file: {mappings_file}")
    print(f"Objects directory: {objects_dir}")
    print(f"Total mappings: {len(mappings)}\n")

    success = 0
    failed = 0

    for label, mapping in mappings.items():
        json_file = objects_dir / f"{label}.json"

        if not json_file.exists():
            print(f"✗ {label:50} [not_found] File does not exist")
            failed += 1
            continue

        try:
            # Read object data
            with open(json_file, encoding="utf-8") as f:
                obj_data = json.load(f)

            # Update source_location
            source_location = obj_data.get("source_location", {})
            source_location["section"] = mapping["section"]
            source_location["line_range"] = mapping["line_range"]

            obj_data["source_location"] = source_location

            # Write updated data
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(obj_data, f, indent=2, ensure_ascii=False)
                f.write("\n")

            section = mapping["section"]
            lr = mapping["line_range"]
            print(f"✓ {label:50} §{section} L{lr[0]}-{lr[1]}")
            success += 1

        except Exception as e:
            print(f"✗ {label:50} [error] {e}")
            failed += 1

    print("=" * 70)
    print(f"\nResults: {success} succeeded, {failed} failed")

    return success, failed


if __name__ == "__main__":
    import sys

    success, failed = apply_manual_mappings()

    if failed > 0:
        sys.exit(1)

    print("\n✓ All manual mappings applied successfully!")
    sys.exit(0)
