#!/usr/bin/env python3
"""
Verify all object entities in the Fragile Gas Framework.

This script validates all JSON files against the MathematicalObject Pydantic schema.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fragile.proofs.core.math_types import MathematicalObject


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def verify_objects(objects_dir: Path) -> None:
    """Verify all object files."""
    json_files = sorted(objects_dir.glob("*.json"))

    print(f"\n{'=' * 80}")
    print("OBJECT VERIFICATION REPORT")
    print(f"{'=' * 80}")
    print(f"Directory: {objects_dir}")
    print(f"Total files: {len(json_files)}")
    print(f"{'=' * 80}\n")

    valid_count = 0
    invalid_count = 0

    for file_path in json_files:
        try:
            data = load_json_file(file_path)
            obj = MathematicalObject(**data)
            valid_count += 1
            print(f"✓ {file_path.name:60} | {obj.object_type.value:12} | {obj.label}")
        except ValidationError as e:
            invalid_count += 1
            print(f"✗ {file_path.name:60} | INVALID")
            for error in e.errors()[:3]:
                field = ".".join(str(loc) for loc in error["loc"])
                print(f"    {field}: {error['msg']}")
        except Exception as e:
            invalid_count += 1
            print(f"✗ {file_path.name:60} | ERROR: {e}")

    print(f"\n{'=' * 80}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files: {len(json_files)}")
    print(f"Valid objects: {valid_count}")
    print(f"Invalid objects: {invalid_count}")
    print(f"Success rate: {100 * valid_count / len(json_files):.1f}%")
    print(f"{'=' * 80}\n")


def main():
    """Main entry point."""
    objects_dir = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/objects"
    )

    if not objects_dir.exists():
        print(f"ERROR: Directory not found: {objects_dir}")
        return

    verify_objects(objects_dir)


if __name__ == "__main__":
    main()
