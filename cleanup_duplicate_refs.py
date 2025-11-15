#!/usr/bin/env python3
"""
Clean up duplicate {prf:ref} directives in 02_euclidean_gas.md
"""

import re
from pathlib import Path
from typing import Tuple

def cleanup_duplicates(doc_path: Path) -> Tuple[str, int]:
    """Remove duplicate consecutive {prf:ref} references."""
    content = doc_path.read_text()

    # Pattern to match duplicate consecutive references
    duplicates_removed = 0

    # Remove exact duplicates like ({prf:ref}`xxx`)({prf:ref}`xxx`)
    while True:
        new_content = re.sub(
            r'(\{prf:ref\}`([a-z0-9\-_]+)`\)\s*\(\{prf:ref\}`\2`\)',
            r'(\{prf:ref\}`\2`)',
            content
        )
        if new_content == content:
            break
        content = new_content
        duplicates_removed += 1

    return content, duplicates_removed

def main():
    """Main execution."""
    doc_path = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/02_euclidean_gas.md")

    print("Cleaning up duplicate references...")
    cleaned_content, duplicates = cleanup_duplicates(doc_path)

    if duplicates > 0:
        doc_path.write_text(cleaned_content)
        print(f"✓ Removed {duplicates} duplicate references")
    else:
        print("✓ No duplicates found")

    # Count final references
    final_count = cleaned_content.count('{prf:ref}')
    print(f"\nFinal reference count: {final_count}")

if __name__ == '__main__':
    main()
