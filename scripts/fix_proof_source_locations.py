#!/usr/bin/env python3
"""
Fix proof source locations to pass strict validation.

This script:
1. Normalizes section numbers (removes "Section " prefix, extracts just digits)
2. Ensures line_range is populated in source_location
3. Uses existing source_lines legacy field when available
4. Finds missing line ranges using text matching
"""

import json
from pathlib import Path
import re
from typing import Any

from fragile.proofs.tools.line_finder import (
    find_text_in_markdown,
    get_file_line_count,
    validate_line_range,
)


def normalize_section(section_str: str | None) -> str | None:
    """
    Normalize section to X.Y.Z format (only digits and dots).

    Examples:
        "Section 7" -> "7"
        "17" -> "17"
        "Section 17: The Revival State" -> "17"
        "14. The Perturbation Operator" -> "14"
        "§7" -> "7"
    """
    if not section_str:
        return None

    # Remove "Section " prefix
    section_str = re.sub(r"^Section\s+", "", section_str, flags=re.IGNORECASE)

    # Remove § symbol
    section_str = section_str.replace("§", "")

    # Extract just the number part (before colon, period followed by space, or end)
    match = re.match(r"^(\d+(?:\.\d+)*)", section_str.strip())
    if match:
        return match.group(1)

    return None


def parse_line_range(line_str: str) -> tuple[int, int] | None:
    """
    Parse line range from string like "1375-1425" or "4881-4881".

    Returns:
        Tuple of (start, end) or None if invalid
    """
    if not line_str:
        return None

    match = re.match(r"^(\d+)-(\d+)$", line_str.strip())
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return (start, end)

    return None


def extract_search_text(entity_data: dict[str, Any]) -> str | None:
    """Extract text for searching in markdown."""
    # For proofs, use the content or proof_text field
    text = entity_data.get("content") or entity_data.get("proof_text")
    if text:
        # Take first 200 chars for better matching
        # Skip markdown header like "**Proof.**" if present
        text = re.sub(r"^\*\*Proof\.\*\*\s*", "", text)
        return text[:200] if len(text) > 200 else text

    return None


def fix_proof_source_location(
    proof_path: Path,
    markdown_file: Path,
    markdown_content: str,
    max_lines: int,
) -> bool:
    """
    Fix source location for a single proof file.

    Returns:
        True if fixed successfully, False otherwise
    """
    try:
        # Read proof JSON
        with open(proof_path, encoding="utf-8") as f:
            proof_data = json.load(f)

        source_location = proof_data.get("source_location", {})

        # Fix section
        old_section = source_location.get("section")
        normalized_section = normalize_section(old_section)

        # Also check legacy fields: source_section, context, section
        if not normalized_section:
            for field in ["source_section", "context", "section"]:
                legacy_section = proof_data.get(field)
                if legacy_section:
                    normalized_section = normalize_section(legacy_section)
                    if normalized_section:
                        break

        # Fix line_range
        line_range = source_location.get("line_range")

        # Try to get line_range from legacy fields
        if not line_range:
            # Check if line_range exists as a root-level field (common pattern)
            root_line_range = proof_data.get("line_range")
            if root_line_range:
                if isinstance(root_line_range, list) and len(root_line_range) == 2:
                    line_range = tuple(root_line_range)

            # Try source_lines field
            if not line_range:
                legacy_lines = proof_data.get("source_lines")
                if legacy_lines:
                    line_range = parse_line_range(legacy_lines)

            # Try start_line/end_line fields
            if not line_range:
                start = proof_data.get("start_line")
                end = proof_data.get("end_line")
                if start and end:
                    line_range = (int(start), int(end))

            # Try single line_number field (make it a single-line range)
            if not line_range:
                line_num = proof_data.get("line_number")
                if line_num:
                    line_range = (int(line_num), int(line_num))

        # If still no line_range, try text matching
        if not line_range:
            search_text = extract_search_text(proof_data)
            if search_text:
                found_range = find_text_in_markdown(
                    markdown_content, search_text, case_sensitive=False
                )
                if found_range and validate_line_range(found_range, max_lines):
                    line_range = found_range
                    print(
                        f"  Found line range {line_range} via text matching for {proof_path.name}"
                    )

        # Update source_location
        changed = False
        if normalized_section and normalized_section != old_section:
            source_location["section"] = normalized_section
            changed = True
            print(
                f"  Fixed section: '{old_section}' -> '{normalized_section}' in {proof_path.name}"
            )

        if line_range and line_range != source_location.get("line_range"):
            source_location["line_range"] = list(line_range)  # JSON needs list, not tuple
            changed = True
            print(f"  Added line_range: {line_range} to {proof_path.name}")

        if changed:
            proof_data["source_location"] = source_location

            # Write back
            with open(proof_path, "w", encoding="utf-8") as f:
                json.dump(proof_data, f, indent=2, ensure_ascii=False)

            return True

        return False

    except Exception as e:
        print(f"  ERROR fixing {proof_path.name}: {e}")
        return False


def main():
    # Paths
    proofs_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/proofs")
    markdown_file = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

    if not proofs_dir.exists():
        print(f"ERROR: Proofs directory not found: {proofs_dir}")
        return

    if not markdown_file.exists():
        print(f"ERROR: Markdown file not found: {markdown_file}")
        return

    # Read markdown content once
    print(f"Reading markdown file: {markdown_file}")
    with open(markdown_file, encoding="utf-8") as f:
        markdown_content = f.read()
    max_lines = get_file_line_count(markdown_content)
    print(f"Markdown file has {max_lines} lines\n")

    # Process all proof JSON files
    proof_files = sorted(proofs_dir.glob("*.json"))
    print(f"Processing {len(proof_files)} proof files...\n")

    fixed_count = 0
    for proof_file in proof_files:
        if fix_proof_source_location(proof_file, markdown_file, markdown_content, max_lines):
            fixed_count += 1

    print(f"\n{'=' * 60}")
    print(f"Fixed {fixed_count} out of {len(proof_files)} proof files")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
