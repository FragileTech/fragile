#!/usr/bin/env python3
"""
Aggressive Text Matching for Remaining Entities.

Uses fuzzy text matching and broader label searches for entities
that couldn't be found via directive matching.
"""

import argparse
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fragile.proofs.tools.line_finder import find_text_in_markdown


def extract_first_sentences(text: str, num_sentences: int = 2) -> str:
    """Extract first N sentences from text."""
    # Split on period followed by space or newline
    sentences = re.split(r"\.\s+|\.\n", text)
    return ". ".join(sentences[:num_sentences]).strip()[:200]  # Limit to 200 chars


def find_label_anywhere(markdown_content: str, label: str) -> tuple[int, int] | None:
    """
    Find label mentioned anywhere in markdown, return surrounding context.
    """
    lines = markdown_content.splitlines()

    # Search for label mention
    for i, line in enumerate(lines):
        if label in line:
            # Found it, try to determine context range
            # Look for previous directive or heading
            start = max(0, i - 10)

            # Look for next directive or heading
            end = min(len(lines), i + 50)
            for j in range(i + 1, min(len(lines), i + 100)):
                if re.match(r"^#+\s+", lines[j]) or re.match(r"^:::+\s*{", lines[j]):
                    end = j
                    break

            return (start + 1, end + 1)  # 1-indexed

    return None


def enrich_with_text_or_label(
    entity_path: Path, markdown_content: str, markdown_lines_count: int
) -> tuple[bool, str | None]:
    """Try text matching or broad label search."""
    try:
        with open(entity_path, encoding="utf-8") as f:
            entity = json.load(f)

        sl = entity.get("source_location", {})
        if sl and sl.get("line_range"):
            return (True, None)  # Already has

        line_range = None
        method = None

        # Try 1: Text matching with natural_language_statement
        if entity.get("natural_language_statement"):
            search_text = extract_first_sentences(entity["natural_language_statement"], 2)
            if search_text and len(search_text) > 20:
                line_range = find_text_in_markdown(markdown_content, search_text)
                if line_range:
                    method = "text matching (natural_language_statement)"

        # Try 2: Text matching with full_statement_text
        if not line_range and entity.get("full_statement_text"):
            search_text = extract_first_sentences(entity["full_statement_text"], 2)
            if search_text and len(search_text) > 20:
                line_range = find_text_in_markdown(markdown_content, search_text)
                if line_range:
                    method = "text matching (full_statement_text)"

        # Try 3: Broad label search
        if not line_range:
            label = entity.get("label") or entity.get("label_text")
            if label:
                line_range = find_label_anywhere(markdown_content, label)
                if line_range:
                    method = "label mention search"

        if not line_range:
            return (False, "No text content and label not found")

        # Validate line_range
        if line_range[1] > markdown_lines_count:
            line_range = (line_range[0], markdown_lines_count)

        # Update
        if "source_location" not in entity:
            entity["source_location"] = {}

        entity["source_location"]["line_range"] = list(line_range)

        # Save
        with open(entity_path, "w", encoding="utf-8") as f:
            json.dump(entity, f, indent=2, ensure_ascii=False)

        return (True, method)

    except Exception as e:
        return (False, f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Aggressive text matching")
    parser.add_argument("--document", "-d", type=Path, required=True)
    args = parser.parse_args()

    if not args.document.exists():
        print(f"Error: Not found: {args.document}")
        sys.exit(1)

    document_id = args.document.name
    raw_data_dir = args.document / "raw_data"
    markdown_file = args.document.parent / f"{document_id}.md"

    if not markdown_file.exists():
        print(f"Error: Markdown not found: {markdown_file}")
        sys.exit(1)

    markdown_content = markdown_file.read_text(encoding="utf-8")
    line_count = len(markdown_content.splitlines())

    stats = {"total": 0, "already_had": 0, "enriched": 0, "failed": 0}

    enriched_by_method = {}

    for json_file in sorted(raw_data_dir.rglob("*.json")):
        if "report" in json_file.name.lower():
            continue

        stats["total"] += 1

        success, result = enrich_with_text_or_label(json_file, markdown_content, line_count)

        if success and result is None:
            stats["already_had"] += 1
        elif success:
            stats["enriched"] += 1
            enriched_by_method[json_file.name] = result
        else:
            stats["failed"] += 1

    print(f"\n{'=' * 80}")
    print(f"AGGRESSIVE TEXT MATCHING REPORT: {document_id}")
    print(f"{'=' * 80}")
    print(f"Total: {stats['total']}")
    print(f"  ✓ Already had line_range: {stats['already_had']}")
    print(f"  ✓ Enriched: {stats['enriched']}")
    print(f"  ✗ Failed: {stats['failed']}")

    if enriched_by_method:
        print(f"\n{'─' * 80}")
        print("ENRICHED FILES:")
        print(f"{'─' * 80}")
        for fname, method in sorted(enriched_by_method.items())[:20]:
            print(f"  ✓ {fname}")
            print(f"     Method: {method}")
        if len(enriched_by_method) > 20:
            print(f"\n... and {len(enriched_by_method) - 20} more")


if __name__ == "__main__":
    main()
