#!/usr/bin/env python3
"""
Handle Special Cases for Remaining Entities.

Handles parameters, embedded mathster, remarks, and unlabeled entities.
"""

import argparse
import json
from pathlib import Path
import re


def find_parameter_in_text(markdown_content: str, param_label: str) -> tuple[int, int] | None:
    """Find parameter definition in markdown (often in tables or inline)."""
    lines = markdown_content.splitlines()

    # Try to find parameter symbol or name
    # Common patterns: $N$, $\kappa_variance$, etc.
    patterns = [
        rf"\${re.escape(param_label)}\$",  # LaTeX inline
        rf"{re.escape(param_label)}",  # Plain text
        param_label.replace("param-", "").replace("-", "_"),  # Convert to Python name
        param_label.replace("param-", "").replace("-", " "),  # Convert to words
    ]

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Found it - return context
                start = max(1, i - 5)
                end = min(len(lines), i + 20)
                return (start, end)

    return None


def find_proof_in_theorem(markdown_content: str, proof_label: str) -> tuple[int, int] | None:
    """Find proof embedded within theorem block."""
    lines = markdown_content.splitlines()

    # Extract theorem label from proof label
    # proof-lem-foo -> look for lem-foo theorem
    # proof-thm-foo -> look for thm-foo theorem
    theorem_label = proof_label.replace("proof-", "")

    # Find theorem with this label
    for i, line in enumerate(lines):
        if f":label: {theorem_label}" in line:
            # Found theorem, look for proof block inside
            # Proof blocks often start with "**Proof:**" or ":::{prf:proof}"
            for j in range(i, min(len(lines), i + 200)):
                if re.search(r"\*\*Proof[:\.]?\*\*|:::\{prf:proof\}", lines[j], re.IGNORECASE):
                    # Found proof start
                    # Find end (next ::: or next section/theorem)
                    for k in range(j + 1, min(len(lines), j + 150)):
                        if re.match(r"^:::+\s*$", lines[k]) or re.match(r"^#+\s+", lines[k]):
                            return (j + 1, k + 1)

                    # No clear end, return to end of theorem
                    return (j + 1, min(len(lines), j + 100))

            # Found theorem but no proof block - return None
            return None

    return None


def find_remark_in_admonition(markdown_content: str, keywords: str) -> tuple[int, int] | None:
    """Find remark in note/tip/warning admonition."""
    lines = markdown_content.splitlines()

    # Extract keywords from filename
    search_terms = keywords.replace("remark-", "").replace("-", " ").split()

    for i, line in enumerate(lines):
        # Check if this is an admonition line
        if re.match(r"^::::?\{(note|tip|important|warning|admonition)\}", line):
            # Check next 20 lines for keywords
            context = " ".join(lines[i : min(len(lines), i + 20)]).lower()
            if any(term in context for term in search_terms):
                # Found it - find end
                for j in range(i + 1, min(len(lines), i + 50)):
                    if re.match(r"^:::+\s*$", lines[j]):
                        return (i + 1, j + 1)

                return (i + 1, min(len(lines), i + 30))

    return None


def enrich_special_case(
    entity_path: Path, markdown_content: str, markdown_lines: int
) -> tuple[bool, str | None]:
    """Handle special case entities."""
    try:
        with open(entity_path, encoding="utf-8") as f:
            entity = json.load(f)

        sl = entity.get("source_location", {})
        if sl and sl.get("line_range"):
            return (True, None)

        label = entity.get("label") or entity.get("label_text")
        if not label:
            return (False, "No label")

        line_range = None
        method = None

        # Try 1: Parameters
        if "param" in entity_path.name:
            line_range = find_parameter_in_text(markdown_content, label)
            if line_range:
                method = "parameter text search"

        # Try 2: Embedded mathster
        elif "proof-" in entity_path.name:
            line_range = find_proof_in_theorem(markdown_content, label)
            if line_range:
                method = "proof in theorem search"

        # Try 3: Remarks
        elif "remark" in entity_path.name:
            line_range = find_remark_in_admonition(markdown_content, entity_path.name)
            if line_range:
                method = "remark in admonition search"

        if not line_range:
            return (False, "Not found")

        # Validate
        if line_range[1] > markdown_lines:
            line_range = (line_range[0], markdown_lines)

        # Update
        if "source_location" not in entity:
            entity["source_location"] = {}

        entity["source_location"]["line_range"] = list(line_range)

        with open(entity_path, "w", encoding="utf-8") as f:
            json.dump(entity, f, indent=2, ensure_ascii=False)

        return (True, method)

    except Exception as e:
        return (False, f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", "-d", type=Path, required=True)
    args = parser.parse_args()

    document_id = args.document.name
    raw_data_dir = args.document / "raw_data"
    markdown_file = args.document.parent / f"{document_id}.md"

    markdown_content = markdown_file.read_text(encoding="utf-8")
    line_count = len(markdown_content.splitlines())

    stats = {"total": 0, "already_had": 0, "enriched": 0, "failed": 0}
    enriched = {}

    for json_file in sorted(raw_data_dir.rglob("*.json")):
        if "report" in json_file.name.lower():
            continue

        stats["total"] += 1
        success, result = enrich_special_case(json_file, markdown_content, line_count)

        if success and result is None:
            stats["already_had"] += 1
        elif success:
            stats["enriched"] += 1
            enriched[json_file.name] = result
        else:
            stats["failed"] += 1

    print(f"\n{'=' * 80}")
    print(f"SPECIAL CASES REPORT: {document_id}")
    print(f"{'=' * 80}")
    print(f"Total: {stats['total']}")
    print(f"  ✓ Enriched: {stats['enriched']}")
    print(f"  ✗ Failed: {stats['failed']}")

    if enriched:
        print(f"\n{'─' * 80}")
        print("ENRICHED:")
        print(f"{'─' * 80}")
        for fname, method in sorted(enriched.items())[:30]:
            print(f"  ✓ {fname}: {method}")


if __name__ == "__main__":
    main()
