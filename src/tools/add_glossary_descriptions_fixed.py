#!/usr/bin/env python3
"""
Add concise descriptions to glossary.md entries - fixed version.
Properly handles the entry structure.
"""

from pathlib import Path
import re


def generate_smart_description(title: str, entry_type: str) -> str:
    """Generate a smart description based on title and type."""

    # Clean title - remove LaTeX and special formatting
    clean_title = re.sub(r"\$[^\$]+\$", "", title)
    clean_title = re.sub(r"[_\{\}\\]", "", clean_title)
    clean_title = clean_title.strip()

    # Extract key concepts from title
    title_lower = clean_title.lower()

    # Type-specific smart descriptions
    if entry_type == "Definition":
        if "axiom" in title_lower:
            core = clean_title.replace("Axiom of ", "").replace("Axiom", "").strip()
            return f"Fundamental assumption about {core.lower()}"
        if "operator" in title_lower:
            return f"Defines {clean_title.lower()} on swarm states"
        if "metric" in title_lower or "distance" in title_lower:
            return f"Distance measure for {clean_title.lower()}"
        if "space" in title_lower:
            return f"Mathematical space for {clean_title.replace('Space', '').strip().lower()}"
        if "measure" in title_lower or "kernel" in title_lower:
            return f"Probability measure for {clean_title.lower()}"
        return f"Defines {clean_title.lower()}"

    if entry_type == "Theorem":
        if "convergence" in title_lower:
            return "Proves convergence to equilibrium"
        if "contraction" in title_lower:
            return "Establishes distance contraction property"
        if "bound" in title_lower:
            return f"Bounds {clean_title.replace('Bound', '').strip().lower()}"
        if "uniqueness" in title_lower:
            return "Proves uniqueness of solution"
        # Truncate if too long
        desc = f"Main result: {clean_title[:40]}"
        return desc if len(desc.split()) <= 15 else " ".join(desc.split()[:15])

    if entry_type == "Lemma":
        if "bound" in title_lower:
            return "Technical bound for analysis"
        if "decomposition" in title_lower:
            return "Decomposes quantity into components"
        if "continuity" in title_lower or "lipschitz" in title_lower:
            return "Establishes regularity property"
        desc = f"Supporting result: {clean_title[:45]}"
        return desc if len(desc.split()) <= 15 else " ".join(desc.split()[:15])

    if entry_type == "Proposition":
        desc = f"Establishes {clean_title[:50]}"
        return desc if len(desc.split()) <= 15 else " ".join(desc.split()[:15])

    if entry_type == "Corollary":
        return "Direct consequence of main result"

    if entry_type in {"Axiom", "Assumption"}:
        core = clean_title.replace("Axiom of", "").replace("Assumption", "").strip()
        return f"Fundamental assumption about {core.lower()}"

    if entry_type == "Remark":
        return f"Explanatory note on {clean_title[:40]}"

    if entry_type == "Algorithm":
        return (
            f"Computational procedure for {clean_title.replace('Algorithm', '').strip().lower()}"
        )

    if entry_type == "Observation":
        return "Observed property of system"

    # Fallback - just use title truncated
    words = clean_title.split()
    return " ".join(words[:12]) if len(words) > 12 else clean_title


def parse_glossary(lines: list[str]) -> list[dict]:
    """Parse glossary into structured entries."""
    entries = []
    current_entry = None
    current_metadata_end = None

    for i, line in enumerate(lines):
        if line.startswith("### ") and not line.startswith("####"):
            # New entry starts
            if current_entry and current_metadata_end:
                current_entry["metadata_end_line"] = current_metadata_end
                entries.append(current_entry)

            current_entry = {
                "title": line.strip("# \n"),
                "start_line": i,
                "type": None,
                "label": None,
                "tags": None,
                "source": None,
                "source_line": None,
                "has_description": False,
            }
            current_metadata_end = None

        elif current_entry:
            if line.startswith("- **Type:**"):
                current_entry["type"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **Label:**"):
                current_entry["label"] = line.split(":", 1)[1].strip().strip("`")
            elif line.startswith("- **Tags:**"):
                current_entry["tags"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **Source:**"):
                current_entry["source"] = line.split(":", 1)[1].strip()
                current_entry["source_line"] = i
            elif line.startswith("- **Description:**"):
                current_entry["has_description"] = True
                current_metadata_end = i
            elif line.strip() == "" and current_entry.get("source_line"):
                # Empty line after source = end of metadata
                current_metadata_end = i

    # Add last entry
    if current_entry and current_metadata_end:
        current_entry["metadata_end_line"] = current_metadata_end
        entries.append(current_entry)

    return entries


def main():
    """Add descriptions to glossary.md."""
    project_root = Path(__file__).parent.parent.parent
    glossary_path = project_root / "docs" / "glossary.md"

    print(f"Reading: {glossary_path}")

    with open(glossary_path, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Parsing {len(lines)} lines...")
    entries = parse_glossary(lines)
    print(f"Found {len(entries)} entries")

    # Add descriptions
    lines_to_insert = {}  # line_number -> description line
    added_count = 0

    for entry in entries:
        if not entry["has_description"] and entry.get("source_line") and entry.get("type"):
            desc = generate_smart_description(entry["title"], entry["type"])

            # Ensure <= 15 words
            words = desc.split()
            if len(words) > 15:
                desc = " ".join(words[:15])

            # Insert after Source line
            insert_line = entry["source_line"] + 1
            lines_to_insert[insert_line] = f"- **Description:** {desc}\n"
            added_count += 1

    print(f"Generating {added_count} descriptions...")

    # Insert descriptions
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if i in lines_to_insert:
            new_lines.append(lines_to_insert[i])
            if len(lines_to_insert) <= 20 or i in list(lines_to_insert.keys())[::50]:
                print(f"  Added description at line {i + 1}")

    # Write output
    with open(glossary_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✓ Added {added_count} descriptions")
    print(f"✓ Updated: {glossary_path}")

    # Verify
    with open(glossary_path, encoding="utf-8") as f:
        content = f.read()
    desc_count = content.count("- **Description:**")
    print(f"✓ Verification: {desc_count} descriptions in file")


if __name__ == "__main__":
    main()
