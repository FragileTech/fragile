#!/usr/bin/env python3
"""
Add concise descriptions to glossary.md entries intelligently.

Uses title and type information to generate meaningful < 15 word descriptions.
"""

from pathlib import Path
import re


def generate_smart_description(title: str, entry_type: str, tags: str) -> str:
    """Generate a smart description based on title, type, and tags."""

    # Clean title - remove LaTeX and special formatting
    clean_title = re.sub(r"\$[^\$]+\$", "", title)
    clean_title = re.sub(r"[_\{\}]", "", clean_title)
    clean_title = clean_title.strip()

    # Extract key concepts from title
    title_lower = clean_title.lower()

    # Type-specific templates with smart extraction
    if entry_type == "Definition":
        if "axiom" in title_lower:
            return f"Fundamental assumption: {clean_title.replace('Axiom of ', '').lower()}"
        if "operator" in title_lower:
            return f"Defines {clean_title.lower()} operation on swarm state"
        if "metric" in title_lower or "distance" in title_lower:
            return f"Defines {clean_title.lower()} for measuring swarm similarity"
        if "measure" in title_lower or "kernel" in title_lower:
            return f"Probability {clean_title.lower()} for stochastic operations"
        if "space" in title_lower:
            return f"Mathematical space: {clean_title.lower()}"
        if "function" in title_lower:
            return f"Function defining {clean_title.replace('Function', '').strip().lower()}"
        return f"Defines {clean_title.lower()}"

    if entry_type == "Theorem":
        if "convergence" in title_lower:
            return f"Proves {clean_title.lower()} to equilibrium"
        if "contraction" in title_lower:
            return f"Establishes {clean_title.lower()} property"
        if "bound" in title_lower or "inequality" in title_lower:
            return f"Bounds {clean_title.lower()}"
        if "uniqueness" in title_lower:
            return (
                f"Proves uniqueness of {clean_title.replace('Uniqueness of', '').strip().lower()}"
            )
        if "existence" in title_lower:
            return f"Establishes existence of {clean_title.replace('Existence of', '').strip().lower()}"
        return f"Main result: {clean_title.lower()}"

    if entry_type == "Lemma":
        if "bound" in title_lower:
            return f"Technical bound on {clean_title.replace('Bound', '').strip().lower()}"
        if "decomposition" in title_lower:
            return f"Decomposes {clean_title.replace('Decomposition', '').strip().lower()}"
        if "continuity" in title_lower or "lipschitz" in title_lower:
            return f"Regularity: {clean_title.lower()}"
        return f"Supporting result for {clean_title.lower()}"

    if entry_type == "Proposition":
        if "property" in title_lower or "properties" in title_lower:
            return f"Characterizes {clean_title.replace('Properties', '').replace('Property', '').strip().lower()}"
        return f"Establishes {clean_title.lower()}"

    if entry_type == "Corollary":
        return f"Direct consequence: {clean_title.lower()}"

    if entry_type in {"Axiom", "Assumption"}:
        return f"Assumes {clean_title.replace('Axiom of', '').replace('Assumption', '').strip().lower()}"

    if entry_type == "Remark":
        return f"Note on {clean_title.lower()}"

    if entry_type == "Algorithm":
        return f"Procedure for {clean_title.lower()}"

    if entry_type == "Observation":
        return f"Observes {clean_title.lower()}"

    if entry_type == "Conjecture":
        return f"Conjectures {clean_title.lower()}"

    # Fallback
    return clean_title[:80] if len(clean_title) <= 80 else clean_title[:77] + "..."


def main():
    """Add descriptions to glossary.md."""
    project_root = Path(__file__).parent.parent.parent
    glossary_path = project_root / "docs" / "glossary.md"

    print(f"Processing: {glossary_path}")

    with open(glossary_path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    entries_processed = 0

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Check if this is an entry header
        if line.startswith("### ") and not line.startswith("#### ") and i > 40:
            title = line.strip("# \n")

            # Look ahead for Type, Label, Tags, Source
            entry_type = None
            tags = None

            j = i + 1
            while j < len(lines) and j < i + 10:
                if lines[j].startswith("- **Type:**"):
                    entry_type = lines[j].split(":", 1)[1].strip()
                elif lines[j].startswith("- **Label:**"):
                    lines[j].split(":", 1)[1].strip()
                elif lines[j].startswith("- **Tags:**"):
                    tags = lines[j].split(":", 1)[1].strip()
                elif lines[j].startswith("- **Source:**"):
                    new_lines.append(lines[j])

                    # Check if description already exists
                    has_description = j + 1 < len(lines) and lines[j + 1].startswith(
                        "- **Description:**"
                    )

                    if not has_description and entry_type:
                        # Generate and insert description
                        desc = generate_smart_description(title, entry_type, tags or "")
                        # Ensure < 15 words
                        words = desc.split()
                        if len(words) > 15:
                            desc = " ".join(words[:15])

                        desc_line = f"- **Description:** {desc}\n"
                        new_lines.append(desc_line)
                        entries_processed += 1

                        if entries_processed % 50 == 0:
                            print(f"Processed {entries_processed} entries...")

                    i = j
                    break
                elif lines[j].startswith("###"):
                    # Next entry, no source found
                    break
                else:
                    new_lines.append(lines[j])
                j += 1

        i += 1

    # Write output
    output_path = project_root / "docs" / "glossary.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✓ Added descriptions to {entries_processed} entries")
    print(f"✓ Updated: {output_path}")


if __name__ == "__main__":
    main()
