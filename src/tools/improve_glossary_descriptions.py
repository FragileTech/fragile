#!/usr/bin/env python3
"""
Improve glossary descriptions using reference.md as a knowledge base.
"""

from pathlib import Path
import re


def load_reference_descriptions(reference_path: Path) -> dict[str, str]:
    """Load descriptions from reference.md for known labels."""
    descriptions = {}

    with open(reference_path, encoding="utf-8") as f:
        content = f.read()

    # Find all entries in reference.md
    pattern = r"\*\*Label:\*\*\s*`([^`]+)`.*?\*\*What it says:\*\*\s*([^\n]+)"
    matches = re.findall(pattern, content, re.DOTALL)

    for label, desc in matches:
        # Clean and truncate description
        desc = desc.strip()
        # Remove any markdown formatting
        desc = re.sub(r"\*\*([^\*]+)\*\*", r"\1", desc)
        # Truncate to 15 words
        words = desc.split()
        if len(words) > 15:
            desc = " ".join(words[:15])
        descriptions[label] = desc

    return descriptions


def generate_concise_description(
    title: str, entry_type: str, reference_desc: str | None = None
) -> str:
    """Generate concise description, using reference if available."""

    if reference_desc:
        return reference_desc

    # Clean title
    clean_title = re.sub(r"\$[^\$]+\$", "", title)
    clean_title = re.sub(r"[_\{\}\\]", "", clean_title)
    clean_title = clean_title.strip()
    title_lower = clean_title.lower()

    # Enhanced type-specific descriptions
    if entry_type == "Definition":
        if "axiom" in title_lower:
            msg = clean_title.replace("Axiom of", "").strip().lower()
            return f"Fundamental assumption ensuring {msg}"
        if "walker" in title_lower:
            return "Position-velocity-status tuple representing single search agent"
        if "swarm" in title_lower:
            return "N-tuple of walkers forming collective population"
        if "operator" in title_lower:
            return f"Transformation {clean_title.lower()} applied to swarm"
        if "metric" in title_lower or "distance" in title_lower:
            return f"Measurement quantifying {clean_title.lower()}"
        if "qsd" in title_lower:
            return "Quasi-stationary distribution for killed process"
        if "measure" in title_lower or "kernel" in title_lower:
            return f"Probability measure governing {clean_title.lower()}"
        return f"Defines {clean_title[:60]}"[:80]

    if entry_type == "Theorem":
        if "convergence" in title_lower:
            return "Proves exponential convergence to equilibrium distribution"
        if "contraction" in title_lower:
            return "Establishes metric contraction under operator composition"
        if "uniqueness" in title_lower:
            return "Proves uniqueness of limiting distribution"
        if "lsi" in title_lower or "logarithmic sobolev" in title_lower:
            return "Logarithmic Sobolev inequality with N-uniform constant"
        return f"Main result: {clean_title[:60]}"[:80]

    if entry_type == "Lemma":
        if "bound" in title_lower:
            return f"Technical bound for {clean_title.replace('Bound', '').strip()[:40]}"[:80]
        if "lipschitz" in title_lower:
            return f"Lipschitz continuity of {clean_title.replace('Lipschitz', '').strip()[:40]}"[
                :80
            ]
        return f"Supporting result: {clean_title[:55]}"[:80]

    if entry_type == "Proposition":
        return f"Establishes {clean_title[:65]}"[:80]

    if entry_type == "Corollary":
        return f"Direct consequence: {clean_title[:60]}"[:80]

    if entry_type in {"Axiom", "Assumption"}:
        return f"Assumes {clean_title.replace('Axiom of', '').replace('Axiom', '').strip()[:60]}"[
            :80
        ]

    if entry_type == "Remark":
        return f"Note: {clean_title[:70]}"[:80]

    if entry_type == "Algorithm":
        return f"Procedure for {clean_title.replace('Algorithm', '').strip()[:60]}"[:80]

    return clean_title[:80]


def main():
    """Improve glossary descriptions."""
    project_root = Path(__file__).parent.parent.parent
    glossary_path = project_root / "docs" / "glossary.md"
    reference_path = project_root / "docs" / "reference.md"

    print("Loading reference.md knowledge...")
    ref_descriptions = load_reference_descriptions(reference_path)
    print(f"Loaded {len(ref_descriptions)} reference descriptions")

    print("\nReading glossary...")
    with open(glossary_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Process and improve descriptions
    new_lines = []
    i = 0
    improvements = 0
    current_entry = {}

    while i < len(lines):
        line = lines[i]

        # Track current entry metadata
        if line.startswith("### ") and not line.startswith("####"):
            current_entry = {"title": line.strip("# \n")}
        elif line.startswith("- **Type:**"):
            current_entry["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Label:**"):
            current_entry["label"] = line.split(":", 1)[1].strip().strip("`")

        # Improve description if it exists
        if line.startswith("- **Description:**"):
            old_desc = line.split(":", 1)[1].strip()

            # Get label if available
            label = current_entry.get("label", "")
            entry_type = current_entry.get("type", "")
            title = current_entry.get("title", "")

            # Try to get better description
            ref_desc = ref_descriptions.get(label) if label else None
            new_desc = generate_concise_description(title, entry_type, ref_desc)

            # Ensure <= 15 words
            words = new_desc.split()
            if len(words) > 15:
                new_desc = " ".join(words[:15])

            # Only update if different and better
            if new_desc != old_desc and len(new_desc) > 10:
                line = f"- **Description:** {new_desc}\n"
                improvements += 1

        new_lines.append(line)
        i += 1

    # Write improved version
    with open(glossary_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✓ Improved {improvements} descriptions")
    print(f"✓ Updated: {glossary_path}")


if __name__ == "__main__":
    main()
