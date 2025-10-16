#!/usr/bin/env python3
"""
Extract mathematical objects from 14_scutoid_geometry_framework.md
and format them for inclusion in 00_reference.md
"""

from pathlib import Path
import re


def extract_math_objects(content: str) -> list[dict[str, str]]:
    """Extract all mathematical objects (definitions, theorems, etc.) from markdown."""

    objects = []

    # Pattern to match Jupyter Book directives: :::{prf:TYPE} TITLE\n:label: LABEL\n...\n:::
    pattern = r":::?\{prf:(definition|theorem|lemma|proposition|corollary|axiom|remark|example|algorithm)\}\s+(.*?)\n:label:\s+([\w\-]+)\n(.*?):::"

    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        obj_type = match.group(1).capitalize()
        title = match.group(2).strip()
        label = match.group(3).strip()
        body = match.group(4).strip()

        objects.append({
            "type": obj_type,
            "title": title,
            "label": label,
            "body": body,
            "full_text": match.group(0),
        })

    return objects


def infer_tags(title: str, body: str, obj_type: str) -> list[str]:
    """Infer appropriate tags based on content."""
    tags = ["scutoid-geometry"]

    # Content-based tags
    if "voronoi" in title.lower() or "voronoi" in body.lower():
        tags.append("voronoi-tessellation")
    if "curvature" in title.lower() or "ricci" in body.lower() or "sectional" in body.lower():
        tags.append("curvature")
    if "deficit" in title.lower() or "deficit" in body.lower():
        tags.append("deficit-angle")
    if "cloning" in title.lower() or "cloning" in body.lower():
        tags.append("cloning")
    if "laplacian" in title.lower() or "laplacian" in body.lower():
        tags.append("graph-laplacian")
    if "spectral" in title.lower() or "eigenvalue" in body.lower():
        tags.append("spectral-geometry")
    if "heat kernel" in title.lower() or "heat kernel" in body.lower():
        tags.append("heat-kernel")
    if "causal set" in title.lower() or "causal set" in body.lower():
        tags.append("causal-set")
    if "hellinger" in title.lower() or "kantorovich" in body.lower() or "hk" in body.lower():
        tags.append("hellinger-kantorovich")
    if "wasserstein" in title.lower() or "wasserstein" in body.lower():
        tags.append("wasserstein")
    if "convergence" in title.lower() or "gromov" in body.lower():
        tags.append("convergence")
    if "riemannian" in title.lower() or "riemannian" in body.lower() or "manifold" in body.lower():
        tags.append("riemannian-geometry")
    if "energy" in title.lower():
        tags.append("energy-functional")
    if "metric" in title.lower() or "metric" in body.lower():
        tags.append("metric-geometry")

    # Type-based tags
    if obj_type in {"Theorem", "Lemma"}:
        tags.append("major-result")

    return tags


def format_for_reference(obj: dict[str, str]) -> str:
    """Format a mathematical object for inclusion in 00_reference.md."""

    tags = infer_tags(obj["title"], obj["body"], obj["type"])
    tag_str = ", ".join([f"`{tag}`" for tag in tags])

    return f"""### {obj["title"]}

**Type:** {obj["type"]}
**Label:** `{obj["label"]}`
**Source:** [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md)
**Tags:** {tag_str}

**Statement:**

:::{{{obj["type"].lower()}}} {obj["title"]}
:label: {obj["label"]}

{obj["body"]}
:::

**Related Results:** See scutoid geometry framework results

---
"""


def main():
    """Main extraction process."""

    # Read the scutoid geometry framework document
    doc_path = Path(__file__).parent / "docs" / "source" / "14_scutoid_geometry_framework.md"

    if not doc_path.exists():
        print(f"Error: {doc_path} not found")
        return

    content = doc_path.read_text()

    # Extract all mathematical objects
    objects = extract_math_objects(content)

    print(f"Found {len(objects)} mathematical objects")
    print("\nBreakdown by type:")

    type_counts = {}
    for obj in objects:
        obj_type = obj["type"]
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

    for obj_type, count in sorted(type_counts.items()):
        print(f"  {obj_type}: {count}")

    # Format for reference document
    output_lines = [
        "## Scutoid Geometry Framework\n",
        "This section contains all mathematical definitions, theorems, and results from the Scutoid Geometry Framework (Chapter 14), which establishes the geometric structure of swarm spacetime evolution through scutoid-like volume cells connecting walker configurations across time slices.\n",
        "**Key Topics:** Riemannian scutoids, Voronoi tessellations, cloning topology, deficit angles, spectral curvature, heat kernel asymptotics, causal set volume, curvature unification\n",
        "---\n",
    ]

    for obj in objects:
        output_lines.append(format_for_reference(obj))

    output_text = "\n".join(output_lines)

    # Write to output file
    output_path = Path(__file__).parent / "SCUTOID_GEOMETRY_REFERENCE.md"
    output_path.write_text(output_text)

    print(f"\nOutput written to: {output_path}")
    print(f"Total length: {len(output_text)} characters")


if __name__ == "__main__":
    main()
