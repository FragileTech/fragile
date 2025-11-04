"""
Label analysis and extraction comparison utilities.

Provides functions for classifying labels, analyzing labels in chapters,
and comparing extractions with source documents.
"""

import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass


def classify_label(label: str) -> str:
    """
    Classify a label by its prefix into an entity type.

    Args:
        label: Entity label (e.g., 'def-lipschitz', 'thm-main-result')

    Returns:
        Entity type string ('definition', 'theorem', 'proof', etc.)

    Example:
        >>> classify_label("def-lipschitz")
        'definition'
        >>> classify_label("thm-convergence")
        'theorem'
    """
    if label.startswith("def-"):
        return "definition"
    if label.startswith("thm-"):
        return "theorem"
    if label.startswith("lem-"):
        return "lemma"
    if label.startswith("prop-"):
        return "proposition"
    if label.startswith("cor-"):
        return "corollary"
    if label.startswith("proof-"):
        return "proof"
    if label.startswith(("axiom-", "ax-", "def-axiom-")):
        return "axiom"
    if label.startswith("param-"):
        return "parameter"
    if label.startswith("remark-"):
        return "remark"
    if label.startswith("assumption-"):
        return "assumption"
    if label.startswith("cite-"):
        return "citation"
    return "unknown"


def analyze_labels_in_chapter(chapter_text: str) -> dict[str, list[str]]:
    """
    Analyze a chapter to find all Jupyter Book directive labels.

    Searches for :label: directives in the chapter text and classifies them.

    Args:
        chapter_text: The chapter text to analyze

    Returns:
        Dictionary mapping entity types to lists of labels found

    Example:
        >>> labels = analyze_labels_in_chapter(chapter_with_directives)
        >>> labels["definitions"]
        ['def-lipschitz', 'def-bounded']
        >>> labels["theorems"]
        ['thm-convergence', 'thm-main-result']
    """
    # Pattern to match :label: directives
    label_pattern = r":label:\s+([a-z]+-[a-z0-9-_]+)"

    labels_by_type = {
        "definitions": [],
        "theorems": [],
        "lemmas": [],
        "propositions": [],
        "corollaries": [],
        "proofs": [],
        "axioms": [],
        "parameters": [],
        "remarks": [],
        "assumptions": [],
        "citations": [],
        "unknown": [],
    }

    # Find all labels
    for match in re.finditer(label_pattern, chapter_text, re.IGNORECASE):
        label = match.group(1)
        entity_type = classify_label(label)

        if entity_type == "theorem":
            labels_by_type["theorems"].append(label)
        elif entity_type == "lemma":
            labels_by_type["lemmas"].append(label)
        elif entity_type == "proposition":
            labels_by_type["propositions"].append(label)
        elif entity_type == "corollary":
            labels_by_type["corollaries"].append(label)
        elif entity_type in labels_by_type:
            labels_by_type[entity_type + "s"].append(label)
        else:
            labels_by_type["unknown"].append(label)

    return labels_by_type


def compare_extraction_with_source(extraction_dict: dict, chapter_text: str) -> str:
    """
    Compare extraction with source text to find missed labels.

    Args:
        extraction_dict: Dictionary representation of ChapterExtraction
        chapter_text: Original chapter text

    Returns:
        Comparison report string showing extracted vs. found labels

    Example:
        >>> report = compare_extraction_with_source(extraction, chapter_text)
        >>> print(report)
        Definitions: Extracted 3, Found in source 5
        Missing: def-bounded, def-continuous
    """
    # Analyze source text
    source_labels = analyze_labels_in_chapter(chapter_text)

    # Extract labels from extraction
    extracted_labels = {
        "definitions": [d.get("label") for d in extraction_dict.get("definitions", [])],
        "theorems": [t.get("label") for t in extraction_dict.get("theorems", [])],
        "proofs": [p.get("label") for p in extraction_dict.get("proofs", [])],
        "axioms": [a.get("label") for a in extraction_dict.get("axioms", [])],
        "parameters": [p.get("label") for p in extraction_dict.get("parameters", [])],
        "remarks": [r.get("label") for r in extraction_dict.get("remarks", [])],
        "assumptions": [a.get("label") for a in extraction_dict.get("assumptions", [])],
    }

    # Build comparison report
    report_lines = []
    total_extracted = 0
    total_found = 0
    total_missed = 0

    for entity_type in [
        "definitions",
        "theorems",
        "proofs",
        "axioms",
        "parameters",
        "remarks",
        "assumptions",
    ]:
        extracted = set(extracted_labels.get(entity_type, []))
        found = set(source_labels.get(entity_type, []))

        # Also check lemmas/propositions/corollaries for theorems
        if entity_type == "theorems":
            found.update(source_labels.get("lemmas", []))
            found.update(source_labels.get("propositions", []))
            found.update(source_labels.get("corollaries", []))

        missed = found - extracted

        if found:
            report_lines.append(
                f"{entity_type.capitalize()}: Extracted {len(extracted)}, Found in source {len(found)}"
            )
            if missed:
                report_lines.append(f"  Missing: {', '.join(sorted(missed))}")
                total_missed += len(missed)

        total_extracted += len(extracted)
        total_found += len(found)

    # Add summary
    report_lines.insert(
        0, f"Total: Extracted {total_extracted}, Found {total_found}, Missed {total_missed}"
    )
    report_lines.insert(1, "")

    return "\n".join(report_lines)
