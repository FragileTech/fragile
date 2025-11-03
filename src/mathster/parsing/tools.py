"""
Text processing tools for mathematical document parsing.

**⚠️ DEPRECATED: This file is legacy code. Use the modular API instead:**

```python
# OLD (deprecated):
from mathster.parsing.tools import add_line_numbers, split_markdown_by_chapters_with_line_numbers

# NEW (recommended):
from mathster.parsing.text_processing import add_line_numbers, split_markdown_by_chapters_with_line_numbers
```

⚠️ For new code, use: `from mathster.parsing.text_processing import ...`
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mathster.core.raw_data import RawDocumentSection

def add_line_numbers(document: str, padding: bool = True, offset: int = 0) -> str:
    """Add line numbers to each line of a document.

    Args:
        document: The text document to add line numbers to
        padding: Whether to pad line numbers with spaces for alignment
        offset: Starting line number offset (default=0, so first line is 1)

    Returns:
        The document with line numbers prepended to each line
    """
    lines = document.split("\n")
    max_line_num = len(lines) + offset
    max_digits = len(str(max_line_num))

    if padding:
        numbered_lines = [
            f"{str(i + 1 + offset).rjust(max_digits)}: {line}" for i, line in enumerate(lines)
        ]
    else:
        numbered_lines = [f"{i + 1 + offset}: {line}" for i, line in enumerate(lines)]

    return "\n".join(numbered_lines)


def split_markdown_by_chapters(file_path: str | Path, header: str = "##") -> list[str]:
    """
    Split a markdown file by chapters based on header level.

    Args:
        file_path: Path to the markdown file to split
        header: The header marker to use for splitting (e.g., "##" for level 2 headers)

    Returns:
        A list of strings where:
        - The first item (chapter 0) contains everything before the first header
        - Subsequent items contain each chapter starting with its header

    Example:
        >>> chapters = split_markdown_by_chapters("03_cloning.md")
        >>> # chapters[0] contains content before first "##"
        >>> # chapters[1] contains "## 0. TLDR" and its content
        >>> # chapters[2] contains "## 1. Introduction" and its content
    """
    file_path = Path(file_path)

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    chapters = []
    current_chapter = []

    for line in lines:
        # Check if the line starts with the specified header
        if line.startswith(header + " "):
            # Save the current chapter if it has content
            if current_chapter or not chapters:
                chapters.append('\n'.join(current_chapter))
                current_chapter = []
            # Start a new chapter with this header line
            current_chapter.append(line)
        else:
            current_chapter.append(line)

    # Add the last chapter
    if current_chapter:
        chapters.append('\n'.join(current_chapter))

    return chapters


def split_markdown_by_chapters_with_line_numbers(
    file_path: str | Path,
    header: str = "##",
    padding: bool = True
) -> list[str]:
    """
    Split a markdown file by chapters and add continuous line numbers across all chapters.

    Args:
        file_path: Path to the markdown file to split
        header: The header marker to use for splitting (e.g., "##" for level 2 headers)
        padding: Whether to pad line numbers with spaces for alignment

    Returns:
        A list of strings where each chapter has line numbers that continue from the previous chapter.
        The first item (chapter 0) contains everything before the first header with line numbers.
        Subsequent items contain each chapter starting with its header, with continuous line numbering.

    Example:
        >>> chapters = split_markdown_by_chapters_with_line_numbers("03_cloning.md")
        >>> # chapters[0] contains content before first "##" with lines 1, 2, 3, ...
        >>> # chapters[1] contains "## 0. TLDR" continuing with lines N, N+1, N+2, ...
        >>> # chapters[2] contains "## 1. Introduction" continuing with lines M, M+1, M+2, ...
    """
    # First, split the markdown into chapters
    chapters = split_markdown_by_chapters(file_path, header)

    # Now add line numbers with continuous counting
    numbered_chapters = []
    current_offset = 0

    for chapter in chapters:
        # Add line numbers starting from the current offset
        numbered_chapter = add_line_numbers(chapter, padding=padding, offset=current_offset)
        numbered_chapters.append(numbered_chapter)

        # Update offset for next chapter (count lines in current chapter)
        current_offset += len(chapter.split('\n'))

    return numbered_chapters


def classify_label(label: str) -> str:
    """
    Classify a label by its prefix into an entity type.

    Args:
        label: The label to classify (e.g., 'def-lipschitz', 'thm-convergence')

    Returns:
        Entity type string (e.g., 'definitions', 'theorems')

    Examples:
        >>> classify_label('def-lipschitz')
        'definitions'
        >>> classify_label('thm-main-result')
        'theorems'
        >>> classify_label('lem-gradient-bound')
        'lemmas'
        >>> classify_label('def-axiom-bounded')
        'axioms'
    """
    # Check more specific prefixes first to avoid incorrect matches
    # (e.g., "def-axiom-" should match "axioms", not "definitions")
    if label.startswith("def-axiom-"):
        return "axioms"
    elif label.startswith("axiom-") or label.startswith("ax-"):
        return "axioms"
    elif label.startswith("assumption-"):
        return "assumptions"
    elif label.startswith("def-"):
        return "definitions"
    elif label.startswith("thm-"):
        return "theorems"
    elif label.startswith("lem-"):
        return "lemmas"
    elif label.startswith("prop-"):
        return "propositions"
    elif label.startswith("cor-"):
        return "corollaries"
    elif label.startswith("param-"):
        return "parameters"
    elif label.startswith("remark-"):
        return "remarks"
    elif label.startswith("proof-"):
        return "proofs"
    elif label.startswith("cite-"):
        return "citations"
    else:
        return "other"


def analyze_labels_in_chapter(chapter_text: str) -> tuple[dict[str, list[str]], str]:
    """
    Analyze a chapter to extract and classify all :label: directives.

    This function pre-processes chapter text to discover what mathematical entities
    exist (by their labels) before extraction. The report helps guide the DSPy ReAct
    agent by providing:
    1. Exact labels to use (no guessing)
    2. Complete list of entities to extract
    3. Entity type classification

    Args:
        chapter_text: Markdown chapter text (with or without line numbers)

    Returns:
        A tuple of (labels_dict, report_string) where:
        - labels_dict: Dictionary mapping entity type → list of label strings
        - report_string: Human-readable report formatted for LLM consumption

    Example:
        >>> text = '''
        ... :::{prf:theorem} Main Result
        ... :label: thm-main-result
        ... :::
        ...
        ... :::{prf:definition} Lipschitz Continuous
        ... :label: def-lipschitz
        ... :::
        ... '''
        >>> labels_dict, report = analyze_labels_in_chapter(text)
        >>> labels_dict
        {'theorems': ['thm-main-result'], 'definitions': ['def-lipschitz']}
        >>> print(report)
        LABELS FOUND IN DOCUMENT:
        =========================
        <BLANKLINE>
        Definitions (1):
          - def-lipschitz
        <BLANKLINE>
        Theorems (1):
          - thm-main-result
        ...
    """
    import re

    # Extract all :label: directives using regex
    # Pattern matches ":label: <label-name>" where label follows pattern ^[a-z]+-[a-z0-9-]+$
    label_pattern = r':label:\s+([a-z][a-z0-9_-]+)'
    matches = re.findall(label_pattern, chapter_text, re.MULTILINE)

    # Classify labels by entity type
    labels_by_type: dict[str, list[str]] = {}
    for label in matches:
        entity_type = classify_label(label)
        if entity_type not in labels_by_type:
            labels_by_type[entity_type] = []
        labels_by_type[entity_type].append(label)

    # Build report string
    report_lines = [
        "LABELS FOUND IN DOCUMENT:",
        "=" * 50,
        ""
    ]

    # Sort entity types for consistent output
    entity_order = [
        "definitions",
        "theorems",
        "lemmas",
        "propositions",
        "corollaries",
        "axioms",
        "assumptions",
        "parameters",
        "remarks",
        "proofs",
        "citations",
        "other"
    ]

    total_count = 0
    for entity_type in entity_order:
        if entity_type in labels_by_type and labels_by_type[entity_type]:
            labels = labels_by_type[entity_type]
            count = len(labels)
            total_count += count

            # Capitalize entity type for display
            display_name = entity_type.capitalize()
            report_lines.append(f"{display_name} ({count}):")
            for label in sorted(labels):  # Sort labels alphabetically
                report_lines.append(f"  - {label}")
            report_lines.append("")

    # Add summary
    report_lines.append("=" * 50)
    report_lines.append(f"TOTAL: {total_count} labeled entities")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Add instruction for LLM
    if total_count > 0:
        report_lines.append("EXTRACTION INSTRUCTIONS:")
        report_lines.append("- Extract ALL entities listed above using their EXACT labels")
        report_lines.append("- Use these labels verbatim in your extraction output")
        report_lines.append("- For entities without explicit :label: directives, generate standardized labels")
        report_lines.append("- Cross-check: your extraction should include all labels listed above")
    else:
        report_lines.append("NOTE: No explicit :label: directives found in this chapter.")
        report_lines.append("Generate appropriate labels for all entities you extract.")

    report_lines.append("")

    report_string = "\n".join(report_lines)

    return labels_by_type, report_string


def _extract_labels_from_data(data: "RawDocumentSection | dict") -> dict[str, list[str]]:
    """
    Extract all labels from RawDocumentSection or dictionary by entity type.

    This is a helper function for compare_extraction_with_source() that normalizes
    label extraction from either a Pydantic model or a raw dictionary.

    Args:
        data: Either a RawDocumentSection instance or a dict with entity lists

    Returns:
        Dictionary mapping entity type to list of label strings

    Example:
        >>> from mathster.core.raw_data import RawDocumentSection
        >>> section = RawDocumentSection(...)
        >>> labels = _extract_labels_from_data(section)
        >>> labels
        {'definitions': ['def-lipschitz'], 'theorems': ['thm-main']}
    """
    labels_by_type: dict[str, list[str]] = {}

    # Entity types to extract (includes all theorem-like entities)
    entity_types = [
        "definitions",
        "theorems",
        "lemmas",
        "propositions",
        "corollaries",
        "proofs",
        "axioms",
        "assumptions",
        "parameters",
        "remarks",
        "citations",
    ]

    if isinstance(data, dict):
        # Extract from dictionary format
        for entity_type in entity_types:
            if entity_type in data and data[entity_type]:
                # Special handling for theorems: classify by statement_type if present
                if entity_type == "theorems":
                    for entity in data[entity_type]:
                        if isinstance(entity, dict):
                            label = entity.get("label")
                            stmt_type = entity.get("statement_type")
                            if label:
                                # Classify by statement_type: "theorem" → "theorems", "lemma" → "lemmas"
                                # Special case: "corollary" → "corollaries" (not "corollarys")
                                if stmt_type == "corollary":
                                    target_type = "corollaries"
                                elif stmt_type:
                                    target_type = f"{stmt_type}s"
                                else:
                                    target_type = "theorems"
                                if target_type not in labels_by_type:
                                    labels_by_type[target_type] = []
                                labels_by_type[target_type].append(label)
                        else:
                            # Object with attributes
                            label = getattr(entity, "label", None)
                            stmt_type = getattr(entity, "statement_type", None)
                            if label:
                                # Special case: "corollary" → "corollaries"
                                if stmt_type == "corollary":
                                    target_type = "corollaries"
                                elif stmt_type:
                                    target_type = f"{stmt_type}s"
                                else:
                                    target_type = "theorems"
                                if target_type not in labels_by_type:
                                    labels_by_type[target_type] = []
                                labels_by_type[target_type].append(label)
                else:
                    # Standard handling for other entity types
                    labels = []
                    for entity in data[entity_type]:
                        if isinstance(entity, dict):
                            label = entity.get("label")
                        else:
                            # Assume it's an object with .label attribute
                            label = getattr(entity, "label", None)
                        if label:
                            labels.append(label)
                    if labels:
                        labels_by_type[entity_type] = labels
    else:
        # Extract from RawDocumentSection (Pydantic model)
        if hasattr(data, "definitions") and data.definitions:
            labels_by_type["definitions"] = [d.label for d in data.definitions]
        if hasattr(data, "theorems") and data.theorems:
            # RawDocumentSection stores all theorem-like entities in a single 'theorems' list
            # Classify them by their statement_type field ("theorem", "lemma", "proposition", "corollary")
            for thm in data.theorems:
                # Map statement_type to entity_type: "theorem" → "theorems", "lemma" → "lemmas"
                # Special case: "corollary" → "corollaries" (not "corollarys")
                if thm.statement_type == "corollary":
                    entity_type = "corollaries"
                else:
                    entity_type = f"{thm.statement_type}s"
                if entity_type not in labels_by_type:
                    labels_by_type[entity_type] = []
                labels_by_type[entity_type].append(thm.label)
        if hasattr(data, "proofs") and data.proofs:
            labels_by_type["proofs"] = [p.label for p in data.proofs]
        if hasattr(data, "axioms") and data.axioms:
            labels_by_type["axioms"] = [a.label for a in data.axioms]
        if hasattr(data, "assumptions") and data.assumptions:
            labels_by_type["assumptions"] = [a.label for a in data.assumptions]
        if hasattr(data, "parameters") and data.parameters:
            labels_by_type["parameters"] = [p.label for p in data.parameters]
        if hasattr(data, "remarks") and data.remarks:
            labels_by_type["remarks"] = [r.label for r in data.remarks]
        if hasattr(data, "citations") and data.citations:
            # Citations use key_in_text, not label
            labels_by_type["citations"] = [c.key_in_text for c in data.citations]

    return labels_by_type


def _format_comparison_report(comparison: dict) -> str:
    """
    Format comparison data into human-readable validation report.

    Args:
        comparison: Dictionary with comparison results from compare_extraction_with_source()

    Returns:
        Formatted report string with validation summary and detailed breakdowns
    """
    report_lines = [
        "EXTRACTION VALIDATION REPORT",
        "=" * 70,
        "",
        "Comparing extracted labels against source document labels.",
        "",
    ]

    # Summary section
    summary = comparison.get("summary", {})
    report_lines.append("SUMMARY:")
    report_lines.append(f"  Labels in source text: {summary.get('total_in_text', 0)}")
    report_lines.append(f"  Labels in extracted data: {summary.get('total_in_data', 0)}")
    report_lines.append(f"  ✓ Correct matches: {summary.get('correct_matches', 0)}")
    report_lines.append(f"  ✗ Hallucinated (in data, not in text): {summary.get('hallucinated', 0)}")
    report_lines.append(f"  ⚠ Missed (in text, not in data): {summary.get('missed', 0)}")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Detailed breakdown by entity type
    entity_order = [
        "definitions",
        "theorems",
        "lemmas",
        "propositions",
        "corollaries",
        "axioms",
        "assumptions",
        "parameters",
        "remarks",
        "proofs",
        "citations",
    ]

    for entity_type in entity_order:
        if entity_type not in comparison or entity_type == "summary":
            continue

        data = comparison[entity_type]
        found = data.get("found", [])
        missing = data.get("missing_from_text", [])
        not_extracted = data.get("not_extracted", [])

        if found or missing or not_extracted:
            report_lines.append(f"{entity_type.upper()}:")

            if found:
                report_lines.append(f"  ✓ Found ({len(found)}) - correctly extracted:")
                for label in found:
                    report_lines.append(f"      {label}")

            if missing:
                report_lines.append(f"  ✗ HALLUCINATED ({len(missing)}) - in data but NOT in text:")
                for label in missing:
                    report_lines.append(f"      {label}")

            if not_extracted:
                report_lines.append(f"  ⚠ MISSED ({len(not_extracted)}) - in text but NOT extracted:")
                for label in not_extracted:
                    report_lines.append(f"      {label}")

            report_lines.append("")

    # Validation status
    report_lines.append("=" * 70)
    hallucinated = summary.get("hallucinated", 0)
    missed = summary.get("missed", 0)

    if hallucinated == 0 and missed == 0:
        report_lines.append("✓ VALIDATION PASSED: Perfect match between text and data")
    elif hallucinated > 0 and missed > 0:
        report_lines.append("✗ VALIDATION FAILED: Both hallucinations and missed extractions detected")
        report_lines.append("  Action required:")
        report_lines.append(f"    - Remove {hallucinated} hallucinated label(s) from data")
        report_lines.append(f"    - Re-extract to capture {missed} missed label(s)")
    elif hallucinated > 0:
        report_lines.append("✗ VALIDATION FAILED: Hallucinated labels detected")
        report_lines.append(f"  Action required: Remove {hallucinated} hallucinated label(s) from data")
    elif missed > 0:
        report_lines.append("⚠ VALIDATION WARNING: Some labels missed in extraction")
        report_lines.append(f"  Action recommended: Re-extract to capture {missed} missed label(s)")

    report_lines.append("=" * 70)
    report_lines.append("")

    return "\n".join(report_lines)


def compare_extraction_with_source(
    extracted_data: "RawDocumentSection | dict",
    chapter_text: str
) -> tuple[dict[str, dict], str]:
    """
    Compare extracted labels against source document to validate extraction quality.

    This function performs a comprehensive comparison between labels found in the
    source markdown text and labels present in the extracted data. It detects:
    - Correct matches (labels in both text and data)
    - Hallucinated labels (in data but NOT in source text) - ERRORS
    - Missed labels (in source text but NOT in extracted data) - WARNINGS

    This is critical for quality control to catch LLM hallucinations and ensure
    extraction completeness.

    Args:
        extracted_data: Either a RawDocumentSection instance or a dict with entity lists
        chapter_text: Source markdown chapter text

    Returns:
        A tuple of (comparison_dict, report_string) where:
        - comparison_dict: Detailed comparison data by entity type
        - report_string: Human-readable validation report

    Example:
        >>> from mathster.core.raw_data import RawDocumentSection
        >>> section = RawDocumentSection(...)
        >>> chapter = ":::{prf:theorem}\\n:label: thm-main\\n:::"
        >>> comparison, report = compare_extraction_with_source(section, chapter)
        >>> print(report)
        EXTRACTION VALIDATION REPORT
        ...
        >>> comparison["summary"]["hallucinated"]
        0
    """
    # Get labels from source text using existing analyzer
    text_labels, _ = analyze_labels_in_chapter(chapter_text)

    # Get labels from extracted data
    data_labels = _extract_labels_from_data(extracted_data)

    # Compare for each entity type
    comparison: dict[str, dict] = {}

    # Entity types to compare (includes both singular and plural forms for text labels)
    entity_types = [
        "definitions",
        "theorems",
        "lemmas",
        "propositions",
        "corollaries",
        "axioms",
        "assumptions",
        "parameters",
        "remarks",
        "proofs",
        "citations",
    ]

    total_stats = {
        "total_in_text": 0,
        "total_in_data": 0,
        "correct_matches": 0,
        "hallucinated": 0,
        "missed": 0,
    }

    for entity_type in entity_types:
        text_set = set(text_labels.get(entity_type, []))
        data_set = set(data_labels.get(entity_type, []))

        found = text_set & data_set
        missing_from_text = data_set - text_set  # HALLUCINATED
        not_extracted = text_set - data_set  # MISSED

        comparison[entity_type] = {
            "found": sorted(found),
            "missing_from_text": sorted(missing_from_text),
            "not_extracted": sorted(not_extracted),
        }

        # Update totals
        total_stats["total_in_text"] += len(text_set)
        total_stats["total_in_data"] += len(data_set)
        total_stats["correct_matches"] += len(found)
        total_stats["hallucinated"] += len(missing_from_text)
        total_stats["missed"] += len(not_extracted)

    comparison["summary"] = total_stats

    # Generate formatted report
    report = _format_comparison_report(comparison)

    return comparison, report
