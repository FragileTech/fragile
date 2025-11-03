"""
Label sanitization and lookup utilities.

Provides functions for converting raw strings into valid SourceLocation labels
and looking up labels from text references in document context.
"""

import re
from typing import Literal


def sanitize_label(raw_label: str) -> str:
    """
    Sanitize a raw label into a valid SourceLocation label format.

    Converts any string into a valid label matching pattern: ^[a-z][a-z0-9_-]*$
    - Converts to lowercase
    - Uses hyphens to separate tag sections (type-name)
    - Preserves underscores ONLY within names (e.g., my_param)
    - Replaces other special characters with hyphens
    - Ensures it starts with a letter

    Tag structure rules:
    - Hyphens separate tag sections: param-my_param ✓
    - Underscores only within names: param_my_param ✗ (converted to param-my-param)
    - Common prefixes: param, def, thm, lem, cor, ax, section, prop, rem, cite

    Args:
        raw_label: Raw label string (may contain uppercase, spaces, periods, etc.)

    Returns:
        Sanitized label with only lowercase letters, digits, underscores, and hyphens

    Examples:
        >>> sanitize_label("## 1. Introduction")
        'section-1-introduction'
        >>> sanitize_label("param-Theta")
        'param-theta'
        >>> sanitize_label("param_my_param")
        'param-my-param'
        >>> sanitize_label("param-my_param")
        'param-my_param'
        >>> sanitize_label("def_Energy")
        'def-energy'
    """
    # Convert to lowercase (CRITICAL: prevents uppercase validation errors)
    label = raw_label.lower()

    # Remove leading markdown headers (##, ###, etc.)
    label = re.sub(r'^#+\s*', '', label)

    # Replace any sequence of non-alphanumeric characters (except underscores/hyphens) with a single hyphen
    # This preserves underscores and hyphens while converting other special chars
    label = re.sub(r'[^a-z0-9_-]+', '-', label)

    # Remove leading/trailing hyphens and underscores
    label = label.strip('-_')

    # Known tag prefixes that should be separated from names with hyphens
    # Format: {prefix}-{name}, where name can contain underscores
    prefixes = [
        'param', 'def', 'thm', 'lem', 'cor', 'ax', 'axiom',
        'section', 'prop', 'rem', 'remark', 'cite', 'eq',
        'obj', 'const', 'notation'
    ]

    # If label starts with a known prefix followed by underscore,
    # convert that first underscore to hyphen (it's a section separator, not part of the name)
    for prefix in prefixes:
        # Match: prefix + underscore + rest
        # Example: "param_my_param" → "param" + "_" + "my_param"
        pattern = f'^({prefix})_(.+)$'
        match = re.match(pattern, label)
        if match:
            # Convert prefix_name to prefix-name
            label = f"{match.group(1)}-{match.group(2)}"
            break

    # Ensure it starts with a letter (not a digit or underscore)
    if label and (label[0].isdigit() or label[0] == '_'):
        label = f"section-{label}"
    elif not label or not label[0].isalpha():
        label = f"section-{label}" if label else "section-unknown"

    # Collapse multiple consecutive hyphens (but preserve underscores)
    label = re.sub(r'-+', '-', label)

    return label


def lookup_label_from_context(
    reference_text: str,
    context: str,
    reference_type: Literal["theorem", "definition", "proof"],
) -> str:
    """
    Look up the actual label for a text reference in the document context.

    Strategy:
    1. Try to find :label: directive near the reference text in context
    2. If not found, generate standardized label from text

    Args:
        reference_text: Text reference like "Theorem 1.4" or "Lipschitz continuous"
        context: Chapter text with line numbers
        reference_type: Type of reference to help with pattern matching

    Returns:
        Label string (e.g., "thm-convergence" or "def-lipschitz-continuous")

    Examples:
        >>> lookup_label_from_context("Theorem 1.4", chapter_text, "theorem")
        "thm-convergence"  # Found :label: thm-convergence near "Theorem 1.4"

        >>> lookup_label_from_context("Lemma 2.3", chapter_text, "theorem")
        "lem-2-3"  # No :label: found, generated from text

        >>> lookup_label_from_context("Lipschitz continuous", chapter_text, "definition")
        "def-lipschitz-continuous"  # Generated from term
    """
    # Prefix mapping
    prefix_map = {
        "theorem": ["thm", "lem", "prop", "cor"],
        "definition": ["def"],
        "proof": ["proof"],
    }
    prefixes = prefix_map.get(reference_type, [])

    # Strategy 1: Search for :label: directive near reference text
    # Pattern: Look for Jupyter Book directive with :label: nearby

    # Normalize reference text for searching
    search_text = reference_text.strip()

    # Try to find the reference in context
    lines = context.split("\n")
    for i, line in enumerate(lines):
        # Remove line numbers from the line (format: "  123: content")
        line_content = re.sub(r"^\s*\d+:\s*", "", line)

        # Check if line contains the reference text
        if search_text.lower() in line_content.lower():
            # Search nearby lines (±10 lines) for :label: directive
            start = max(0, i - 10)
            end = min(len(lines), i + 10)
            context_window = "\n".join(lines[start:end])

            # Look for :label: directives matching expected prefixes
            for prefix in prefixes:
                label_pattern = rf":label:\s+({prefix}-[a-z0-9-_]+)"
                match = re.search(label_pattern, context_window)
                if match:
                    return match.group(1)

    # Strategy 2: Generate standardized label from text
    # Use sanitize_label to ensure correct format

    if reference_type == "theorem":
        # Extract theorem number/type from text
        # "Theorem 1.4" → "thm-1-4"
        # "Lemma 2.3" → "lem-2-3"
        thm_match = re.match(
            r"(Theorem|Lemma|Proposition|Corollary)\s+([\d.]+)", reference_text, re.IGNORECASE
        )
        if thm_match:
            thm_type = thm_match.group(1).lower()
            thm_num = thm_match.group(2).replace(".", "-")
            prefix = {
                "theorem": "thm",
                "lemma": "lem",
                "proposition": "prop",
                "corollary": "cor",
            }[thm_type]
            return f"{prefix}-{thm_num}"

    elif reference_type == "definition":
        # "Lipschitz continuous" → "def-lipschitz-continuous"
        # Use sanitize_label to normalize
        term = sanitize_label(reference_text)
        if not term.startswith("def-"):
            return f"def-{term}"
        return term

    # Fallback: generic sanitization
    label = sanitize_label(reference_text)
    if not any(label.startswith(p + "-") for p in prefixes):
        label = f"{prefixes[0]}-{label}"
    return label
