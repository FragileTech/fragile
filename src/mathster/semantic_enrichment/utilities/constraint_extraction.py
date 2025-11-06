"""
Constraint extraction utilities.

Extracts mathematical constraints from parameter definitions and meanings using regex patterns.
"""

import re


def extract_constraints_from_text(text: str, symbol: str) -> list[str]:
    """
    Extract mathematical constraints for a parameter from text.

    Searches for patterns like:
    - "γ > 0"
    - "N ≥ 2"
    - "τ ∈ (0, 1)"
    - "0 < h < 1"

    Args:
        text: Text containing parameter definition/usage
        symbol: Parameter symbol to find constraints for

    Returns:
        List of constraint strings

    Examples:
        >>> extract_constraints_from_text("friction coefficient γ > 0", "γ")
        ["γ > 0"]
        >>> extract_constraints_from_text("number of walkers N ≥ 1", "N")
        ["N ≥ 1"]
    """
    constraints = []

    # Escape special regex characters in symbol
    symbol_escaped = re.escape(symbol)

    # Pattern 1: "symbol > value" or "symbol >= value" etc.
    # Matches: γ > 0, N ≥ 1, τ < 1, h ≤ 0.1
    pattern1 = rf"{symbol_escaped}\s*([>≥<≤]=?)\s*([-+]?\d+(?:\.\d+)?)"
    for match in re.finditer(pattern1, text):
        operator = match.group(1)
        value = match.group(2)
        constraint = f"{symbol} {operator} {value}"
        constraints.append(constraint)

    # Pattern 2: "value < symbol < value" (range)
    # Matches: 0 < γ < 1, -1 ≤ β ≤ 1
    pattern2 = rf"([-+]?\d+(?:\.\d+)?)\s*([<≤])\s*{symbol_escaped}\s*([<≤])\s*([-+]?\d+(?:\.\d+)?)"
    for match in re.finditer(pattern2, text):
        lower = match.group(1)
        op1 = match.group(2)
        op2 = match.group(3)
        upper = match.group(4)
        constraint = f"{lower} {op1} {symbol} {op2} {upper}"
        constraints.append(constraint)

    # Pattern 3: "symbol ∈ set" or "symbol in set"
    # Matches: γ ∈ ℝ⁺, N ∈ ℕ, x in X
    pattern3 = rf"{symbol_escaped}\s*(?:∈|in)\s*([^\s,\.]+)"
    for match in re.finditer(pattern3, text):
        set_name = match.group(1)
        constraint = f"{symbol} ∈ {set_name}"
        constraints.append(constraint)

    # Pattern 4: Prose descriptions converted to constraints
    # "positive" → "> 0", "non-negative" → "≥ 0"
    if "positive" in text.lower() and symbol in text:
        constraints.append(f"{symbol} > 0")
    elif "non-negative" in text.lower() and symbol in text:
        constraints.append(f"{symbol} ≥ 0")
    elif "negative" in text.lower() and symbol in text:
        constraints.append(f"{symbol} < 0")

    # Deduplicate
    return list(set(constraints))


def infer_domain_from_constraints(constraints: list[str], symbol: str) -> str | None:
    """
    Infer parameter domain from constraints using pattern matching.

    Args:
        constraints: List of constraint strings
        symbol: Parameter symbol

    Returns:
        Inferred domain string or None

    Examples:
        >>> infer_domain_from_constraints(["N ≥ 1"], "N")
        "natural"
        >>> infer_domain_from_constraints(["γ > 0"], "γ")
        "positive_real"
    """
    if not constraints:
        return None

    constraint_text = " ".join(constraints).lower()

    # Natural numbers: N ≥ 1, N > 0, N ∈ ℕ
    if any(
        pattern in constraint_text
        for pattern in ["≥ 1", "> 0", "∈ ℕ", "in ℕ", "natural", "integer ≥"]
    ):
        return "natural"

    # Positive real: γ > 0, τ ∈ ℝ⁺, "positive"
    if any(pattern in constraint_text for pattern in ["> 0", "∈ ℝ⁺", "positive"]):
        return "positive_real"

    # Real numbers: γ ∈ ℝ, "real"
    if any(pattern in constraint_text for pattern in ["∈ ℝ", "in ℝ", "real"]):
        return "real"

    # Integer: n ∈ ℤ
    if any(pattern in constraint_text for pattern in ["∈ ℤ", "in ℤ", "integer"]):
        return "integer"

    # Rational: q ∈ ℚ
    if any(pattern in constraint_text for pattern in ["∈ ℚ", "rational"]):
        return "rational"

    # Boolean: true/false
    if any(pattern in constraint_text for pattern in ["true", "false", "boolean"]):
        return "boolean"

    return None
