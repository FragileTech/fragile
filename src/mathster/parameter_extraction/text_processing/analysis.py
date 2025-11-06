"""
Parameter declaration finding utilities.

Provides functions for finding parameter declarations in mathematical documents
using regex patterns and symbol variant matching.
"""

import re
from typing import Any


def find_parameter_declarations(
    chapter_text: str,
    parameter_symbols: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Find parameter declaration statements in chapter text.

    Searches for common patterns where parameters are defined:
    - "Let X be ..."
    - "X denotes ..."
    - "Throughout, X represents ..."
    - Algorithm parameter lists: "Parameters: α, β, γ"
    - Notation tables with Greek letters

    Args:
        chapter_text: The chapter text with line numbers
        parameter_symbols: List of parameter symbols to find (e.g., ["alpha", "beta", "tau"])

    Returns:
        Dictionary mapping symbols to their declaration context:
        {
            "alpha": {
                "line_start": 45,
                "line_end": 47,
                "context": "Let α be the exploitation weight...",
                "pattern": "let_be"
            },
            ...
        }
    """
    declarations = {}
    lines = chapter_text.split("\n")

    for symbol in parameter_symbols:
        # Try multiple search patterns
        result = _search_parameter_patterns(symbol, lines)
        if result:
            declarations[symbol] = result

    return declarations


def _search_parameter_patterns(symbol: str, lines: list[str]) -> dict[str, Any] | None:
    """
    Search for a parameter declaration using multiple patterns.

    Args:
        symbol: Parameter symbol to find (e.g., "alpha", "tau", "N")
        lines: Chapter lines (with line numbers)

    Returns:
        Declaration info dict or None if not found
    """
    # Normalize symbol for matching (handle LaTeX, Greek, etc.)
    search_variants = _get_symbol_variants(symbol)

    # Pattern 1: "Let X be ..." or "Let X denote ..."
    for i, line in enumerate(lines):
        for variant in search_variants:
            if re.search(rf"\bLet\s+{re.escape(variant)}\s+(be|denote|represent)", line, re.IGNORECASE):
                context = _extract_context(lines, i, window=3)
                return {
                    "line_start": i + 1,
                    "line_end": min(i + 4, len(lines)),
                    "context": context,
                    "pattern": "let_be",
                }

    # Pattern 2: "X denotes ..." or "X represents ..." or "X := ..." or "X = ..."
    for i, line in enumerate(lines):
        for variant in search_variants:
            if re.search(rf"\b{re.escape(variant)}\s*(denotes?|represents?|:=|=)", line, re.IGNORECASE):
                context = _extract_context(lines, i, window=2)
                return {
                    "line_start": i + 1,
                    "line_end": min(i + 3, len(lines)),
                    "context": context,
                    "pattern": "definition",
                }

    # Pattern 3: "Throughout, X is ..." or "where X is ..."
    for i, line in enumerate(lines):
        for variant in search_variants:
            pattern = rf"\b(Throughout|where)\s.*{re.escape(variant)}\s+(is|are)"
            if re.search(pattern, line, re.IGNORECASE):
                context = _extract_context(lines, i, window=2)
                return {
                    "line_start": i + 1,
                    "line_end": min(i + 3, len(lines)),
                    "context": context,
                    "pattern": "throughout",
                }

    # Pattern 4: Algorithm parameter lists "Parameters: α, β, ..."
    for i, line in enumerate(lines):
        if re.search(r"Parameters?:", line, re.IGNORECASE):
            # Check if our symbol is mentioned in next few lines
            context_lines = lines[i : i + 5]
            context_text = " ".join(context_lines)
            for variant in search_variants:
                if variant in context_text:
                    context = _extract_context(lines, i, window=4)
                    return {
                        "line_start": i + 1,
                        "line_end": min(i + 5, len(lines)),
                        "context": context,
                        "pattern": "parameter_list",
                    }

    # Pattern 5: First mention in LaTeX formula (fallback)
    # Match $...symbol...$ or $$...symbol...$$
    for i, line in enumerate(lines):
        for variant in search_variants:
            # Match LaTeX math mode (single or double dollar signs)
            if re.search(rf'\$[^$]*{re.escape(variant)}[^$]*\$', line):
                context = _extract_context(lines, i, window=1)
                return {
                    "line_start": i + 1,
                    "line_end": min(i + 2, len(lines)),
                    "context": context,
                    "pattern": "first_mention_latex",
                }

    # Pattern 6: First mention in text (last resort fallback)
    for i, line in enumerate(lines):
        for variant in search_variants:
            # Simple substring match
            if variant in line:
                context = _extract_context(lines, i, window=1)
                return {
                    "line_start": i + 1,
                    "line_end": min(i + 2, len(lines)),
                    "context": context,
                    "pattern": "first_mention_text",
                }

    return None


def _get_symbol_variants(symbol: str) -> list[str]:
    """
    Get different variants of a parameter symbol for matching.

    Args:
        symbol: Base symbol (e.g., "alpha", "tau", "N")

    Returns:
        List of variants to search for

    Examples:
        "alpha" → ["alpha", "α", "\\alpha"]
        "tau" → ["tau", "τ", "\\tau"]
        "N" → ["N"]
        "gamma_fric" → ["gamma_fric", "γ_fric", "\\gamma_{\\mathrm{fric}}"]
    """
    variants = [symbol]

    # Greek letter mappings
    greek_map = {
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "epsilon": "ε",
        "zeta": "ζ",
        "eta": "η",
        "theta": "θ",
        "lambda": "λ",
        "mu": "μ",
        "nu": "ν",
        "rho": "ρ",
        "sigma": "σ",
        "tau": "τ",
        "phi": "φ",
        "psi": "ψ",
        "omega": "ω",
    }

    # If symbol is a Greek name, add Greek character
    if symbol.lower() in greek_map:
        variants.append(greek_map[symbol.lower()])
        variants.append(f"\\{symbol.lower()}")

    # If symbol contains underscore (e.g., "gamma_fric"), handle LaTeX variants
    if "_" in symbol:
        parts = symbol.split("_", 1)  # Split only on first underscore
        base = parts[0]
        subscript = parts[1]

        if base.lower() in greek_map:
            # Add multiple LaTeX subscript formats
            variants.append(f"\\{base.lower()}_{{\\mathrm{{{subscript}}}}}")  # \gamma_{\mathrm{fric}}
            variants.append(f"\\{base.lower()}_{{\\text{{{subscript}}}}}")    # \gamma_{\text{fric}}
            variants.append(f"\\{base.lower()}_{{{subscript}}}")              # \gamma_{fric}
            variants.append(f"\\{base.lower()}_{subscript}")                  # \gamma_fric (no braces)

            # Greek character variants
            greek_char = greek_map[base.lower()]
            variants.append(f"{greek_char}_{{\\mathrm{{{subscript}}}}}")     # γ_{\mathrm{fric}}
            variants.append(f"{greek_char}_{{\\text{{{subscript}}}}}")       # γ_{\text{fric}}
            variants.append(f"{greek_char}_{{{subscript}}}")                  # γ_{fric}
            variants.append(f"{greek_char}_{subscript}")                      # γ_fric
        else:
            # Non-Greek subscripted variables (e.g., x_i, v_n)
            variants.append(f"{base}_{{{subscript}}}")                        # x_{i}
            variants.append(f"{base}_{subscript}")                            # x_i

    return variants


def _extract_context(lines: list[str], center_line: int, window: int = 2) -> str:
    """
    Extract context around a line.

    Args:
        lines: All lines in chapter
        center_line: Center line index (0-based)
        window: Number of lines before/after to include

    Returns:
        Context string
    """
    start = max(0, center_line - window)
    end = min(len(lines), center_line + window + 1)

    context_lines = lines[start:end]

    # Remove line numbers if present (format: "NNN: content")
    cleaned_lines = []
    for line in context_lines:
        # Remove line number prefix (e.g., "042: " → "")
        cleaned = re.sub(r"^\d+:\s*", "", line)
        cleaned_lines.append(cleaned)

    return " ".join(cleaned_lines).strip()


def collect_parameters_from_extraction(extraction: dict) -> set[str]:
    """
    Collect all parameter symbols mentioned in an extraction.

    Scans all definitions, theorems, etc. for parameters_mentioned fields.

    Args:
        extraction: ChapterExtraction as dict

    Returns:
        Set of unique parameter symbols

    Example:
        >>> extraction = {"definitions": [{"parameters_mentioned": ["alpha", "beta"]}]}
        >>> collect_parameters_from_extraction(extraction)
        {'alpha', 'beta'}
    """
    parameters = set()

    # Collect from definitions
    for defn in extraction.get("definitions", []):
        params = defn.get("parameters_mentioned", [])
        parameters.update(params)

    # Collect from theorems
    for thm in extraction.get("theorems", []):
        params = thm.get("parameters_mentioned", [])
        parameters.update(params)

    # Collect from proofs (if they have parameters_mentioned)
    for proof in extraction.get("proofs", []):
        params = proof.get("parameters_mentioned", [])
        parameters.update(params)

    # Collect from axioms
    for axiom in extraction.get("axioms", []):
        params = axiom.get("parameters_mentioned", [])
        parameters.update(params)

    return parameters
