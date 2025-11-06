"""
Parameter enrichment: RawParameter → ParameterBox transformation.

Transforms raw parameter extractions into fully enriched ParameterBox instances
with domain inference, constraint extraction, and cross-reference tracking.
"""

import logging
from typing import Any

from mathster.core.enriched_data import ParameterBox, ParameterScope
from mathster.core.raw_data import RawParameter
from mathster.proof_pipeline.math_types import ParameterType
from mathster.semantic_enrichment.utilities import (
    extract_chapter_from_source,
    extract_constraints_from_text,
    extract_document_from_source,
    infer_domain_from_constraints,
    map_parameter_scope,
)

logger = logging.getLogger(__name__)


def generate_latex_from_symbol(symbol: str) -> str:
    """
    Generate LaTeX representation from parameter symbol.

    Args:
        symbol: Parameter symbol (e.g., "tau", "gamma_fric", "N")

    Returns:
        LaTeX string (e.g., "\\tau", "\\gamma_{\\mathrm{fric}}", "N")
    """
    # Greek letter mappings
    greek_map = {
        "alpha": "\\alpha",
        "beta": "\\beta",
        "gamma": "\\gamma",
        "delta": "\\delta",
        "epsilon": "\\epsilon",
        "zeta": "\\zeta",
        "eta": "\\eta",
        "theta": "\\theta",
        "lambda": "\\lambda",
        "mu": "\\mu",
        "nu": "\\nu",
        "rho": "\\rho",
        "sigma": "\\sigma",
        "tau": "\\tau",
        "phi": "\\phi",
        "psi": "\\psi",
        "omega": "\\omega",
    }

    # Handle subscripted symbols (e.g., "gamma_fric")
    if "_" in symbol:
        parts = symbol.split("_", 1)
        base = parts[0].lower()
        subscript = parts[1]

        if base in greek_map:
            # Greek letter with subscript
            return f"{greek_map[base]}_{{\\mathrm{{{subscript}}}}}"
        else:
            # Regular symbol with subscript
            return f"{parts[0]}_{{{subscript}}}"

    # Simple Greek letter
    if symbol.lower() in greek_map:
        return greek_map[symbol.lower()]

    # Already LaTeX or non-Greek symbol
    if symbol.startswith("\\"):
        return symbol

    # Regular symbol (N, C, etc.)
    return symbol


def infer_parameter_type(constraints: list[str], meaning: str, symbol: str) -> ParameterType:
    """
    Infer ParameterType from constraints and meaning.

    Uses pattern matching on constraints. Falls back to semantic analysis of meaning.

    Args:
        constraints: List of mathematical constraints
        meaning: Parameter meaning/description
        symbol: Parameter symbol

    Returns:
        Inferred ParameterType
    """
    # Try automated inference from constraints
    inferred_domain = infer_domain_from_constraints(constraints, symbol)

    domain_map = {
        "natural": ParameterType.NATURAL,
        "integer": ParameterType.INTEGER,
        "real": ParameterType.REAL,
        "positive_real": ParameterType.REAL,  # Map to REAL with constraint
        "rational": ParameterType.RATIONAL,
        "complex": ParameterType.COMPLEX,
        "boolean": ParameterType.BOOLEAN,
    }

    if inferred_domain and inferred_domain in domain_map:
        return domain_map[inferred_domain]

    # Fallback: Analyze meaning text
    meaning_lower = meaning.lower()

    if any(word in meaning_lower for word in ["number of", "count", "index"]):
        return ParameterType.NATURAL

    if any(
        word in meaning_lower
        for word in ["coefficient", "rate", "weight", "parameter", "time", "size"]
    ):
        return ParameterType.REAL

    if any(word in meaning_lower for word in ["boolean", "flag", "indicator"]):
        return ParameterType.BOOLEAN

    # Ultimate fallback: SYMBOLIC (unknown domain)
    return ParameterType.SYMBOLIC


def enrich_parameter(
    raw_param: dict,
    parameter_usage_index: dict[str, list[str]] | None = None,
) -> dict:
    """
    Enrich a RawParameter into ParameterBox format.

    Performs automated enrichment:
    - Scope enum mapping
    - LaTeX generation
    - Constraint extraction
    - Domain inference
    - Cross-reference linking

    Args:
        raw_param: RawParameter as dict
        parameter_usage_index: Optional index mapping symbols to entity labels

    Returns:
        ParameterBox data as dict (ready for Pydantic validation)
    """
    symbol = raw_param.get("symbol", "")
    meaning = raw_param.get("meaning", "")
    full_text = raw_param.get("full_text", "")
    scope_str = raw_param.get("scope", "global")
    source = raw_param.get("source", {})

    # Extract constraints
    text_for_constraints = full_text if full_text else meaning
    constraints = extract_constraints_from_text(text_for_constraints, symbol)

    # Infer domain
    domain = infer_parameter_type(constraints, meaning, symbol)

    # Generate LaTeX
    latex = generate_latex_from_symbol(symbol)

    # Map scope to enum
    try:
        scope = map_parameter_scope(scope_str)
    except ValueError as e:
        logger.warning(f"Invalid scope '{scope_str}', defaulting to LOCAL: {e}")
        scope = ParameterScope.LOCAL

    # Extract chapter and document
    chapter = extract_chapter_from_source(source)
    document = extract_document_from_source(source)

    # Get cross-references
    appears_in = []
    if parameter_usage_index and symbol in parameter_usage_index:
        appears_in = parameter_usage_index[symbol]

    # Build enriched parameter
    enriched = {
        "label": raw_param.get("label"),
        "symbol": symbol,
        "latex": latex,
        "domain": domain,
        "meaning": meaning,
        "scope": scope,
        "constraints": constraints,
        "default_value": None,  # Could extract with DSPy if needed
        "full_definition_text": full_text if full_text else None,
        "appears_in": appears_in,
        "source": source,
        "validation_errors": [],
        "raw_fallback": raw_param,
    }

    return enriched
