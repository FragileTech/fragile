"""
Cross-reference analysis utilities.

Builds indices and graphs for entity relationships and parameter usage tracking.
"""

from collections import defaultdict


def build_parameter_usage_index(enriched_data: dict) -> dict[str, list[str]]:
    """
    Build index of which entities use which parameters.

    Scans all entities for parameters_mentioned or input_parameters fields
    and creates reverse mapping: parameter_symbol → [entity_labels].

    Args:
        enriched_data: Enriched chapter data with all entity types

    Returns:
        Dict mapping parameter symbols to lists of entity labels that use them

    Example:
        >>> data = {
        ...     "definitions": [{"label": "def-test", "parameters_mentioned": ["alpha", "beta"]}],
        ...     "theorems": [{"label": "thm-main", "input_parameters": ["alpha"]}]
        ... }
        >>> index = build_parameter_usage_index(data)
        >>> index["alpha"]
        ["def-test", "thm-main"]
    """
    usage_index = defaultdict(list)

    # Scan definitions
    for defn in enriched_data.get("definitions", []):
        label = defn.get("label")
        for param in defn.get("parameters_mentioned", []):
            if label:
                usage_index[param].append(label)

    # Scan theorems
    for thm in enriched_data.get("theorems", []):
        label = thm.get("label")
        # Check both parameters_mentioned and input_parameters
        params = thm.get("parameters_mentioned", []) + thm.get("input_parameters", [])
        for param in params:
            if label and param not in usage_index[param]:
                usage_index[param].append(label)

    # Scan axioms
    for axiom in enriched_data.get("axioms", []):
        label = axiom.get("label")
        for param in axiom.get("parameters_text", []):
            # parameters_text is list of strings, extract symbols
            # For now, just use input_parameters if available
            if label:
                pass  # TODO: Parse parameters_text for symbols

    # Scan proofs (might mention parameters)
    for proof in enriched_data.get("proofs", []):
        label = proof.get("label")
        for param in proof.get("parameters_mentioned", []):
            if label:
                usage_index[param].append(label)

    return dict(usage_index)
