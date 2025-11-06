"""
DSPy agent for finding parameter line numbers when regex patterns fail.

This module provides a ChainOfThought agent that can locate parameter definitions
or first mentions in mathematical documents, even when automated regex patterns fail.

Used as Stage 2 fallback in hybrid parameter extraction pipeline.
"""

import dspy

from mathster.parsing.dspy_components.signatures import FindParameterLineNumber


class ParameterLineFinder(dspy.Module):
    """
    DSPy agent to find parameter line numbers when regex fails.

    This agent uses Chain-of-Thought reasoning to search through a numbered document
    and locate where a parameter is defined or first mentioned. It's used as a fallback
    for the ~14% of parameters that automated regex patterns cannot find.

    The agent receives:
    - Parameter symbol (e.g., "V_alg", "gamma_fric")
    - Symbol variants (LaTeX, Greek, subscripted forms)
    - Full document with line numbers
    - Usage context (where parameter appears)

    The agent returns:
    - line_start, line_end: Precise line numbers
    - confidence: "high", "medium", or "low"
    - reasoning: Explanation of what was found

    Only results with "high" or "medium" confidence are accepted.
    Low confidence results are rejected (parameter stays at line 1).
    """

    def __init__(self):
        super().__init__()
        # Use ChainOfThought (not ReAct) - no tools needed, just reasoning
        self.prog = dspy.ChainOfThought(FindParameterLineNumber)

    def forward(
        self,
        parameter_symbol: str,
        symbol_variants: list[str],
        document_with_lines: str,
        context_from_entity: str = "",
    ):
        """
        Find line number for a parameter.

        Args:
            parameter_symbol: Symbol to find (e.g., "tau", "V_alg")
            symbol_variants: List of variants to search
            document_with_lines: Full document with line numbers
            context_from_entity: Where parameter is used (helps agent understand it)

        Returns:
            Dict with line_start, line_end, confidence, reasoning
        """
        import json

        # Convert variants to JSON string for DSPy
        variants_str = json.dumps(symbol_variants)

        # Call agent
        result = self.prog(
            parameter_symbol=parameter_symbol,
            symbol_variants=variants_str,
            document_with_lines=document_with_lines,
            context_from_entity=context_from_entity,
        )

        return {
            "line_start": result.line_start,
            "line_end": result.line_end,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
