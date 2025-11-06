"""
DSPy Signature definitions for parameter extraction.

Defines the input/output specifications for DSPy agents that extract
parameter definitions and find parameter line numbers in mathematical documents.
"""

import dspy


class ExtractParameters(dspy.Signature):
    """
    Extract parameter definitions from chapter text.

    Parameters are mathematical symbols/variables that are mentioned in definitions
    and theorems but don't have their own Jupyter Book directives. This signature
    extracts standalone Parameter objects for each symbol.

    The agent should:
    1. Analyze the parameters_mentioned list from existing extraction
    2. Find where each parameter is defined in the chapter text
    3. Extract the parameter's meaning/description from context
    4. Determine if the parameter is global (document-wide) or local (section-specific)
    5. Create ParameterExtraction objects with precise line ranges

    Common parameter declaration patterns:
    - "Let α be the..." or "Let α denote..."
    - "α represents..." or "α denotes..."
    - "Throughout, α is..."
    - Algorithm parameter lists: "Parameters: α, β, γ, ..."
    - Notation tables with Greek letters and their meanings
    """

    chapter_with_lines: str = dspy.InputField(
        desc="Chapter text with line numbers in format 'NNN: content'"
    )

    parameters_mentioned: str = dspy.InputField(
        desc="Comma-separated list of parameter symbols found in definitions/theorems (e.g., 'alpha,beta,tau,N')"
    )

    parameter_declarations: str = dspy.InputField(
        desc="JSON string of parameter declarations found in text with line numbers and context"
    )

    file_path: str = dspy.InputField(desc="Path to source markdown file")

    article_id: str = dspy.InputField(desc="Article identifier (e.g., '01_fragile_gas_framework')")

    previous_error_report: str = dspy.InputField(
        default="", desc="Error report from previous attempt (empty on first attempt)"
    )

    parameters: list = dspy.OutputField(
        desc="List of ParameterExtraction objects with label, symbol, meaning, scope, line ranges"
    )


class FindParameterLineNumber(dspy.Signature):
    """
    Find the precise line number where a parameter is defined or first meaningfully mentioned.

    This signature is used when automated regex patterns fail to find a parameter's definition.
    The agent should search the numbered document text and locate:

    1. **Formal definition** (highest priority):
       - "Let X be..." or "Let X denote..."
       - "X denotes..." or "X represents..."
       - "X := ..." (definition with assignment)
       - "where X is..." or "where X = ..."

    2. **First meaningful mention** (fallback):
       - First occurrence in a displayed formula ($$...$$)
       - First occurrence in algorithm specification
       - First occurrence in theorem statement

    3. **Context clues** to help:
       - Usage context shows where parameter appears in definitions/theorems
       - Symbol variants help match LaTeX, Greek letters, subscripts

    The agent should:
    - Search for the parameter using all variants
    - Prioritize formal definitions over mentions
    - Return precise line numbers from numbered text
    - Explain reasoning and indicate confidence
    - Return line 1 only if truly not found anywhere
    """

    parameter_symbol: str = dspy.InputField(
        desc="Parameter symbol to locate (e.g., 'tau', 'gamma_fric', 'V_alg', 'N')"
    )

    symbol_variants: str = dspy.InputField(
        desc="JSON list of symbol variants to search: LaTeX (\\tau), Greek (τ), subscripted (\\gamma_{\\mathrm{fric}}), etc."
    )

    document_with_lines: str = dspy.InputField(
        desc="Full document text with line numbers in format 'NNN: content' (use these line numbers in output)"
    )

    context_from_entity: str = dspy.InputField(
        desc="Context showing how parameter is used (from definition/theorem that mentions it) - helps understand what to look for"
    )

    line_start: int = dspy.OutputField(
        desc="Starting line number where parameter is defined or first meaningfully mentioned (extract from 'NNN:' prefix)"
    )

    line_end: int = dspy.OutputField(
        desc="Ending line number (usually line_start + 1 or line_start + 2 for multi-line definitions)"
    )

    confidence: str = dspy.OutputField(
        desc="Confidence level: 'high' (formal definition found), 'medium' (clear first mention), 'low' (guess/not found - return line 1)"
    )

    reasoning: str = dspy.OutputField(
        desc="Brief explanation of where parameter was found and why these line numbers were chosen (1-2 sentences)"
    )
