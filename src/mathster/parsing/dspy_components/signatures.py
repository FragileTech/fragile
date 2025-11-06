"""
DSPy Signature definitions for mathematical entity extraction and improvement.

Defines the input/output specifications for DSPy ReAct agents that extract
mathematical entities from markdown documents or improve existing extractions.
"""

import dspy

from mathster.parsing.models.entities import ChapterExtraction


class ExtractMathematicalConcepts(dspy.Signature):
    """
    Extract all mathematical entities from a numbered markdown chapter.

    You are an expert mathematical document parser. Your task is to identify and extract
    ALL mathematical entities from the provided chapter text. The text has line numbers
    in the format "NNN: content" which you must use to identify precise boundaries.

    See extract_workflow.py lines 631-761 for full documentation.
    """

    chapter_with_lines: str = dspy.InputField(
        desc="Chapter text with line numbers in format 'NNN: content'"
    )
    chapter_number: int = dspy.InputField(desc="Chapter number (0 for preamble, 1+ for sections)")

    extraction: ChapterExtraction = dspy.OutputField(
        desc="All mathematical entities found with precise line ranges"
    )


class ExtractWithValidation(dspy.Signature):
    """
    Extract mathematical entities with built-in validation feedback.

    This signature extends ExtractMathematicalConcepts by providing validation
    feedback to guide the agent's extraction process.
    """

    chapter_with_lines: str = dspy.InputField(desc="Chapter text with line numbers")
    chapter_number: int = dspy.InputField(desc="Chapter number")
    previous_error_report: str = dspy.InputField(
        default="", desc="Error report from previous attempt (empty on first attempt)"
    )

    extraction: ChapterExtraction = dspy.OutputField(
        desc="Validated mathematical entities extraction"
    )


class ExtractSingleLabel(dspy.Signature):
    """
    Extract a single specific mathematical entity by label.

    Used for targeted extraction when only one entity needs to be extracted
    or re-extracted (e.g., during improvement workflow).
    """

    chapter_with_lines: str = dspy.InputField(desc="Chapter text with line numbers")
    target_label: str = dspy.InputField(
        desc="Specific entity label to extract (e.g., 'def-lipschitz')"
    )
    entity_type: str = dspy.InputField(desc="Entity type (definitions, theorems, proofs, etc.)")
    previous_error_report: str = dspy.InputField(
        default="", desc="Error report from previous attempt"
    )

    extraction: ChapterExtraction = dspy.OutputField(
        desc="Chapter extraction containing the single target entity"
    )


class ImproveMathematicalConcepts(dspy.Signature):
    """
    Improve existing extraction by finding missing entities.

    Compares existing extraction with source text to identify and extract
    any mathematical entities that were missed in the original extraction.
    """

    chapter_with_lines: str = dspy.InputField(desc="Chapter text with line numbers")
    existing_extraction: str = dspy.InputField(desc="JSON string of current extraction")
    missed_labels: str = dspy.InputField(desc="Comma-separated list of labels that were missed")
    previous_error_report: str = dspy.InputField(
        default="", desc="Error report from previous attempt"
    )

    improved_extraction: ChapterExtraction = dspy.OutputField(
        desc="Updated extraction including previously missed entities"
    )


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
