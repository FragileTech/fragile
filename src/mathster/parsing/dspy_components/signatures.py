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
