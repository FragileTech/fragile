"""
DSPy Module implementations for mathematical entity extraction.

Provides ReAct-based extraction agents that use tools for validation
and self-correction during the extraction process.
"""

import dspy

from mathster.parsing.dspy_components.signatures import (
    ExtractMathematicalConcepts,
    ExtractSingleLabel,
    ExtractWithValidation,
)
from mathster.parsing.dspy_components.tools import (
    validate_extraction_tool,
    validate_single_entity_tool,
)
from mathster.parsing.text_processing import analyze_labels_in_chapter


class MathematicalConceptExtractor(dspy.Module):
    """
    ReAct agent for extracting mathematical entities from markdown.

    Uses tools for validation and comparison to self-correct during extraction.
    Implements retry logic with validation feedback.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.ReAct(
            signature=ExtractMathematicalConcepts,
            tools=[
                analyze_labels_in_chapter,
                validate_extraction_tool,
                validate_single_entity_tool,
            ],
        )

    def forward(self, chapter_with_lines: str, chapter_number: int):
        """
        Extract mathematical entities from a chapter.

        Args:
            chapter_with_lines: Numbered chapter text
            chapter_number: Chapter index

        Returns:
            ChapterExtraction with all found entities
        """
        return self.prog(chapter_with_lines=chapter_with_lines, chapter_number=chapter_number)


class MathematicalConceptExtractorWithValidation(dspy.Module):
    """
    Enhanced extractor with built-in validation feedback loop.

    Provides previous error reports to help the agent self-correct.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.ReAct(
            ExtractWithValidation,
            tools=[
                analyze_labels_in_chapter,
                validate_extraction_tool,
                validate_single_entity_tool,
            ],
        )

    def forward(
        self, chapter_with_lines: str, chapter_number: int, previous_error_report: str = ""
    ):
        """
        Extract with validation feedback.

        Args:
            chapter_with_lines: Numbered chapter text
            chapter_number: Chapter index
            previous_error_report: Errors from previous attempt

        Returns:
            ChapterExtraction with validated entities
        """
        return self.prog(
            chapter_with_lines=chapter_with_lines,
            chapter_number=chapter_number,
            previous_error_report=previous_error_report,
        )


class SingleLabelExtractor(dspy.Module):
    """
    ReAct agent for extracting a single specific entity.

    Used during improvement workflow or targeted re-extraction.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.ReAct(
            ExtractSingleLabel,
            tools=[
                validate_single_entity_tool,
            ],
        )

    def forward(
        self,
        chapter_with_lines: str,
        target_label: str,
        entity_type: str,
        previous_error_report: str = "",
    ):
        """
        Extract a single entity by label.

        Args:
            chapter_with_lines: Numbered chapter text
            target_label: Specific entity label to extract
            entity_type: Entity type (definitions, theorems, etc.)
            previous_error_report: Errors from previous attempt

        Returns:
            ChapterExtraction containing the single target entity
        """
        return self.prog(
            chapter_with_lines=chapter_with_lines,
            target_label=target_label,
            entity_type=entity_type,
            previous_error_report=previous_error_report,
        )


class ParameterExtractor(dspy.Module):
    """
    ReAct agent for extracting parameter definitions from chapter text.

    Unlike other mathematical entities (definitions, theorems, etc.), parameters
    don't have their own Jupyter Book directives. This agent extracts parameters
    by analyzing parameters_mentioned fields and finding their declarations in text.

    The agent uses multiple search patterns to locate parameter definitions:
    - "Let X be ..." declarations
    - "X denotes ..." statements
    - Algorithm parameter lists
    - Notation tables

    Uses validation tools for self-correction during extraction.
    """

    def __init__(self):
        super().__init__()
        from mathster.parsing.dspy_components.signatures import ExtractParameters
        from mathster.parsing.dspy_components.tools import validate_parameter_tool

        self.prog = dspy.ReAct(
            signature=ExtractParameters,
            tools=[validate_parameter_tool],
        )

    def forward(
        self,
        chapter_with_lines: str,
        parameters_mentioned: list[str],
        parameter_declarations: dict,
        file_path: str,
        article_id: str,
        previous_error_report: str = "",
    ):
        """
        Extract parameter definitions from chapter.

        Args:
            chapter_with_lines: Numbered chapter text
            parameters_mentioned: List of parameter symbols from definitions/theorems
            parameter_declarations: Dict mapping symbols to their declaration locations
            file_path: Path to source markdown file
            article_id: Article identifier
            previous_error_report: Errors from previous attempt

        Returns:
            List of ParameterExtraction objects
        """
        import json

        # Convert inputs to strings for DSPy
        parameters_str = ",".join(parameters_mentioned)
        declarations_str = json.dumps(parameter_declarations)

        result = self.prog(
            chapter_with_lines=chapter_with_lines,
            parameters_mentioned=parameters_str,
            parameter_declarations=declarations_str,
            file_path=file_path,
            article_id=article_id,
            previous_error_report=previous_error_report,
        )

        # Return the parameters list from result
        return result.parameters if hasattr(result, "parameters") else []
