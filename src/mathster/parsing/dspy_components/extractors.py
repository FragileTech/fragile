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
