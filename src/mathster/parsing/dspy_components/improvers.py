"""
DSPy Module implementations for mathematical entity improvement.

Provides ReAct-based agents that enhance existing extractions by finding
missing entities and correcting errors.
"""

import dspy

from mathster.parsing.dspy_components.signatures import ImproveMathematicalConcepts


class MathematicalConceptImprover(dspy.Module):
    """
    ReAct agent for improving existing mathematical entity extractions.

    Compares existing extraction with source text to find missed entities
    and extract them. Uses validation tools to ensure quality.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.ReAct(ImproveMathematicalConcepts)

    def forward(
        self,
        chapter_with_lines: str,
        existing_extraction: str,
        missed_labels: str,
        previous_error_report: str = ""
    ):
        """
        Improve existing extraction by adding missed entities.

        Args:
            chapter_with_lines: Numbered chapter text
            existing_extraction: JSON string of current extraction
            missed_labels: Comma-separated list of missed labels
            previous_error_report: Errors from previous attempt

        Returns:
            Improved ChapterExtraction including previously missed entities
        """
        return self.prog(
            chapter_with_lines=chapter_with_lines,
            existing_extraction=existing_extraction,
            missed_labels=missed_labels,
            previous_error_report=previous_error_report
        )
