"""
Result models for validation and improvement workflows.

Defines Pydantic models for tracking the results of validation and improvement
operations, including change tracking and metadata.
"""

from pydantic import BaseModel, Field

from mathster.parsing.models.changes import ChangeOperation, EntityChange


class ValidationResult(BaseModel):
    """Result of validating an extraction attempt."""

    is_valid: bool = Field(..., description="Whether the extraction is valid")
    errors: list[str] = Field(default_factory=list, description="List of validation errors")
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    entities_validated: dict[str, int] = Field(
        default_factory=dict,
        description="Count of successfully validated entities by type"
    )

    def get_feedback(self) -> str:
        """Get human-readable feedback for the agent."""
        if self.is_valid:
            return (
                f"✓ Validation successful! Entities validated: {self.entities_validated}. "
                "All entities have valid labels, line numbers, and required fields."
            )

        feedback = "✗ Validation failed. Please fix the following errors:\n"
        for i, error in enumerate(self.errors, 1):
            feedback += f"{i}. {error}\n"

        if self.warnings:
            feedback += "\nWarnings:\n"
            for i, warning in enumerate(self.warnings, 1):
                feedback += f"{i}. {warning}\n"

        return feedback


class ImprovementResult(BaseModel):
    """Result of improvement workflow with change tracking."""

    changes: list[EntityChange] = Field(default_factory=list)
    entities_added: int = Field(default=0, description="Count of entities added")
    entities_modified: int = Field(default=0, description="Count of entities modified")
    entities_deleted: int = Field(default=0, description="Count of entities deleted")
    entities_unchanged: int = Field(default=0, description="Count of entities unchanged")

    def add_change(self, change: EntityChange) -> None:
        """Add a change and update counters."""
        self.changes.append(change)

        if change.operation == ChangeOperation.ADD:
            self.entities_added += 1
        elif change.operation == ChangeOperation.MODIFY:
            self.entities_modified += 1
        elif change.operation == ChangeOperation.DELETE:
            self.entities_deleted += 1
        elif change.operation == ChangeOperation.NO_CHANGE:
            self.entities_unchanged += 1

    def get_summary(self) -> str:
        """Get human-readable summary of changes."""
        return (
            f"Improvement Summary:\n"
            f"  Added: {self.entities_added}\n"
            f"  Modified: {self.entities_modified}\n"
            f"  Deleted: {self.entities_deleted}\n"
            f"  Unchanged: {self.entities_unchanged}\n"
            f"  Total changes: {len(self.changes)}"
        )
