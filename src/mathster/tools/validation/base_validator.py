"""Base validator classes and utilities for entity validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError as PydanticValidationError


@dataclass
class ValidationError:
    """Represents a validation error that must be fixed."""

    file: str
    field: str | None
    message: str
    severity: str = "error"  # error, critical

    def __str__(self) -> str:
        """Format error message."""
        if self.field:
            return f"[{self.severity.upper()}] {self.file}::{self.field}: {self.message}"
        return f"[{self.severity.upper()}] {self.file}: {self.message}"


@dataclass
class ValidationWarning:
    """Represents a validation warning (non-blocking issue)."""

    file: str
    field: str | None
    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        """Format warning message."""
        msg = f"[WARNING] {self.file}"
        if self.field:
            msg += f"::{self.field}"
        msg += f": {self.message}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Result of validation for a single entity or collection."""

    is_valid: bool
    entity_count: int = 0
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Count of validation errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Count of validation warnings."""
        return len(self.warnings)

    @property
    def critical_error_count(self) -> int:
        """Count of critical validation errors."""
        return sum(1 for e in self.errors if e.severity == "critical")

    def add_error(
        self, file: str, message: str, field: str | None = None, severity: str = "error"
    ) -> None:
        """Add validation error."""
        self.errors.append(
            ValidationError(file=file, field=field, message=message, severity=severity)
        )
        self.is_valid = False

    def add_warning(
        self, file: str, message: str, field: str | None = None, suggestion: str | None = None
    ) -> None:
        """Add validation warning."""
        self.warnings.append(
            ValidationWarning(file=file, field=field, message=message, suggestion=suggestion)
        )

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.entity_count += other.entity_count
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
        self.metadata.update(other.metadata)

    def summary(self) -> str:
        """Generate summary string."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines = [
            f"Validation Result: {status}",
            f"  Entities: {self.entity_count}",
            f"  Errors: {self.error_count}",
            f"  Warnings: {self.warning_count}",
        ]
        if self.critical_error_count > 0:
            lines.append(f"  Critical Errors: {self.critical_error_count}")
        return "\n".join(lines)


class BaseValidator(ABC):
    """Abstract base class for all entity validators."""

    def __init__(self, strict: bool = False):
        """Initialize validator.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict

    @abstractmethod
    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate a single entity.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult with errors and warnings
        """

    def validate_file(self, file_path: Path) -> ValidationResult:
        """Load and validate entity from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            ValidationResult
        """
        try:
            with open(file_path) as f:
                data = json.load(f)
            return self.validate_entity(data, file_path)
        except json.JSONDecodeError as e:
            result = ValidationResult(is_valid=False, entity_count=0)
            result.add_error(
                file=file_path.name, message=f"Invalid JSON: {e}", severity="critical"
            )
            return result
        except Exception as e:
            result = ValidationResult(is_valid=False, entity_count=0)
            result.add_error(
                file=file_path.name, message=f"Unexpected error: {e}", severity="critical"
            )
            return result

    def validate_directory(self, directory: Path, pattern: str = "*.json") -> ValidationResult:
        """Validate all matching files in directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files

        Returns:
            Aggregated ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=0)

        if not directory.exists():
            result.add_error(
                file=str(directory), message="Directory does not exist", severity="critical"
            )
            return result

        files = list(directory.glob(pattern))
        if not files:
            result.add_warning(
                file=str(directory),
                message=f"No files matching pattern '{pattern}'",
                suggestion="Check directory path and pattern",
            )
            return result

        for file_path in files:
            file_result = self.validate_file(file_path)
            result.merge(file_result)

        return result

    def validate_required_fields(
        self, data: dict, required_fields: list[str], file_name: str, result: ValidationResult
    ) -> None:
        """Validate that required fields exist and are non-empty.

        Args:
            data: Entity data dictionary
            required_fields: List of required field names
            file_name: Name of file being validated
            result: ValidationResult to populate
        """
        for field in required_fields:
            if field not in data:
                result.add_error(file=file_name, field=field, message="Required field missing")
            elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
                result.add_error(file=file_name, field=field, message="Required field is empty")

    def validate_pydantic_schema(
        self, data: dict, model_class: type[BaseModel], file_name: str, result: ValidationResult
    ) -> BaseModel | None:
        """Validate data against Pydantic schema.

        Args:
            data: Entity data dictionary
            model_class: Pydantic model class
            file_name: Name of file being validated
            result: ValidationResult to populate

        Returns:
            Validated Pydantic model instance, or None if validation fails
        """
        try:
            return model_class.model_validate(data)
        except PydanticValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                result.add_error(
                    file=file_name, field=field_path, message=error["msg"], severity="error"
                )
            return None
        except Exception as e:
            result.add_error(
                file=file_name, message=f"Schema validation failed: {e}", severity="critical"
            )
            return None

    def validate_label_format(
        self, label: str, expected_prefix: str, file_name: str, result: ValidationResult
    ) -> None:
        """Validate label follows expected format.

        Args:
            label: Entity label
            expected_prefix: Expected label prefix (e.g., 'thm-', 'lem-', 'obj-')
            file_name: Name of file being validated
            result: ValidationResult to populate
        """
        if not label:
            result.add_error(file=file_name, field="label", message="Label is empty")
            return

        if not label.startswith(expected_prefix):
            result.add_warning(
                file=file_name,
                field="label",
                message=f"Label '{label}' does not start with expected prefix '{expected_prefix}'",
                suggestion=f"Consider renaming to '{expected_prefix}{label}'",
            )

        # Check kebab-case format
        if not all(c.islower() or c.isdigit() or c == "-" for c in label):
            result.add_warning(
                file=file_name,
                field="label",
                message=f"Label '{label}' not in kebab-case format",
                suggestion="Use lowercase letters, numbers, and hyphens only",
            )
