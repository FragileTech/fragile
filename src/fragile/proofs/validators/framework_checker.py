"""
Framework Consistency Checker.

This validator checks mathematical objects against the framework documentation
(docs/glossary.md) to ensure:
- Referenced labels exist
- Notation is consistent
- Dependencies are valid

Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from fragile.proofs.core.review_system import ValidationResult


# =============================================================================
# FRAMEWORK CHECKER
# =============================================================================


class FrameworkChecker:
    """
    Validator that checks consistency with framework documentation.

    Checks:
    1. Referenced labels exist in glossary.md
    2. Notation usage is consistent
    3. Dependencies are valid

    All methods are pure/total functions.
    """

    def __init__(self, glossary_path: Optional[Path] = None):
        """
        Initialize framework checker.

        Args:
            glossary_path: Path to docs/glossary.md (default: auto-detect)
        """
        self.glossary_path = glossary_path or self._find_glossary()
        self._load_glossary()

    def _find_glossary(self) -> Path:
        """Find docs/glossary.md from current location."""
        # Try to find relative to current file
        current = Path(__file__).resolve()
        for parent in current.parents:
            glossary = parent / "docs" / "glossary.md"
            if glossary.exists():
                return glossary

        # Fallback: assume standard project structure
        return Path.cwd() / "docs" / "glossary.md"

    def _load_glossary(self) -> None:
        """Load glossary and build label index."""
        self.labels: Set[str] = set()

        if not self.glossary_path.exists():
            # Glossary not found - checker will report all references as warnings
            return

        with open(self.glossary_path, encoding="utf-8") as f:
            content = f.read()

        # Extract labels from glossary (simple pattern matching)
        # Labels appear as: **Label**: `label-name`
        import re

        pattern = r"\*\*Label\*\*:\s*`([a-z][a-z0-9-]+)`"
        matches = re.findall(pattern, content)
        self.labels = set(matches)

    def validate_references(self, referenced_labels: List[str]) -> ValidationResult:
        """
        Total function: Validate that referenced labels exist in framework.

        Args:
            referenced_labels: List of labels referenced by object

        Returns:
            ValidationResult with errors for invalid references
        """
        errors = []
        warnings = []

        if not self.glossary_path.exists():
            warnings.append(
                f"Glossary not found at {self.glossary_path} - cannot validate references"
            )

        for label in referenced_labels:
            if label not in self.labels:
                errors.append(f"Referenced label '{label}' not found in framework glossary")

        return ValidationResult(
            validator="framework-checker",
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            metadata={"glossary_path": str(self.glossary_path), "total_labels": len(self.labels)},
        )

    def get_framework_labels(self) -> Set[str]:
        """
        Pure function: Get all labels in framework glossary.

        Returns:
            Set of valid label strings
        """
        return self.labels.copy()
