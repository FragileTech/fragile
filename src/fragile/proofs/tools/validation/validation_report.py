"""Validation report generation utilities."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional

from fragile.proofs.tools.validation.base_validator import ValidationResult


class ValidationReport:
    """Generates comprehensive validation reports in multiple formats."""

    def __init__(self, result: ValidationResult, refined_dir: Path, mode: str = "complete"):
        """Initialize validation report.

        Args:
            result: ValidationResult to report on
            refined_dir: Path to refined_data directory
            mode: Validation mode ('schema', 'complete', etc.)
        """
        self.result = result
        self.refined_dir = refined_dir
        self.mode = mode
        self.timestamp = datetime.now()

    def generate_markdown(self) -> str:
        """Generate markdown-formatted validation report.

        Returns:
            Markdown string
        """
        lines = [
            "# Validation Report",
            "",
            f"**Generated**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Directory**: `{self.refined_dir}`",
            f"**Mode**: `{self.mode}`",
            "",
            "---",
            "",
            "## Summary",
            "",
            self._generate_summary_table(),
            "",
        ]

        # Errors section
        if self.result.errors:
            lines.extend(self._generate_errors_section())
            lines.append("")

        # Warnings section
        if self.result.warnings:
            lines.extend(self._generate_warnings_section())
            lines.append("")

        # Metadata section
        if self.result.metadata:
            lines.extend(self._generate_metadata_section())
            lines.append("")

        # Recommendations
        lines.extend(self._generate_recommendations())

        return "\n".join(lines)

    def _generate_summary_table(self) -> str:
        """Generate summary statistics table."""
        status_emoji = "✅" if self.result.is_valid else "❌"
        status_text = "VALID" if self.result.is_valid else "INVALID"

        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Status** | {status_emoji} **{status_text}** |",
            f"| **Entities Validated** | {self.result.entity_count} |",
            f"| **Errors** | {self.result.error_count} |",
            f"| **Warnings** | {self.result.warning_count} |",
        ]

        if self.result.critical_error_count > 0:
            lines.append(f"| **Critical Errors** | ⚠️ {self.result.critical_error_count} |")

        return "\n".join(lines)

    def _generate_errors_section(self) -> list[str]:
        """Generate errors section."""
        lines = [
            "## Validation Errors",
            "",
            f"Found **{self.result.error_count}** validation errors that must be fixed:",
            "",
        ]

        # Group errors by file
        errors_by_file: dict[str, list] = {}
        for error in self.result.errors:
            if error.file not in errors_by_file:
                errors_by_file[error.file] = []
            errors_by_file[error.file].append(error)

        # Generate error list grouped by file
        for file_name, file_errors in sorted(errors_by_file.items()):
            lines.extend((f"### {file_name}", ""))
            for error in file_errors:
                severity_badge = "🔴" if error.severity == "critical" else "🟠"
                if error.field:
                    lines.append(
                        f"- {severity_badge} **Field**: `{error.field}` - {error.message}"
                    )
                else:
                    lines.append(f"- {severity_badge} {error.message}")
            lines.append("")

        return lines

    def _generate_warnings_section(self) -> list[str]:
        """Generate warnings section."""
        lines = [
            "## Validation Warnings",
            "",
            f"Found **{self.result.warning_count}** validation warnings (non-blocking):",
            "",
        ]

        # Group warnings by file
        warnings_by_file: dict[str, list] = {}
        for warning in self.result.warnings:
            if warning.file not in warnings_by_file:
                warnings_by_file[warning.file] = []
            warnings_by_file[warning.file].append(warning)

        # Generate warning list grouped by file
        for file_name, file_warnings in sorted(warnings_by_file.items()):
            lines.extend((f"### {file_name}", ""))
            for warning in file_warnings:
                if warning.field:
                    lines.append(f"- ⚠️ **Field**: `{warning.field}` - {warning.message}")
                else:
                    lines.append(f"- ⚠️ {warning.message}")
                if warning.suggestion:
                    lines.append(f"  - *Suggestion*: {warning.suggestion}")
            lines.append("")

        return lines

    def _generate_metadata_section(self) -> list[str]:
        """Generate metadata section."""
        lines = [
            "## Validation Metadata",
            "",
        ]

        for key, value in sorted(self.result.metadata.items()):
            lines.append(f"- **{key}**: `{value}`")

        return lines

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on validation results."""
        lines = [
            "---",
            "",
            "## Recommendations",
            "",
        ]

        if self.result.is_valid:
            lines.extend((
                "✅ **All validations passed!** The refined data is ready for the next stage.",
                "",
                "**Next Steps:**",
                "1. Run transformation: `python -m fragile.proofs.tools.enriched_to_math_types`",
                "2. Build registries: Use `registry-management` skill",
                "3. Verify cross-references: Use `relationship_validator`",
            ))
        else:
            lines.extend((
                "❌ **Validation failed.** Please fix the errors above before proceeding.",
                "",
                "**Recommended Actions:**",
            ))

            if self.result.critical_error_count > 0:
                lines.append(
                    f"1. **Fix {self.result.critical_error_count} critical errors first** "
                    "(these prevent loading)"
                )

            if self.result.error_count - self.result.critical_error_count > 0:
                non_critical = self.result.error_count - self.result.critical_error_count
                lines.append(
                    f"2. **Fix {non_critical} validation errors** (these violate schema requirements)"
                )

            if self.result.warning_count > 0:
                lines.append(
                    f"3. **Review {self.result.warning_count} warnings** "
                    "(these suggest improvements)"
                )

            lines.extend((
                "",
                "**Useful Commands:**",
                "```bash",
                "# Find incomplete entities",
                f"python -m fragile.proofs.tools.find_incomplete_entities \\\n"
                f"  --refined-dir {self.refined_dir}",
                "",
                "# Complete partial refinements",
                f"python -m fragile.proofs.tools.complete_refinement \\\n"
                f"  --refined-dir {self.refined_dir}",
                "```",
            ))

        return lines

    def generate_json(self) -> dict[str, Any]:
        """Generate JSON-formatted validation report.

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "refined_dir": str(self.refined_dir),
            "mode": self.mode,
            "summary": {
                "is_valid": self.result.is_valid,
                "entity_count": self.result.entity_count,
                "error_count": self.result.error_count,
                "warning_count": self.result.warning_count,
                "critical_error_count": self.result.critical_error_count,
            },
            "errors": [
                {
                    "file": e.file,
                    "field": e.field,
                    "message": e.message,
                    "severity": e.severity,
                }
                for e in self.result.errors
            ],
            "warnings": [
                {
                    "file": w.file,
                    "field": w.field,
                    "message": w.message,
                    "suggestion": w.suggestion,
                }
                for w in self.result.warnings
            ],
            "metadata": self.result.metadata,
        }

    def save_markdown(self, output_path: Path) -> None:
        """Save markdown report to file.

        Args:
            output_path: Path to output file
        """
        markdown = self.generate_markdown()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown)

    def save_json(self, output_path: Path) -> None:
        """Save JSON report to file.

        Args:
            output_path: Path to output file
        """
        report_data = self.generate_json()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "=" * 70)
        print(self.result.summary())
        print("=" * 70)

        if self.result.errors:
            print(f"\n{self.result.error_count} Errors:")
            for error in self.result.errors[:5]:  # Show first 5
                print(f"  • {error}")
            if self.result.error_count > 5:
                print(f"  ... and {self.result.error_count - 5} more")

        if self.result.warnings:
            print(f"\n{self.result.warning_count} Warnings:")
            for warning in self.result.warnings[:5]:  # Show first 5
                print(f"  • {warning}")
            if self.result.warning_count > 5:
                print(f"  ... and {self.result.warning_count - 5} more")

        print("\n" + "=" * 70 + "\n")
