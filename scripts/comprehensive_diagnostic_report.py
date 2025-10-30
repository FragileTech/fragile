#!/usr/bin/env python3
"""
Comprehensive Diagnostic Report for Source Locations.

Generates detailed per-file report showing:
- What fields are missing
- What fields are invalid
- Specific issues for each file
- Actionable fixes needed
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys


class FileIssue:
    """Single issue found in a file."""

    def __init__(
        self, field: str, issue_type: str, message: str, current_value=None, suggested_fix=None
    ):
        self.field = field
        self.issue_type = issue_type  # MISSING, INVALID, OUT_OF_BOUNDS
        self.message = message
        self.current_value = current_value
        self.suggested_fix = suggested_fix


class FileDiagnostic:
    """Complete diagnostic for a single file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_name = file_path.name
        self.issues: list[FileIssue] = []
        self.is_valid = True
        self.label = None
        self.entity_type = None

    def add_issue(
        self, field: str, issue_type: str, message: str, current_value=None, suggested_fix=None
    ):
        """Add an issue to this file's diagnostic."""
        self.issues.append(FileIssue(field, issue_type, message, current_value, suggested_fix))
        self.is_valid = False

    def get_severity(self) -> str:
        """Get severity: CRITICAL (no line_range), HIGH (invalid format), MEDIUM (missing optional)."""
        if any(i.field == "line_range" and i.issue_type == "MISSING" for i in self.issues):
            return "CRITICAL"
        if any(i.issue_type in {"INVALID", "OUT_OF_BOUNDS"} for i in self.issues):
            return "HIGH"
        if any(i.issue_type == "MISSING" for i in self.issues):
            return "MEDIUM"
        return "OK"


def diagnose_file(file_path: Path, markdown_path: Path | None = None) -> FileDiagnostic:
    """Perform comprehensive diagnostic on a single file."""
    diagnostic = FileDiagnostic(file_path)

    try:
        with open(file_path, encoding="utf-8") as f:
            entity = json.load(f)
    except Exception as e:
        diagnostic.add_issue("file", "INVALID", f"Cannot read JSON: {e}")
        return diagnostic

    # Store metadata
    diagnostic.label = entity.get("label") or entity.get("label_text")
    diagnostic.entity_type = entity.get("statement_type") or "unknown"

    # Check if source_location exists
    if "source_location" not in entity or entity["source_location"] is None:
        diagnostic.add_issue(
            "source_location", "MISSING", "source_location field missing entirely"
        )
        return diagnostic

    sl = entity["source_location"]

    # Get markdown line count if provided
    markdown_lines = None
    if markdown_path and markdown_path.exists():
        try:
            content = markdown_path.read_text(encoding="utf-8")
            markdown_lines = len(content.splitlines())
        except:
            pass

    # 1. Check document_id
    document_id = sl.get("document_id")
    if not document_id:
        diagnostic.add_issue("document_id", "MISSING", "document_id is missing")
    elif not isinstance(document_id, str):
        diagnostic.add_issue(
            "document_id",
            "INVALID",
            f"Must be string, got {type(document_id).__name__}",
            str(document_id),
        )
    elif not re.match(r"^[0-9]{2}_[a-z_]+$", document_id):
        diagnostic.add_issue(
            "document_id", "INVALID", "Must match pattern [0-9]{2}_[a-z_]+", document_id
        )

    # 2. Check file_path
    file_path_val = sl.get("file_path")
    if not file_path_val:
        diagnostic.add_issue("file_path", "MISSING", "file_path is missing")
    elif not isinstance(file_path_val, str):
        diagnostic.add_issue(
            "file_path",
            "INVALID",
            f"Must be string, got {type(file_path_val).__name__}",
            str(file_path_val),
        )

    # 3. Check section (CRITICAL CHECK)
    section = sl.get("section")
    if section is None or section == "":
        diagnostic.add_issue(
            "section",
            "MISSING",
            "section is missing (REQUIRED)",
            suggested_fix="Extract from line_range if available",
        )
    elif not isinstance(section, str):
        diagnostic.add_issue(
            "section", "INVALID", f"Must be string, got {type(section).__name__}", str(section)
        )
    else:
        # Check for § symbol
        if "§" in section:
            diagnostic.add_issue(
                "section",
                "INVALID",
                "Contains § symbol - must be removed",
                section,
                suggested_fix=f'Remove §: {section.replace("§", "").strip()}',
            )

        # Check for "Section " prefix
        if section.lower().startswith("section "):
            diagnostic.add_issue(
                "section",
                "INVALID",
                'Contains "Section " prefix - must be removed',
                section,
                suggested_fix=f'Remove prefix: {section.lower().replace("section ", "").strip()}',
            )

        # Check for text after number
        if re.search(r"\d+\s+\w+", section):
            # Extract number suggestion
            match = re.match(r"(\d+(?:\.\d+)*)", section)
            suggested = match.group(1) if match else "?"
            diagnostic.add_issue(
                "section",
                "INVALID",
                "Contains text after number",
                section,
                suggested_fix=f"Extract number only: {suggested}",
            )

        # Final format check
        if not re.match(r"^\d+(\.\d+)*$", section):
            diagnostic.add_issue(
                "section",
                "INVALID",
                "Must be X.Y.Z format (only digits and dots)",
                section,
                suggested_fix="Normalize to numeric format",
            )

    # 4. Check line_range (MOST CRITICAL)
    line_range = sl.get("line_range")
    if line_range is None:
        diagnostic.add_issue(
            "line_range",
            "MISSING",
            "line_range is MISSING (REQUIRED)",
            suggested_fix="Use directive matching or text search",
        )
    elif not isinstance(line_range, list):
        diagnostic.add_issue(
            "line_range",
            "INVALID",
            f"Must be list, got {type(line_range).__name__}",
            str(line_range),
            suggested_fix="Convert to [start, end] list format",
        )
    elif len(line_range) != 2:
        diagnostic.add_issue(
            "line_range",
            "INVALID",
            f"Must have exactly 2 elements, got {len(line_range)}",
            str(line_range),
            suggested_fix="Format as [start_line, end_line]",
        )
    # Check element types
    elif not all(isinstance(x, int) for x in line_range):
        types = [type(x).__name__ for x in line_range]
        diagnostic.add_issue(
            "line_range",
            "INVALID",
            f"Elements must be integers, got {types}",
            str(line_range),
            suggested_fix=f'Convert to integers: [{int(line_range[0]) if line_range[0] else "?"}, {int(line_range[1]) if line_range[1] else "?"}]',
        )
    else:
        start, end = line_range

        # Check start >= 1
        if start < 1:
            diagnostic.add_issue(
                "line_range",
                "INVALID",
                f"start must be >= 1, got {start}",
                str(line_range),
                suggested_fix=f"Set start to 1: [1, {end}]",
            )

        # Check end >= start
        if end < start:
            diagnostic.add_issue(
                "line_range",
                "INVALID",
                f"end must be >= start, got start={start}, end={end}",
                str(line_range),
                suggested_fix=f"Swap or fix: [{min(start, end)}, {max(start, end)}]",
            )

        # Check bounds against markdown
        if markdown_lines and end > markdown_lines:
            diagnostic.add_issue(
                "line_range",
                "OUT_OF_BOUNDS",
                f"end {end} exceeds file length {markdown_lines}",
                str(line_range),
                suggested_fix=f"Truncate to file length: [{start}, {markdown_lines}]",
            )

    # 5. Check directive_label (required if entity has label_text)
    directive_label = sl.get("directive_label")
    entity_has_label = bool(entity.get("label_text") or entity.get("label"))

    if entity_has_label and not directive_label:
        label_val = entity.get("label_text") or entity.get("label")
        diagnostic.add_issue(
            "directive_label",
            "MISSING",
            f"directive_label missing but entity has label: {label_val}",
            suggested_fix=f"Set to: {label_val}",
        )
    elif directive_label and not isinstance(directive_label, str):
        diagnostic.add_issue(
            "directive_label",
            "INVALID",
            f"Must be string, got {type(directive_label).__name__}",
            str(directive_label),
        )
    elif directive_label and not re.match(r"^[a-z][a-z0-9-]*$", directive_label):
        diagnostic.add_issue(
            "directive_label", "INVALID", "Must match pattern [a-z][a-z0-9-]*", directive_label
        )

    # 6. Check equation (optional but if present must be string or None)
    equation = sl.get("equation")
    if equation is not None and not isinstance(equation, str):
        diagnostic.add_issue(
            "equation",
            "INVALID",
            f"Must be string or None, got {type(equation).__name__}",
            str(equation),
        )

    # 7. Check url_fragment (optional but if present must be string or None)
    url_fragment = sl.get("url_fragment")
    if url_fragment is not None and not isinstance(url_fragment, str):
        diagnostic.add_issue(
            "url_fragment",
            "INVALID",
            f"Must be string or None, got {type(url_fragment).__name__}",
            str(url_fragment),
        )

    return diagnostic


def generate_comprehensive_report(
    directory: Path, markdown_path: Path | None = None, output_file: Path | None = None
):
    """Generate comprehensive diagnostic report for all files."""

    diagnostics: list[FileDiagnostic] = []

    # Process all JSON files
    for json_file in sorted(directory.rglob("*.json")):
        # Skip report files
        if "report" in json_file.name.lower():
            continue

        diagnostic = diagnose_file(json_file, markdown_path)
        diagnostics.append(diagnostic)

    # Categorize by severity
    critical = [d for d in diagnostics if d.get_severity() == "CRITICAL"]
    high = [d for d in diagnostics if d.get_severity() == "HIGH"]
    medium = [d for d in diagnostics if d.get_severity() == "MEDIUM"]
    ok = [d for d in diagnostics if d.get_severity() == "OK"]

    # Generate report
    report_lines = []

    report_lines.extend((
        "=" * 100,
        "COMPREHENSIVE SOURCE LOCATION DIAGNOSTIC REPORT",
        "=" * 100,
        "",
        f"Total files analyzed: {len(diagnostics)}",
        f"  ✓ Valid (OK):           {len(ok):3} ({100 * len(ok) / len(diagnostics):.1f}%)",
        f"  ⚠ Medium issues:        {len(medium):3} ({100 * len(medium) / len(diagnostics):.1f}%)",
        f"  ✗ High severity:        {len(high):3} ({100 * len(high) / len(diagnostics):.1f}%)",
        f"  ✗✗ CRITICAL (no line_range): {len(critical):3} ({100 * len(critical) / len(diagnostics):.1f}%)",
        "",
    ))

    # Statistics by issue type
    issue_counts = defaultdict(int)
    field_counts = defaultdict(int)
    for d in diagnostics:
        for issue in d.issues:
            issue_counts[issue.issue_type] += 1
            field_counts[issue.field] += 1

    if issue_counts:
        report_lines.extend(("─" * 100, "ISSUES BY TYPE:", "─" * 100))
        for issue_type in ["MISSING", "INVALID", "OUT_OF_BOUNDS"]:
            if issue_type in issue_counts:
                report_lines.append(f"  {issue_type:15} {issue_counts[issue_type]:3} issues")
        report_lines.extend(("", "─" * 100, "ISSUES BY FIELD:", "─" * 100))
        for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
            report_lines.append(f"  {field:20} {count:3} issues")
        report_lines.append("")

    # Detailed file-by-file report
    def print_file_section(files: list[FileDiagnostic], title: str):
        if not files:
            return

        report_lines.append("")
        report_lines.append("=" * 100)
        report_lines.append(f"{title} ({len(files)} files)")
        report_lines.append("=" * 100)
        report_lines.append("")

        for diagnostic in files[:50]:  # Limit to first 50 per category
            report_lines.append(f"File: {diagnostic.file_name}")
            if diagnostic.label:
                report_lines.append(f"  Label: {diagnostic.label}")
            if diagnostic.entity_type:
                report_lines.append(f"  Type: {diagnostic.entity_type}")
            report_lines.append("")

            for issue in diagnostic.issues:
                symbol = (
                    "✗✗"
                    if issue.issue_type == "MISSING" and issue.field == "line_range"
                    else "✗"
                    if issue.issue_type in {"MISSING", "INVALID"}
                    else "⚠"
                )

                report_lines.append(f"  {symbol} {issue.field}: {issue.issue_type}")
                report_lines.append(f"     Problem: {issue.message}")

                if issue.current_value:
                    report_lines.append(f"     Current: {issue.current_value}")

                if issue.suggested_fix:
                    report_lines.append(f"     Fix: {issue.suggested_fix}")

                report_lines.append("")

            report_lines.append("─" * 100)
            report_lines.append("")

        if len(files) > 50:
            report_lines.append(f"... and {len(files) - 50} more files in this category")
            report_lines.append("")

    # Print sections in order of severity
    print_file_section(critical, "CRITICAL: Files without line_range")
    print_file_section(high, "HIGH SEVERITY: Invalid formats")
    print_file_section(medium, "MEDIUM: Missing optional fields")

    # Summary of fixes needed
    report_lines.extend(("", "=" * 100, "SUMMARY OF FIXES NEEDED", "=" * 100, ""))

    if critical:
        report_lines.extend((
            f"1. CRITICAL: Find line_range for {len(critical)} files",
            "   Use: scripts/enrich_line_ranges_from_directives.py",
            "   Or:  scripts/aggressive_text_matching.py",
            "   Or:  scripts/handle_special_cases.py",
            "",
        ))

    if high:
        # Count § symbols
        section_with_symbol = sum(
            1 for d in high for i in d.issues if "§" in str(i.current_value or "")
        )
        if section_with_symbol:
            report_lines.extend((
                f"2. HIGH: Remove § symbols from {section_with_symbol} sections",
                "   Use: scripts/normalize_section_format.py --batch DIR",
                "",
            ))

        # Count invalid formats
        invalid_formats = sum(1 for d in high for i in d.issues if i.issue_type == "INVALID")
        if invalid_formats:
            report_lines.extend((
                f"3. HIGH: Fix {invalid_formats} invalid formats",
                "   Use: scripts/normalize_section_format.py --batch DIR",
                "",
            ))

        # Count out of bounds
        oob = sum(1 for d in high for i in d.issues if i.issue_type == "OUT_OF_BOUNDS")
        if oob:
            report_lines.extend((
                f"4. HIGH: Fix {oob} out-of-bounds line_range values",
                "   Manually correct line_range in JSON files",
                "",
            ))

    if medium:
        report_lines.extend((
            f"5. MEDIUM: Add {len(medium)} missing optional fields",
            "   Extract sections: scripts/extract_sections_from_markdown.py",
            "",
        ))

    report_lines.extend((
        "=" * 100,
        f"Valid files: {len(ok)}/{len(diagnostics)} ({100 * len(ok) / len(diagnostics):.1f}%)",
        f"Files needing fixes: {len(diagnostics) - len(ok)}/{len(diagnostics)} ({100 * (len(diagnostics) - len(ok)) / len(diagnostics):.1f}%)",
        "=" * 100,
    ))

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file if requested
    if output_file:
        output_file.write_text(report_text, encoding="utf-8")
        print(f"\n✓ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive diagnostic report for source locations"
    )

    parser.add_argument(
        "--directory", "-d", type=Path, required=True, help="Directory containing raw_data"
    )
    parser.add_argument(
        "--markdown", "-m", type=Path, help="Markdown file for line_range validation"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output file for report (default: print to console)"
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    generate_comprehensive_report(args.directory, args.markdown, args.output)


if __name__ == "__main__":
    main()
